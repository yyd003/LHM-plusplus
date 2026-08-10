# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-07-17 23:07:39
# @Function      : efficient point-image transformer block

import pdb
import sys

sys.path.append("./")
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.utils import is_torch_version
from einops import rearrange

from core.models.auto_PI_transformer import (
    PITransformerA4OBase,
    PITransformerA4OE2EEncoderDecoder,
    PITransformerA4OEncoder,
    PITransformerA4OEncoderOnly,
)
from core.models.encoders.sonata import transform
from core.models.encoders.sonata.model import (
    PointEmbed,
    PointTransformerV3,
    offset2bincount,
)
from core.models.encoders.sonata.structure import Point
from core.models.ImageToME.merge import (
    bipartite_soft_matching,
    merge_source,
    merge_wavg,
)
from core.models.transformer_block.transformer_dit import (
    SD3PMMJointTransformerBlock,
    SD3PMMJointTransformerToMEBlock,
    SD3PMMJointTransformerToMEViewBlock,
)
from core.models.vggt_transformer import VGGTAggregator, slice_expand_and_flatten


class EfficientPITransformerA4OE2EEncoderDecoder(PITransformerA4OE2EEncoderDecoder):
    def __init__(
        self,
        point_backbone: Dict[str, Any] = None,
        image_backbone: Dict[str, Any] = None,
        gradient_checkpointing=False,
        **kwargs,
    ):
        super(PITransformerA4OBase, self).__init__()

        self.point_processing = transform.default()
        # 1. build point transformer
        self.build_point_transformer(point_backbone)
        # 2. image transformer
        self.build_image_transformer(image_backbone)

        depth = len(self.point_backbone.enc)
        assert depth - 1 == self.image_backbone.aa_block_num

        # 3. build P-I transformer
        # point-image multi-modality transformer
        self.pi_transformer_enc = nn.ModuleList()
        self.pi_transformer_dec = nn.ModuleList()

        enc_channels = point_backbone["enc_channels"]
        self.enc_patch_size = point_backbone["enc_patch_size"]
        aa_order = image_backbone["aa_order"]

        # enc
        for i in range(1, depth):
            self.pi_transformer_enc.append(
                SD3PMMJointTransformerToMEViewBlock(
                    pv_dim=enc_channels[i],
                    img_dim=kwargs["img_dim"],
                    dim=kwargs["dim"],
                    num_heads=kwargs["num_heads"],
                    eps=1e-5,
                    context_pre_only=False,
                    qk_norm="rms_norm",
                    mlp_ratio=4.0,
                    require_mapping=True if len(aa_order) == 2 else False,
                    require_merge=True,
                )
            )
        # dec
        self.build_decoder(point_backbone, image_backbone, **kwargs)
        self.gradient_checkpointing = gradient_checkpointing

        # freeze model
        freeze_point = kwargs.get("freeze_point", True)
        freeze_image = kwargs.get("freeze_image", True)
        self.only_global_attention(freeze_point, freeze_image)

        # accumulate methods
        self.accumulate = kwargs.get("accumulate", "concat")

    def forward(
        self,
        query_points: dict,
        cond_imgs: torch.Tensor = None,
        motion_emb: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        """
        Forward pass of the transformer model.
        Args:
            query_points (torch.Tensor): Input tensor of shape [N, L, D].
            cond_imgs (torch.Tensor, optional):  [B S P C] from pretrained_models
            motion_emb (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.  # For SD3_MM_Cond, temb means MotionCLIP
        Returns:
            torch.Tensor: Output tensor of shape [N, L, D].
        """

        def sample_batch(query_points, batch):
            batch_sample = dict()
            for key in query_points.keys():
                if key == "mesh_meta":
                    continue
                batch_sample[key] = query_points[key][batch : batch + 1]

            return batch_sample

        batch_size = cond_imgs.shape[0]
        mesh_meta = query_points["mesh_meta"]

        query_feats_list = []
        img_feats_list = []

        for batch in range(batch_size):

            sample = sample_batch(query_points, batch)
            sample["mesh_meta"] = mesh_meta
            # ref_imgs_bool
            batch_ref_imgs_bool = kwargs["ref_imgs_bool"][batch]
            batch_cond_imgs = cond_imgs[batch : batch + 1]
            batch_motion_emb = motion_emb[batch : batch + 1]

            valid_cond_imgs = batch_cond_imgs[:, batch_ref_imgs_bool]
            valid_cond_motion_emb = batch_motion_emb[:, batch_ref_imgs_bool].mean(dim=1)

            query_feats, img_feats = self._forward_per_batch(
                sample, valid_cond_imgs, valid_cond_motion_emb, **kwargs
            )

            query_feats_list.append(query_feats)
            img_feats_list.append(img_feats)

        return dict(
            query_feats=torch.cat(query_feats_list, dim=0),
            # img_feats=torch.cat(img_feats_list, dim=0),
            img_feats=img_feats_list,
        )

    def build_decoder(self, point_backbone, image_backbone, **kwargs):
        # decode

        dec_channels = list(point_backbone["dec_channels"])[::-1]
        depth = len(self.point_backbone.dec)
        aa_order = image_backbone["aa_order"]

        for i in range(depth):
            self.pi_transformer_dec.append(
                SD3PMMJointTransformerToMEBlock(
                    pv_dim=dec_channels[i],
                    img_dim=kwargs["img_dim"],
                    dim=kwargs["dim"],
                    num_heads=kwargs["num_heads"],
                    eps=1e-5,
                    context_pre_only=False if i != depth - 1 else True,
                    qk_norm="rms_norm",
                    mlp_ratio=4.0,
                    require_mapping=True if len(aa_order) == 2 else False,
                    require_merge=True,
                    merge_r=0.36,  ## q + q^2 = 0.5; q -> 0.36
                )
            )

        self.outnorm = nn.LayerNorm(dec_channels[-1])

    def multi_modality_layer(
        self,
        mv_tokens,
        query_points,
        motion_emb,
        depth_i,
        mode="enc",
        use_auto_patch=True,
        img_tokens_size=None,
        pos=None,
    ):
        """Process multi-modality features with configurable encoder/decoder transformer.
        Args:
            mv_tokens:          Multi-view tokens [B, S, P, C]
            query_points:       Query points structure with features
            motion_emb:         Motion embeddings
            depth_i:            Current depth index
            mode:               Processing mode: "enc" for encoder or "dec" for decoder

        Returns:
            mv_tokens:          Processed multi-view tokens [B, S, P, C]
            query_points:       Updated query points with new features
        """

        if motion_emb is None:
            raise ValueError("motion_emb cannot be None for multi-modality processing")

        # 1. Reshape multi-view tokens
        mv_shape = mv_tokens.shape

        mv_p = mv_shape[-2] if mv_tokens.dim() == 4 else mv_shape[-1]
        mv_views = mv_shape[1] if mv_tokens.dim() == 4 else 1

        merge_mv_tokens, merge_mv_tokens_size = self._merge_mv(
            mv_tokens, img_tokens_size, mode, depth_i
        )

        # 2. Prepare point ordering and padding
        offset = query_points.offset

        # 3. Prepare point features
        # TODO fix to 10
        auto_patch_size = 10 * mv_p

        if (
            use_auto_patch and offset.item() > auto_patch_size
        ):  # 10 is the expected value when sampling from [1, 2, 4, 8 ,10, 12, 14, 16]
            # TODO fix a bug for Encoder-Decoder framework, we find if use auto patch, when using E-D framework, the graident cannot pass, always leads to nan error.
            pad, unpad, _, patch_size = self.get_padding_and_inverse(
                query_points, auto_patch_size
            )
            order = query_points.mm_order[pad]
            inverse = unpad[query_points.mm_inverse]
            order_feat = query_points.feat[order]
            point_latents = order_feat.view(-1, patch_size, order_feat.shape[-1])
        else:
            order = query_points.mm_order
            inverse = query_points.mm_inverse
            order_feat = query_points.feat[order]
            point_latents = order_feat.unsqueeze(0)

        # 4. Select appropriate transformer
        transformer = (
            self.pi_transformer_enc[depth_i - 1]
            if mode == "enc"
            else self.pi_transformer_dec[depth_i - 1]
        )

        # 5. Process through transformer
        point_latents, merge_mv_tokens, img_tokens_size, pos = (
            self.forward_multi_modality_layer(
                transformer,
                point_latents,
                merge_mv_tokens,
                motion_emb=motion_emb,
                img_tokens_size=img_tokens_size,
                views=mv_views,
                pos=pos,
            )
        )

        # 6. demerge tokens
        if merge_mv_tokens is not None:
            mv_tokens = self._demerge_mv(
                mv_tokens, merge_mv_tokens, mv_views, img_tokens_size, mode, depth_i
            )
        else:
            mv_tokens = None

        # 7. Update query points
        points_latents = point_latents.view(-1, order_feat.shape[-1])[inverse]
        query_points.feat = points_latents
        query_points.sparse_conv_feat = query_points.sparse_conv_feat.replace_feature(
            points_latents
        )

        return mv_tokens, query_points, img_tokens_size, pos

    def _forward_per_batch(self, query_points, cond_imgs, motion_emb=None, **kwargs):
        """
        Forward pass of the transformer model.
        Args:
            query_points (torch.Tensor): Input tensor of shape [N, L, D].
            cond_imgs (torch.Tensor, optional):  [B S P C] from pretrained_models
            motion_emb (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.  # For SD3_MM_Cond, temb means MotionCLIP
        Returns:
            torch.Tensor: Output tensor of shape [N, L, D].
        """

        # x: [N, L, D]
        # cond: [B S, L_cond, D_cond] or None
        # mod: [N, D_mod] or None

        img_tokens_size = None

        query_points = self.point_processing(query_points)
        query_points = Point(query_points)
        query_points.cuda()
        query_points = self.point_backbone.serialization_sparisy(query_points)

        # encode first
        query_points = self.point_backbone.enc[0](query_points)

        # prepare for image tokens
        B, S, P, C = cond_imgs.shape

        img_tokens, pos = self.image_backbone.prepare_image_tokens(
            cond_imgs, latent_size=kwargs["latent_size"]
        )
        _, P, C = img_tokens.shape

        frame_idx = 0
        global_idx = 0

        depth = len(self.point_backbone.enc)

        # encode
        for depth_i in range(1, depth):
            # 1. point transformer
            query_points = self.point_backbone.enc[depth_i](query_points)

            # 2. vggt transformer
            for attn_type in self.image_backbone.aa_order:

                if attn_type == "frame":
                    img_tokens, frame_idx, frame_intermediates = (
                        self.image_backbone._process_frame_attention(
                            img_tokens, B, S, P, C, frame_idx, pos=pos
                        )
                    )
                elif attn_type == "global":
                    img_tokens, global_idx, global_intermediates = (
                        self.image_backbone._process_global_attention(
                            img_tokens, B, S, P, C, global_idx, pos=pos
                        )
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            # 3. modality transformer
            if self.image_backbone.aa_block_num == 2:
                mv_tokens = torch.cat(
                    [frame_intermediates[-1], global_intermediates[-1]], dim=-1
                )
            else:
                mv_tokens = frame_intermediates[-1]

            mv_tokens, query_points, img_tokens_size, pos = self.multi_modality_layer(
                mv_tokens,
                query_points,
                motion_emb,
                depth_i,
                mode="enc",
                use_auto_patch=False,
                img_tokens_size=img_tokens_size,
                pos=pos,
            )
            B, S, P, C = mv_tokens.shape
            img_tokens = mv_tokens

        # decode
        depth = len(self.point_backbone.dec)
        img_tokens = rearrange(img_tokens, "b s p c -> b (s p) c")

        for depth_i in range(depth):
            # 1. point transformer
            query_points = self.point_backbone.dec[depth_i](query_points)
            # 2. mm_transformer
            mv_tokens, query_points, img_tokens_size, pos = self.multi_modality_layer(
                img_tokens,
                query_points,
                motion_emb,
                depth_i + 1,
                mode="dec",
                use_auto_patch=False,
                img_tokens_size=img_tokens_size,
                pos=None,
            )

            if mv_tokens is not None:
                img_tokens = mv_tokens

        print(img_tokens.shape)
        # inverse to  original size
        query_points.feat = self.outnorm(query_points.feat[query_points.inverse])

        return query_points.feat.unsqueeze(0), img_tokens

    def _merge_mv(self, mv_tokens, img_tokens_size=None, mode="enc", depth_i=0):
        """
        Merge multi-view tokens by concatenating the sequence and patch dimensions
        when in encoder mode and accumulation strategy is 'concat'.

        Args:
            mv_tokens (Tensor): Input tokens of shape (b, s, p, c)
            img_tokens_size (Optional): Ignored, kept for interface compatibility
            mode (str): Operation mode, e.g., 'enc' for encoder
            depth_i (int): Depth index, used for potential future logic

        Returns:
            Tuple[Tensor, Tensor]:
                - Merged tokens of shape (b, s*p, c)
                - Merge size mask of shape (b, s*p, 1), all ones
        """
        # Only 'concat' accumulation is supported
        if self.accumulate != "concat":
            raise NotImplementedError("Only 'concat' accumulation is supported")

        # In encoder mode, merge the sequence and patch dimensions
        if mode == "enc":
            mv_tokens = rearrange(mv_tokens, "b s p c -> b (s p) c")

        # Create a merge size indicator tensor (used for tracking token merging)
        # Shape: (b, seq_len, 1), same device and dtype as input
        merge_size = torch.ones(
            mv_tokens.shape[0],
            mv_tokens.shape[1],
            1,
            device=mv_tokens.device,
            dtype=mv_tokens.dtype,
        )

        return mv_tokens, merge_size

    def _demerge_mv(
        self,
        mv_tokens,
        merge_mv_tokens,
        mv_views,
        img_tokens_size=None,
        mode="enc",
        depth_i=0,
    ):
        """
        Reverse the merging of multi-view tokens by reshaping back to original
        multi-view layout when in encoder mode and using 'concat' accumulation.

        Args:
            mv_tokens (Tensor): Placeholder for output (not used when merging)
            merge_mv_tokens (Tensor): Merged tokens of shape (b, s*p, c)
            mv_p (int): Number of patches per view (used to unflatten)
            img_tokens_size (Optional): Ignored, kept for interface compatibility
            mode (str): Operation mode, e.g., 'enc' for encoder
            depth_i (int): Depth index, used for potential future logic

        Returns:
            Tensor: Demerged tokens of shape (b, s, p, c)
        """
        # Only 'concat' accumulation is supported
        if self.accumulate != "concat":
            raise NotImplementedError("Only 'concat' accumulation is supported")

        # Only demerge in encoder mode; otherwise, keep as-is
        if mode == "enc":
            mv_tokens = rearrange(merge_mv_tokens, "b (s p) c -> b s p c", s=mv_views)
        else:
            mv_tokens = merge_mv_tokens

        return mv_tokens

    def forward_multi_modality_layer(
        self,
        layer,
        x,
        cond,
        motion_emb=None,
        attn_masks=None,
        img_tokens_size=None,
        views=None,
        pos=None,
    ):

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )

            x, cond, img_tokens_size, pos = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer),
                x,
                cond,
                motion_emb,
                attn_masks,
                img_tokens_size,
                views,
                pos,
                **ckpt_kwargs,
            )
        else:
            x, cond, img_tokens_size, pos = layer(
                x,
                cond,
                motion_emb,
                attn_masks,
                img_tokens_size,
                views,
                pos,
            )

        return x, cond, img_tokens_size, pos
