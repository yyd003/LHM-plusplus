# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-04-25 13:41:45
# @Function      : Point-Image Multi-Modality Transformer

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import logging
from diffusers.utils import is_torch_version
from einops import rearrange

from core.models.encoders.sonata import transform
from core.models.encoders.sonata.model import (
    PointEmbed,
    PointTransformerV3,
    offset2bincount,
)
from core.models.encoders.sonata.structure import Point
from core.models.transformer_block.transformer_dit import SD3PMMJointTransformerBlock
from core.models.vggt_transformer import VGGTAggregator

logger = logging.getLogger(__name__)


class PITransformerA4OBase(nn.Module):
    """
    Point-Image Multi-Modality Transformer
    Given image_list encoded by pretrained model [for example Dino-v2],
    and also PointClound, we design an efficient and faster transformer to embedding each features.
    """

    def __init__(
        self,
        point_backbone: Dict[str, Any] = None,
        image_backbone: Dict[str, Any] = None,
        gradient_checkpointing=False,
        **kwargs,
    ):
        """
        Initializes the PITransformerA4OBase module.

        Args:
            point_backbone (Dict[str, Any], optional): Configuration dictionary for the point backbone model.
            image_backbone (Dict[str, Any], optional): Configuration dictionary for the image backbone model.
            gradient_checkpointing (bool, optional): If True, enables gradient checkpointing for memory savings. Default is False.
            **kwargs: Additional keyword arguments, typically including transformer configuration details
                      such as 'img_dim', 'dim', and 'num_heads'.
        """

        super().__init__()
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
                SD3PMMJointTransformerBlock(
                    pv_dim=enc_channels[i],
                    img_dim=kwargs["img_dim"],
                    dim=kwargs["dim"],
                    num_heads=kwargs["num_heads"],
                    eps=1e-5,
                    context_pre_only=False,
                    qk_norm="rms_norm",
                    mlp_ratio=4.0,
                    require_mapping=True if len(aa_order) == 2 else False,
                )
            )

        # decode
        decoder_concat_stop_feat = point_backbone["decoder_concat_stop_feat"]
        decoder_channels = enc_channels[::-1]
        decoder_channels = np.cumsum(decoder_channels)

        for i in range(1, depth):
            if i - 1 < decoder_concat_stop_feat:
                decoder_channel = decoder_channels[i]
            self.pi_transformer_dec.append(
                SD3PMMJointTransformerBlock(
                    pv_dim=decoder_channel,
                    img_dim=kwargs["img_dim"],
                    dim=kwargs["dim"],
                    num_heads=kwargs["num_heads"],
                    eps=1e-5,
                    context_pre_only=False if i != depth - 1 else True,
                    qk_norm="rms_norm",
                    mlp_ratio=4.0,
                    require_mapping=True if len(aa_order) == 2 else False,
                )
            )

        self.outnorm = nn.LayerNorm(decoder_channel)
        self.gradient_checkpointing = gradient_checkpointing

    def build_point_transformer(self, point_backbone):
        """
        Build and initialize the point transformer backbone.

        Args:
            point_backbone (dict): Configuration dictionary for the point transformer.
                Should include the type (e.g., 'pv3' or 'pe') and all required initialization parameters
                for the corresponding point transformer class.

        Raises:
            NotImplementedError: If a specified type is not supported.
        """

        pv_type = point_backbone.pop("type", "pv3")

        if pv_type == "pv3":
            self.point_backbone = PointTransformerV3(**point_backbone)
        elif pv_type == "pe":
            self.point_backbone = PointEmbed(**point_backbone)
        else:
            raise NotImplementedError

    def build_image_transformer(self, image_backbone):
        """
        Build and initialize the image transformer backbone.

        Args:
            image_backbone (dict): Configuration dictionary for the image transformer.
                Should include the type (e.g., 'vggt') and all required initialization parameters
                for the corresponding image transformer class.

        Raises:
            NotImplementedError: If a specified type is not supported.
        """

        img_type = image_backbone.pop("type", "vggt")

        if img_type == "vggt":
            self.image_backbone = VGGTAggregator(**image_backbone)
        else:
            raise NotImplementedError

    def assert_runtime_integrity(
        self, x: torch.Tensor, cond: torch.Tensor, mod: torch.Tensor
    ):
        assert x is not None, f"Input tensor must be specified"

    @torch.no_grad()
    def get_padding_and_inverse(self, point, patch_size):
        """
        Compute the padding and its inverse for point cloud batches, so that all patches have the same number of points.

        Args:
            point: A Point structure or dictionary with an 'offset' tensor attribute (1D tensor), representing the cumulative sum of points in each batch/sample.
            patch_size (int): Desired patch size to which batches should be padded.

        Modifies:
            point (dict): Adds two new keys:
                - "pad": Indices for the padded points.
                - "unpad": Indices for mapping from padded points back to the original points.

        Returns:
            None: Updates the input `point` in-place by adding "pad" and "unpad".
        """

        pad_key = "pad"
        unpad_key = "unpad"

        offset = point.offset
        bincount = offset2bincount(offset)
        bincount_pad = (
            torch.div(
                bincount + patch_size - 1,
                patch_size,
                rounding_mode="trunc",
            )
            * patch_size
        )
        # only pad point when num of points larger than patch_size
        mask_pad = bincount > patch_size
        bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
        _offset = nn.functional.pad(offset, (1, 0))
        _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
        pad = torch.arange(_offset_pad[-1], device=offset.device)
        unpad = torch.arange(_offset[-1], device=offset.device)
        cu_seqlens = []
        for i in range(len(offset)):
            unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
            if bincount[i] != bincount_pad[i]:
                pad[
                    _offset_pad[i + 1]
                    - patch_size
                    + (bincount[i] % patch_size) : _offset_pad[i + 1]
                ] = pad[
                    _offset_pad[i + 1]
                    - 2 * patch_size
                    + (bincount[i] % patch_size) : _offset_pad[i + 1]
                    - patch_size
                ]
            pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
            cu_seqlens.append(
                torch.arange(
                    _offset_pad[i],
                    _offset_pad[i + 1],
                    step=patch_size,
                    dtype=torch.int32,
                    device=offset.device,
                )
            )
        point[pad_key] = pad
        point[unpad_key] = unpad
        cu_seqlens = nn.functional.pad(
            torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
        )

        return pad, unpad, cu_seqlens, patch_size

    def forward_multi_modality_layer(self, layer, x, cond, motion_emb=None):
        """
        Applies a multi-modality transformer layer on input tokens.

        This function forwards the point and image modality tokens (`x`, `cond`) and
        an optional `motion_emb` through the provided transformer `layer`,
        potentially using gradient checkpointing if enabled and in training mode.

        Args:
            layer (nn.Module): The transformer block to apply.
            x (Tensor): Point tokens/tensor, shape-specific to architecture (e.g., (B, N, C)).
            cond (Tensor): Image tokens/tensor, shape-specific to architecture.
            motion_emb (Tensor, optional): Optional motion embedding for conditioning.

        Returns:
            Tuple[Tensor, Tensor]:
                Updated (x, cond) tensors after passing through the transformer layer.
        """

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )

            x, cond = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer),
                x,
                cond,
                motion_emb,
                **ckpt_kwargs,
            )
        else:
            x, cond = layer(
                x,
                cond,
                motion_emb,
            )

        return x, cond

    def _merge_mv(self, mv_tokens):
        """
        Merges multi-view tokens according to the accumulation strategy.

        This function reduces the dimensionality of the multi-view tokens (`mv_tokens`)
        depending on the accumulation mode specified by `self.accumulate`.

        Args:
            mv_tokens (Tensor): Input tokens of shape (b, s, p, c) where
                b = batch size,
                s = number of views,
                p = number of points,
                c = feature dimension.

        Returns:
            Tensor: Merged tokens.
                - If accumulation is 'concat': shape (b, s*p, c).
                - If 'mean': shape (b, p, c).
        """

        if self.accumulate == "concat":
            mv_tokens = rearrange(mv_tokens, "b s p c -> b (s p) c")
        elif self.accumulate == "mean":
            mv_tokens = mv_tokens.mean(dim=1)
        else:
            raise NotImplementedError

        return mv_tokens

    def _demerge_mv(self, mv_tokens, merge_mv_tokens, mv_p):
        """
        Demerges multi-view tokens according to the accumulation strategy.

        This function reconstructs the multi-view token structure from the merged form,
        depending on the accumulation mode specified by `self.accumulate`.

        Args:
            mv_tokens (Tensor): The original tokens, potentially to be updated, shape varies.
            merge_mv_tokens (Tensor): Merged tokens to be demerged.
            mv_p (int): The number of points per view.

        Returns:
            Tensor: Demerged tokens with shape:
                - If accumulation is 'concat': (b, s, p, c)
                - If 'mean': shape as `mv_tokens`, updated in-place.
        """

        if self.accumulate == "concat":
            mv_tokens = rearrange(merge_mv_tokens, "b (s p) c -> b s p c", p=mv_p)
        elif self.accumulate == "mean":
            norm_merge_mv_tokens = merge_mv_tokens / merge_mv_tokens.norm(
                dim=-1, keepdim=True
            )
            mv_tokens += norm_merge_mv_tokens.unsqueeze(
                1
            )  # every mm-tokens has norm, so we do not requires process.
        else:
            raise NotImplementedError

        return mv_tokens

    def multi_modality_layer(
        self,
        mv_tokens,
        query_points,
        motion_emb,
        depth_i,
        mode="enc",
        use_auto_patch=True,
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
        # 1. Reshape multi-view tokens
        _, _, mv_p, _ = mv_tokens.shape

        merge_mv_tokens = self._merge_mv(mv_tokens)

        # 2. Prepare point ordering and padding
        offset = query_points.offset

        # 3. Prepare point features
        # TODO fix to 10
        auto_patch_size = 10 * mv_p

        # 10 is the expected value when sampling from [1, 2, 4, 8, 10, 12, 14, 16]
        if use_auto_patch and offset.item() > auto_patch_size:
            # TODO fix a bug for Encoder-Decoder framework,
            # we find if use auto patch, when using E-D framework,
            # the graident cannot pass, always leads to nan error.

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
        if motion_emb is None:
            raise NotImplementedError

        point_latents, merge_mv_tokens = self.forward_multi_modality_layer(
            transformer, point_latents, merge_mv_tokens, motion_emb=motion_emb
        )

        if use_auto_patch and offset.item() > auto_patch_size:
            # 6. Restore original shapes and ordering
            points_latents = point_latents.view(-1, order_feat.shape[-1])[inverse]

            if merge_mv_tokens is not None:
                mv_tokens = self._demerge_mv(mv_tokens, merge_mv_tokens, mv_p)
            else:
                mv_tokens = None

            # 7. Update query points
            query_points.feat = points_latents
            query_points.sparse_conv_feat = (
                query_points.sparse_conv_feat.replace_feature(points_latents)
            )
        else:
            # 6. Restore original shapes and ordering
            points_latents = point_latents.view(-1, order_feat.shape[-1])[inverse]

            if merge_mv_tokens is not None:
                mv_tokens = self._demerge_mv(mv_tokens, merge_mv_tokens, mv_p)
            else:
                mv_tokens = None
            # 7. Update query points
            query_points.feat = points_latents
            query_points.sparse_conv_feat = (
                query_points.sparse_conv_feat.replace_feature(points_latents)
            )

        return mv_tokens, query_points

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

        # x: [N, L, D]
        # cond: [B S, L_cond, D_cond] or None
        # mod: [N, D_mod] or None

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

        img_enc_feat_list = []

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

            img_enc_feat_list.append(img_tokens)

            # 3. modality transformer
            if self.image_backbone.aa_block_num == 2:
                concat_inter = torch.cat(
                    [frame_intermediates[-1], global_intermediates[-1]], dim=-1
                )
            else:
                concat_inter = frame_intermediates[-1]

            patch_start_idx = self.image_backbone.patch_start_idx
            mv_tokens = concat_inter[:, :, patch_start_idx:]
            mv_tokens, query_points = self.multi_modality_layer(
                mv_tokens, query_points, motion_emb, depth_i, mode="enc"
            )

            img_tokens = img_tokens.view(B, S, P, C)

        # decode
        decoder_layer_i = 0
        img_enc_feat_list = img_enc_feat_list[::-1]

        while "pooling_parent" in query_points.keys():

            dec_img_tokens = torch.cat(
                [img_tokens, img_enc_feat_list[decoder_layer_i].view(B, S, P, C)], dim=3
            )
            query_points, decoder_layer_i = self.point_backbone.decoder(
                query_points, decoder_layer_i
            )

            mv_tokens = dec_img_tokens[:, :, patch_start_idx:]

            mv_tokens, query_points = self.multi_modality_layer(
                mv_tokens, query_points, motion_emb, decoder_layer_i, mode="dec"
            )

            img_tokens = img_tokens.view(B, S, P, C)
            img_tokens = torch.cat(
                [img_tokens[:, :, :patch_start_idx], mv_tokens], dim=2
            )

        query_feats = query_points.feat[query_points.inverse]

        return dict(query_feats=query_feats.unsqueeze(0), img_feats=img_tokens)


class PITransformerA4OEncoderOnly(PITransformerA4OBase):
    """
    PITransformerA4OEncoderOnly

    This class implements the encoder-only variant of the Point-Image Multi-Modality Transformer.
    It processes input point clouds and corresponding images through multi-stage transformer blocks,
    encoding features from both modalities for downstream tasks.

    Attributes:
        point_processing: Preprocessing function for points.
        point_backbone: Backbone network for point cloud feature extraction.
        image_backbone: Backbone network for image feature extraction.
        pi_transformer_enc: nn.ModuleList of point-image encoder transformer blocks.
        pi_transformer_dec: nn.ModuleList of point-image decoder transformer blocks (unused in encoder-only mode).
        enc_patch_size: Patch sizes used for point backbone encoding.
    """

    def __init__(
        self,
        point_backbone: Dict[str, Any] = None,
        image_backbone: Dict[str, Any] = None,
        gradient_checkpointing=False,
        **kwargs,
    ):
        """
        Initializes the PITransformerA4OEncoderOnly module.

        Args:
            point_backbone (Dict[str, Any], optional): Configuration dictionary for the point backbone model.
            image_backbone (Dict[str, Any], optional): Configuration dictionary for the image backbone model.
            gradient_checkpointing (bool, optional): If True, enables gradient checkpointing for memory savings. Default is False.
            **kwargs: Additional keyword arguments, typically including transformer configuration details
                      such as 'img_dim', 'dim', and 'num_heads'.
        """

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
                SD3PMMJointTransformerBlock(
                    pv_dim=enc_channels[i],
                    img_dim=kwargs["img_dim"],
                    dim=kwargs["dim"],
                    num_heads=kwargs["num_heads"],
                    eps=1e-5,
                    context_pre_only=False,
                    qk_norm="rms_norm",
                    mlp_ratio=4.0,
                    require_mapping=True if len(aa_order) == 2 else False,
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

    def only_global_attention(self, freeze_point=True, freeze_image=True):
        """
        Freeze all attention except P-I global attention.

        Args:
            freeze_point (bool, optional): If True, freeze all parameters of the point backbone (default: True).
            freeze_image (bool, optional): If True, freeze all parameters of the image backbone (default: True).

        This function disables gradient computation for the selected backbone networks to
        ensure only the global point-image (P-I) attention modules remain trainable.
        """

        # freeze image backbone
        if freeze_image:
            logger.info("freezing image backbone")
            for param in self.image_backbone.parameters():
                param.requires_grad = False

        # freeze point backbone
        if freeze_point:
            logger.info("freezing point backbone")
            for param in self.point_backbone.parameters():
                param.requires_grad = False

    def build_decoder(self, point_backbone, image_backbone, **kwargs):
        """
        Build and initialize the decoder part of the point-image transformer.

        Args:
            point_backbone (dict): Configuration dictionary for the point backbone model.
            image_backbone (dict): Configuration dictionary for the image backbone model.
            **kwargs: Additional keyword arguments for transformer configuration such as
                'img_dim', 'dim', and 'num_heads'.

        This function initializes the decoder transformer blocks for the multi-modality model.
        Depending on 'enc_channels', it configures the appropriate channel sizes for each decoder layer,
        and sets up PI (Point-Image) joint transformer modules for each stage.
        """

        # decoder
        enc_channels = point_backbone["enc_channels"]
        depth = len(self.point_backbone.enc)
        aa_order = image_backbone["aa_order"]

        decoder_concat_stop_feat = point_backbone["decoder_concat_stop_feat"]
        decoder_channels = enc_channels[::-1]
        decoder_channels = np.cumsum(decoder_channels)

        for i in range(1, depth):
            if i - 1 < decoder_concat_stop_feat:
                decoder_channel = decoder_channels[i]
            self.pi_transformer_dec.append(
                SD3PMMJointTransformerBlock(
                    pv_dim=decoder_channel,
                    img_dim=kwargs["img_dim"],
                    dim=kwargs["dim"],
                    num_heads=kwargs["num_heads"],
                    eps=1e-5,
                    context_pre_only=False if i != depth - 1 else True,
                    qk_norm="rms_norm",
                    mlp_ratio=4.0,
                    require_mapping=True if len(aa_order) == 2 else False,
                )
            )

        self.outnorm = nn.LayerNorm(decoder_channel)

    def _forward_per_batch(self, query_points, cond_imgs, motion_emb=None, **kwargs):
        """
        Performs a forward pass for a single batch through the point-image transformer.

        Args:
            query_points (dict): Dictionary representing the query points, expected to include fields for input features and structure.
            cond_imgs (torch.Tensor): Conditional images as input, typically a tensor of shape [B, S, P, C] (batch, views, patches, channels).
            motion_emb (torch.Tensor, optional): Optional motion embedding provided for multi-modality conditioning.
            **kwargs: Arbitrary keyword arguments, may include model/hyperparameter options such as 'latent_size'.

        Returns:
            Output tensors after passing through the point backbone, image backbone,
            and multi-modality layers. The returned structure depends on the calling function
            but often includes processed point and image tokens/features.
        """

        # x: [N, L, D]
        # cond: [B S, L_cond, D_cond] or None
        # mod: [N, D_mod] or None

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
            if len(self.image_backbone.aa_order) == 2:
                mv_tokens = torch.cat(
                    [frame_intermediates[-1], global_intermediates[-1]], dim=-1
                )
            else:
                mv_tokens = frame_intermediates[-1]

            mv_tokens, query_points = self.multi_modality_layer(
                mv_tokens, query_points, motion_emb, depth_i, mode="enc"
            )
            img_tokens = mv_tokens.view(B, S, P, C)

        # decode
        decoder_layer_i = 0

        while "pooling_parent" in query_points.keys():
            query_points, decoder_layer_i = self.point_backbone.decoder(
                query_points, decoder_layer_i
            )
            mv_tokens, query_points = self.multi_modality_layer(
                img_tokens, query_points, motion_emb, decoder_layer_i, mode="dec"
            )

            if mv_tokens is not None:
                img_tokens = mv_tokens.view(B, S, P, C)

        # feat-copy
        query_points.feat = self.outnorm(query_points.feat[query_points.inverse])

        return query_points.feat.unsqueeze(0), img_tokens

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
        motion_emb_list = []
        for batch in range(batch_size):

            sample = sample_batch(query_points, batch)
            sample["mesh_meta"] = mesh_meta
            # ref_imgs_bool
            batch_ref_imgs_bool = kwargs["ref_imgs_bool"][batch]
            batch_cond_imgs = cond_imgs[batch : batch + 1]
            batch_motion_emb = motion_emb[batch : batch + 1]

            valid_cond_imgs = batch_cond_imgs[:, batch_ref_imgs_bool]
            valid_cond_motion_emb = batch_motion_emb[:, batch_ref_imgs_bool].mean(dim=1)

            query_feats, img_feats, motion_emb = self._forward_per_batch(
                sample, valid_cond_imgs, valid_cond_motion_emb, **kwargs
            )

            query_feats_list.append(query_feats)
            img_feats_list.append(img_feats)
            motion_emb_list.append(motion_emb)

        return dict(
            query_feats=torch.cat(query_feats_list, dim=0),
            img_feats=img_feats_list,
            motion_embs=motion_emb_list,
        )


class PITransformerA4OE2EEncoderDecoder(PITransformerA4OEncoderOnly):
    """
    PITransformerA4OE2EEncoderDecoder

    This class implements the end-to-end encoder-decoder variant of the Point-Image Multi-Modality Transformer.
    It processes input point clouds and corresponding images with full encoder-decoder transformer blocks
    for both modalities, enabling joint feature extraction and cross-modality reasoning.

    Inherits from:
        PITransformerA4OEncoderOnly

    Attributes:
        pi_transformer_dec (nn.ModuleList): Decoder blocks for point-image multi-modality transformer.
        outnorm (nn.LayerNorm): Output normalization for decoder's query point features.

    Methods:
        build_decoder: Constructs decoder transformer blocks for cross-modality fusion.
        _forward_per_batch: Processes a single batch of inputs through encoder and decoder pipeline.
    """

    def build_decoder(self, point_backbone, image_backbone, **kwargs):
        """
        Build and initialize the decoder part of the point-image transformer.

        Args:
            point_backbone (dict): Configuration dictionary for the point backbone model.
            image_backbone (dict): Configuration dictionary for the image backbone model.
            **kwargs: Additional keyword arguments for transformer configuration such as
                'img_dim', 'dim', and 'num_heads'.

        This function initializes the decoder transformer blocks for the multi-modality model.
        It sets up SD3PMMJointTransformerBlock modules for each decoder stage using the provided channels
        and attention settings, supporting point-image cross-modality fusion in the decoding process.
        """

        # decoder
        dec_channels = list(point_backbone["dec_channels"])[::-1]
        depth = len(self.point_backbone.dec)
        aa_order = image_backbone["aa_order"]

        for i in range(depth):

            self.pi_transformer_dec.append(
                SD3PMMJointTransformerBlock(
                    pv_dim=dec_channels[i],
                    img_dim=kwargs["img_dim"],
                    dim=kwargs["dim"],
                    num_heads=kwargs["num_heads"],
                    eps=1e-5,
                    context_pre_only=False if i != depth - 1 else True,
                    qk_norm="rms_norm",
                    mlp_ratio=4.0,
                    require_mapping=True if len(aa_order) == 2 else False,
                )
            )

        self.outnorm = nn.LayerNorm(dec_channels[-1])

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

            mv_tokens, query_points = self.multi_modality_layer(
                mv_tokens,
                query_points,
                motion_emb,
                depth_i,
                mode="enc",
                use_auto_patch=False,
            )
            img_tokens = mv_tokens.view(B, S, P, C)

        # decode
        depth = len(self.point_backbone.dec)
        for depth_i in range(depth):
            # 1. point transformer
            query_points = self.point_backbone.dec[depth_i](query_points)
            # 2. mm_transformer
            mv_tokens, query_points = self.multi_modality_layer(
                img_tokens,
                query_points,
                motion_emb,
                depth_i + 1,
                mode="dec",
                use_auto_patch=False,
            )

            if mv_tokens is not None:
                img_tokens = mv_tokens.view(B, S, P, C)

        # inverse to original size
        query_points.feat = self.outnorm(query_points.feat[query_points.inverse])

        return query_points.feat.unsqueeze(0), img_tokens


class PITransformerA4OEncoder(PITransformerA4OEncoderOnly):
    """
    PITransformerA4OEncoder

    This class extends PITransformerA4OEncoderOnly with an additional "middle" module
    which enables further point-image multi-modality transformer mixing after the encoding stage.
    The middle transformer blocks refine the joint embedding learned from both the point and image backbones,
    allowing more flexible and deeper integration of appearance and geometric cues.
    Suitable for use in encoder-only point-image multimodal architectures where
    mid-depth fusion and processing is required.

    Attributes:
        pi_transformer_middle (nn.ModuleList): The list of transformer blocks applied in the middle of the encoder.
        (Inherits all attributes from PITransformerA4OEncoderOnly.)
    """

    def __init__(
        self,
        point_backbone: Dict[str, Any] = None,
        image_backbone: Dict[str, Any] = None,
        gradient_checkpointing=False,
        **kwargs,
    ):
        """
        Initializes the PITransformerA4OEncoder module.

        This constructor extends PITransformerA4OEncoderOnly by incorporating a "middle" section,
        which consists of additional transformer blocks designed to further process and fuse
        multimodal (point-image) features after the encoding stage but before decoding.

        Args:
            point_backbone (Dict[str, Any], optional): Configuration for the point cloud transformer backbone network.
            image_backbone (Dict[str, Any], optional): Configuration for the image transformer backbone network.
            gradient_checkpointing (bool, optional): Enables gradient checkpointing for reduced memory usage. Default: False.
            **kwargs: Additional transformer-specific keyword arguments, including:
                - 'img_dim' (int): Image feature channel dimension.
                - 'dim' (int): Transformer block hidden/channel dimension.
                - 'num_heads' (int): Number of attention heads.
                - 'middle_layer' (int): The number of "middle" transformer blocks to construct.
                And potentially other custom arguments used in block construction.
        """

        # build middle
        super(PITransformerA4OEncoderOnly, self).__init__(
            point_backbone, image_backbone, gradient_checkpointing, **kwargs
        )

        self.pi_transformer_middle = nn.ModuleList()

        self.build_middle(point_backbone, image_backbone, **kwargs)

    def build_middle(self, point_backbone, image_backbone, **kwargs):
        """
        Build and initialize the "middle" transformer blocks for further multimodal fusion.

        Args:
            point_backbone (dict): Configuration dictionary for the point backbone network.
            image_backbone (dict): Configuration dictionary for the image backbone network.
            **kwargs: Additional keyword arguments for transformer block configuration, including:
                - 'img_dim' (int): Image feature channel dimensionality.
                - 'dim' (int): Hidden/channel dimension for transformer blocks.
                - 'num_heads' (int): Number of attention heads in each block.
                - 'middle_layer' (int): Number of transformer blocks to create in the 'middle' stage.

        This function populates the `pi_transformer_middle` ModuleList with a stack of
        SD3PMMJointTransformerBlock modules, using the final encoder channel size for point tokens
        and the specified image dimension, for deep feature fusing between point and image modalities
        before (optionally) entering a decoder stage.
        """

        # middle
        enc_channels = point_backbone["enc_channels"]
        depth = len(self.point_backbone.enc)
        aa_order = image_backbone["aa_order"]
        middle_layer = kwargs["middle_layer"]

        channel = enc_channels[-1]

        for i in range(0, middle_layer):
            self.pi_transformer_middle.append(
                SD3PMMJointTransformerBlock(
                    pv_dim=channel,
                    img_dim=kwargs["img_dim"],
                    dim=kwargs["dim"],
                    num_heads=kwargs["num_heads"],
                    eps=1e-5,
                    context_pre_only=False,
                    qk_norm="rms_norm",
                    mlp_ratio=4.0,
                    require_mapping=True if len(aa_order) == 2 else False,
                )
            )

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

        # x: [N, L, D]
        # cond: [B S, L_cond, D_cond] or None
        # mod: [N, D_mod] or None

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
            if len(self.image_backbone.aa_order) == 2:
                mv_tokens = torch.cat(
                    [frame_intermediates[-1], global_intermediates[-1]], dim=-1
                )
            else:
                mv_tokens = frame_intermediates[-1]

            mv_tokens, query_points = self.multi_modality_layer(
                mv_tokens, query_points, motion_emb, depth_i, mode="enc"
            )
            img_tokens = mv_tokens.view(B, S, P, C)

        # middle
        for middle_depth_i in range(len(self.pi_transformer_middle)):
            mv_tokens, query_points = self.middle_multi_modality_layer(
                img_tokens,
                query_points,
                motion_emb,
                middle_depth_i,
            )
            img_tokens = mv_tokens.view(B, S, P, C)

        # decode
        decoder_layer_i = 0
        while "pooling_parent" in query_points.keys():
            query_points, decoder_layer_i = self.point_backbone.decoder(
                query_points, decoder_layer_i
            )
            mv_tokens, query_points = self.multi_modality_layer(
                img_tokens, query_points, motion_emb, decoder_layer_i, mode="dec"
            )
            if mv_tokens is not None:
                img_tokens = mv_tokens.view(B, S, P, C)

        # feat-copy
        query_points.feat = self.outnorm(query_points.feat[query_points.inverse])

        return dict(query_feats=query_points.feat.unsqueeze(0), img_feats=img_tokens)

    def middle_multi_modality_layer(self, mv_tokens, query_points, motion_emb, depth_i):
        """Process multi-modality features with configurable encoder/decoder transformer.

        Args:
            mv_tokens:          Multi-view tokens [B, S, P, C]
            query_points:       Query points structure with features
            motion_emb:         Motion embeddings
            depth_i:            Current depth index

        Returns:
            mv_tokens:          Processed multi-view tokens [B, S, P, C]
            query_points:       Updated query points with new features
        """
        # 1. Reshape multi-view tokens
        _, _, mv_p, _ = mv_tokens.shape
        mv_tokens = rearrange(mv_tokens, "b s p c -> b (s p) c")

        # 2. Prepare point ordering and padding
        order = query_points.mm_order
        inverse = query_points.mm_inverse

        # 3. Prepare point features
        order_feat = query_points.feat[order]
        point_latents = order_feat.unsqueeze(0)

        # 4. Select appropriate transformer
        transformer = self.pi_transformer_middle[depth_i]
        # 5. Process through transformer
        if motion_emb is None:
            raise NotImplementedError

        point_latents, mv_tokens = self.forward_multi_modality_layer(
            transformer,
            point_latents,
            mv_tokens,
            motion_emb=motion_emb,
        )

        # 6. Restore original shapes and ordering
        points_latents = point_latents.view(-1, order_feat.shape[-1])[inverse]

        if mv_tokens is not None:
            mv_tokens = rearrange(mv_tokens, "b (s p) c -> b s p c", p=mv_p)

        # 7. Update query points
        query_points.feat = points_latents
        query_points.sparse_conv_feat = query_points.sparse_conv_feat.replace_feature(
            points_latents
        )

        return mv_tokens, query_points


if __name__ == "__main__":
    point_backbone = dict(
        type="pv3",
        in_channels=6,  # xyz, normal
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(1, 2),
        enc_depths=(4, 4, 4),
        enc_channels=(64, 256, 512),
        enc_num_head=(4, 8, 16),
        enc_patch_size=(4096, 4096, 2048),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        shuffle_orders=False,  # for patch interaction.
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=True,
        mask_token=False,
        enc_mode=True,
        freeze_encoder=False,
        decoder_concat_stop_feat=2,
    )

    image_backbone = dict(
        type="vggt",
        depth=2,  # depends on the number of point_backbone: (the size of stride)
        aa_order=["frame"],  # only frame attention
    )

    point_backbone = dict(
        type="pv3",
        in_channels=6,  # xyz, normal
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(1, 2, 2),
        enc_depths=(4, 4, 4, 4),
        enc_channels=(64, 128, 256, 512),
        enc_num_head=(4, 8, 16, 16),
        enc_patch_size=(4096, 4096, 2048, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        shuffle_orders=False,  # for patch interaction.
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=True,
        mask_token=False,
        enc_mode=True,
        freeze_encoder=False,
        decoder_concat_stop_feat=2,
    )

    image_backbone = dict(
        type="vggt",
        depth=3,  # depends on the number of point_backbone: (the size of stride)
        aa_order=["frame"],  # only frame attention
    )

    pi_model = PITransformerA4OEncoderOnly(
        point_backbone=point_backbone,
        image_backbone=image_backbone,
        img_dim=1024,
        dim=1024,
        num_heads=16,
        gradient_checkpointing=True,
        freeze_point=True,
        accumulate="mean",
    )

    # point_backbone = {
    #     "type": "pv3",
    #     "in_channels": 6,  # xyz, normal
    #     "order": [
    #         "z",
    #         "z-trans",
    #         "hilbert",  # mm-order
    #         "hilbert-trans"
    #     ],
    #     "stride": [
    #         1,
    #         2,
    #         2
    #     ],
    #     "enc_depths": [
    #         4,
    #         4,
    #         4,
    #         4
    #     ],
    #     "enc_channels": [
    #         64,
    #         128,
    #         256,
    #         512
    #     ],
    #     "enc_num_head": [
    #         4,
    #         8,
    #         16,
    #         16
    #     ],
    #     "enc_patch_size": [
    #         4096,
    #         4096,
    #         2048,
    #         1024
    #     ],
    #     "mlp_ratio": 4,
    #     "qkv_bias": True,
    #     "qk_scale": None,
    #     "attn_drop": 0.0,
    #     "proj_drop": 0.0,
    #     "drop_path": 0.0,
    #     "shuffle_orders": False,  # for patch interaction
    #     "pre_norm": True,
    #     "enable_rpe": False,
    #     "enable_flash": True,
    #     "upcast_attention": False,
    #     "upcast_softmax": False,
    #     "traceable": True,
    #     "mask_token": False,
    #     "enc_mode": True,
    #     "freeze_encoder": False,
    #     "decoder_concat_stop_feat": 2,
    # }

    # pi_model = PITransformerA4OEncoder(point_backbone=point_backbone, image_backbone=image_backbone, img_dim=1024, dim=1024, num_heads=16, gradient_checkpointing=True, middle_layer=5)
    pi_model.cuda()

    from core.models.encoders.sonata import transform
    from core.models.rendering.skinnings.smplx_voxel_skinning import get_query_points
    from core.models.vggt_transformer import VGGTAggregator

    points = get_query_points()

    img_latents = torch.randn(1, 1, 73 * 47, 1024).float().cuda()

    import time

    start = time.time()
    with torch.no_grad():
        out = pi_model(
            points,
            img_latents,
            motion_emb=torch.randn(1, 1, 1024).cuda(),
            latent_size=(73, 47),
            ref_imgs_bool=torch.ones(1, 1).cuda() > 0,
        )

    print(time.time() - start)
    print(out["query_feats"].shape)
