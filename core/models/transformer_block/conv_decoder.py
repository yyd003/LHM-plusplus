# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : Conv decoder for RGB and mask prediction

"""
Convolutional decoder for image-based dense prediction (RGB + mask).

This module implements a lightweight decoder that takes patchified image features
and produces per-pixel RGB and mask predictions via a ConvHead. It avoids
heavy transformer encoders and uses a merge-and-unmerge style tokenizer for
efficient decoding.

Classes:
    ConvDecoderOnly: Decoder-only model for RGB and mask prediction from image patches.
"""
import sys

sys.path.append("./")
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from core.models.heads.conv_head import ConvHead


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    """Initializes weights for linear and embedding layers.

    Uses normal initialization with configurable standard deviation.
    Bias terms are zero-initialized when present.

    Args:
        module (nn.Module): Module to initialize (in-place).
        std (float): Standard deviation for weight initialization. Defaults to 0.02.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)


class ConvDecoderOnly(nn.Module):
    """Convolutional decoder without transformer encoder.

    Predicts RGB and mask from patchified image features using a ConvHead.
    Designed for efficient dense prediction without cross-attention or
    heavy transformer blocks.

    Shape:
        - Input query_imgs: (B, V, C, H_patch*P, W_patch*P) where P=enc_patch_size.
        - Output rgb: (B, V, H_render, W_render, 3).
        - Output mask: (B, V, H_render, W_render, 1).

    Attributes:
        enc_patch_size (int): Patch size for tokenization (e.g., 14 for DINOv2).
        patchify (nn.Sequential): Rearrange + Linear to produce patch tokens.
        dense_predictor (ConvHead): Decoder head for RGB and mask.
        accumulate (str): Accumulation strategy for multi-view (e.g., 'concat').
        activation: Output activation (e.g., sigmoid for RGB/mask in [0, 1]).
    """

    def __init__(
        self,
        enc_channels: int,
        img_dim: int,
        num_heads: int = 16,
        enc_patch_size: int = 14,
        gradient_checkpointing: bool = False,
        depth: int = 4,
        merge_ratio: float = 0.5,
        **kwargs: Any,
    ):
        """Initializes the ConvDecoderOnly.

        Args:
            enc_channels (int): Number of channels in encoder patch features.
            img_dim (int): Token dimension after patchification (d_model).
            num_heads (int): Unused; kept for API compatibility. Defaults to 16.
            enc_patch_size (int): Patch size (height/width of each patch).
                Defaults to 14.
            gradient_checkpointing (bool): Unused; kept for API compatibility.
            depth (int): Unused; kept for API compatibility.
            merge_ratio (float): Unused; kept for API compatibility.
            **kwargs: Optional arguments, including:
                - accumulate (str): Accumulation strategy. Defaults to 'concat'.
        """
        super().__init__()

        self.enc_patch_size = enc_patch_size
        self._build_convhead()

        self.accumulate = kwargs.get("accumulate", "concat")
        self.patchify = self._create_tokenizer(enc_channels, enc_patch_size, img_dim)
        self.activation = torch.sigmoid

    def _create_tokenizer(
        self, in_channels: int, patch_size: int, d_model: int
    ) -> nn.Sequential:
        """Creates patch tokenizer: Rearrange to patches + Linear projection.

        Rearranges (B*V, C, H, W) into (B*V, N_patch, P*P*C) and projects to d_model.

        Args:
            in_channels (int): Input feature channels.
            patch_size (int): Patch height and width.
            d_model (int): Output token dimension.

        Returns:
            nn.Sequential: Tokenizer module.
        """
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(
                in_channels * (patch_size**2),
                d_model,
                bias=False,
            ),
        )
        tokenizer.apply(init_weights)
        return tokenizer

    def _build_convhead(self) -> None:
        """Builds the ConvHead dense predictor for RGB and mask output."""
        self.dense_predictor = ConvHead(
            num_features=4,
            dim_in=1024,
            projects=nn.Identity(),
            dim_out=[3, 1],  # rgb + mask
            dim_proj=1024,
            dim_upsample=[256, 128, 64],
            dim_times_res_block_hidden=2,
            num_res_blocks=2,
            res_block_norm="group_norm",
            last_res_blocks=0,
            last_conv_channels=32,
            last_conv_size=1,
            using_uv=True,
        )

    def predict_rgbs(
        self,
        cond_imgs: torch.Tensor,
        query_imgs: torch.Tensor,
        motion_emb: torch.Tensor,
        render_h: int,
        render_w: int,
        pos_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts RGB and mask for a single batch item.

        Args:
            cond_imgs (torch.Tensor): Conditioning images (unused in current impl).
            query_imgs (torch.Tensor): Query image features to decode.
            motion_emb (torch.Tensor): Motion embedding (unused in current impl).
            render_h (int): Target render height.
            render_w (int): Target render width.
            pos_emb (torch.Tensor, optional): Position embedding (unused).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (rgb, mask), each of shape
                (1, V, H, W, C) with C=3 for rgb, C=1 for mask.
        """
        query_tokens = self.patchify(query_imgs.unsqueeze(0))
        patch_h = render_h // self.enc_patch_size
        patch_w = render_w // self.enc_patch_size

        outputs = self.dense_predictor(
            query_tokens,
            image=None,
            patch_h=patch_h,
            patch_w=patch_w,
        )
        rgb = self.activation(outputs[0])
        mask = self.activation(outputs[1])

        rgb = rgb.unsqueeze(0).permute(0, 1, 3, 4, 2)
        mask = mask.unsqueeze(0).permute(0, 1, 3, 4, 2)
        return rgb, mask

    def forward(
        self,
        cond_imgs_list: Optional[List[torch.Tensor]] = None,
        batch_query_imgs: Optional[torch.Tensor] = None,
        motion_emb_list: Optional[List[torch.Tensor]] = None,
        render_h: Optional[int] = None,
        render_w: Optional[int] = None,
        pos_emb_list: Optional[List[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs forward pass over a batch of samples.

        Loops over batch items and predicts RGB and mask for each, then
        concatenates results.

        Args:
            cond_imgs_list (List[torch.Tensor], optional): Per-sample conditioning
                images. Shape varies per sample.
            batch_query_imgs (torch.Tensor, optional): Batched query images.
                Shape (B, ...) with each item passed to predict_rgbs.
            motion_emb_list (List[torch.Tensor], optional): Per-sample motion
                embeddings.
            render_h (int, optional): Target render height.
            render_w (int, optional): Target render width.
            pos_emb_list (List[torch.Tensor], optional): Per-sample position
                embeddings.
            **kwargs: Additional arguments (unused).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (pred_rgbs, pred_masks), each
                concatenated over batch with shape (B, V, H, W, C).
        """
        batch = len(cond_imgs_list)
        pred_rgbs: List[torch.Tensor] = []
        pred_masks: List[torch.Tensor] = []

        for batch_id in range(batch):
            cond_imgs = cond_imgs_list[batch_id]
            pos_emb = pos_emb_list[batch_id] if pos_emb_list is not None else None
            query_imgs = batch_query_imgs[batch_id]
            motion_emb = motion_emb_list[batch_id]

            pred_rgb, pred_mask = self.predict_rgbs(
                cond_imgs, query_imgs, motion_emb, render_h, render_w, pos_emb
            )
            pred_rgbs.append(pred_rgb)
            pred_masks.append(pred_mask)

        return torch.cat(pred_rgbs, dim=0), torch.cat(pred_masks, dim=0)


if __name__ == "__main__":
    from core.structures.slidewindows import SlidingAverageTimer

    pi_model = ConvDecoderOnly(
        enc_channels=128, img_dim=1024, depth=1, enc_patch_size=7
    )
    pi_model.cuda()

    cond_imgs_list = [torch.randn(1, 4, 2565, 1024).cuda().float()]
    query_imgs = torch.randn(4, 4, 128, int(640 // 14 * 14), int(384 // 14 * 14)).cuda()
    motion_emb_list = [torch.randn(1, 1024).cuda().float()]

    slide_window = SlidingAverageTimer(window_size=10)

    for i in range(100):
        with torch.no_grad():
            slide_window.start()
            output = pi_model(
                cond_imgs_list,
                query_imgs,
                motion_emb_list,
                (640 // 14) * 14,
                (384 // 14) * 14,
            )
            print(f"output shapes: {[tuple(x.shape) for x in output]}")
            torch.cuda.synchronize()
            slide_window.record()
            print(slide_window)
