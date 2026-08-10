# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pdb

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class Dinov2Wrapper(nn.Module):
    """
    Dino v2 wrapper using original implementation, hacked with modulation.
    """

    def __init__(
        self,
        model_name: str,
        modulation_dim: int = None,
        freeze: bool = True,
        encoder_feat_dim: int = 384,
        antialias=True,
        **encoder_params,
    ):
        super().__init__()

        self.modulation_dim = modulation_dim
        self.model = self._build_dinov2(model_name, modulation_dim=modulation_dim)
        self.antialias = antialias
        self.downsample_ratio = 14
        if freeze:
            if modulation_dim is not None:
                raise ValueError(
                    "Modulated Dinov2 requires training, freezing is not allowed."
                )
            self._freeze()

    def _freeze(self):
        logger.warning(f"======== Freezing Dinov2Wrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @staticmethod
    def _build_dinov2(
        model_name: str, modulation_dim: int = None, pretrained: bool = True
    ):
        from importlib import import_module

        dinov2_hub = import_module(".dinov2.hub.backbones", package=__package__)
        model_fn = getattr(dinov2_hub, model_name)
        logger.debug(f"Modulation dim for Dinov2 is {modulation_dim}.")
        model = model_fn(modulation_dim=modulation_dim, pretrained=pretrained)
        return model

    @torch.compile
    def forward(self, image: torch.Tensor, mod: torch.Tensor = None):
        # image: [N, C, H, W]
        # mod: [N, D] or None
        # RGB image with [0,1] scale and properly sized

        # resize image to dino input
        image = self._preprocess_image(image, downsample_ratio=self.downsample_ratio)

        if self.modulation_dim is None:
            assert mod is None, "Unexpected modulation input in dinov2 forward."
            outs = self.model(image, is_training=True)
        else:
            assert (
                mod is not None
            ), "Modulation input is required in modulated dinov2 forward."
            outs = self.model(image, mod=mod, is_training=True)

        ret = torch.cat(
            [
                outs["x_norm_clstoken"].unsqueeze(1),
                outs["x_norm_patchtokens"],
            ],
            dim=1,
        )

        return ret

    def _preprocess_image(
        self,
        image: torch.tensor,
        downsample_ratio: int = 14,
    ) -> torch.Tensor:

        _, __, H, W = image.shape

        new_H = (H // downsample_ratio) * downsample_ratio
        new_W = (W // downsample_ratio) * downsample_ratio

        image = kornia.geometry.resize(
            image,
            (new_H, new_W),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )

        return image

    def _forward_class_tokens(self, image, mod=None):

        # DINO training-space
        image = kornia.geometry.resize(
            image,
            (518, 518),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )

        if self.modulation_dim is None:
            assert mod is None, "Unexpected modulation input in dinov2 forward."
            outs = self.model(image, is_training=True)
        else:
            assert (
                mod is not None
            ), "Modulation input is required in modulated dinov2 forward."
            outs = self.model(image, mod=mod, is_training=True)

        ret = outs["x_norm_clstoken"].unsqueeze(1)

        return ret
