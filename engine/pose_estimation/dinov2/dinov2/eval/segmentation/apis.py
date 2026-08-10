# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""MMEngine-compatible helpers for DINOv2 linear/multiscale segmentation."""

import itertools
import math
from copy import deepcopy
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import load_checkpoint
from ._compat import inference_model, upgrade_test_pipeline
from .models import build_model


class CenterPadding(torch.nn.Module):
    """Pad image height/width symmetrically to a patch-size multiple."""

    def __init__(self, multiple):
        super().__init__()
        if isinstance(multiple, (tuple, list)):
            if len(multiple) != 2 or multiple[0] != multiple[1]:
                raise ValueError("CenterPadding currently requires a square patch size")
            multiple = multiple[0]
        self.multiple = int(multiple)

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_left = pad_size // 2
        return pad_left, pad_size - pad_left

    @torch.inference_mode()
    def forward(self, x):
        pads = list(
            itertools.chain.from_iterable(
                self._get_pad(size) for size in x.shape[:1:-1]
            )
        )
        return F.pad(x, pads)


def _load_config(config, cfg_options=None):
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif isinstance(config, dict):
        config = Config(deepcopy(config))
    elif isinstance(config, Config):
        config = deepcopy(config)
    else:
        raise TypeError(f"config must be a path, dict or Config, got {type(config)!r}")
    if cfg_options:
        config.merge_from_dict(cfg_options)
    return config


def attach_backbone(model, backbone_model):
    """Attach a real DINOv2 backbone to the config-only MMSeg placeholder."""
    out_indices = model.cfg.model.backbone.get("out_indices")
    if out_indices is None:
        raise KeyError("model.backbone.out_indices is required for DINOv2 features")
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=out_indices,
        reshape=True,
    )
    patch_size = getattr(backbone_model, "patch_size", None)
    if patch_size is not None:
        model.backbone.register_forward_pre_hook(
            lambda _module, args: (CenterPadding(patch_size)(args[0]),)
        )
    model.dinov2_backbone = backbone_model
    return model


def init_model(
    config,
    checkpoint=None,
    device="cuda:0",
    cfg_options=None,
    backbone_model=None,
):
    """Build a DINOv2 segmentation evaluator without global registry patches."""
    cfg = upgrade_test_pipeline(_load_config(config, cfg_options))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.setdefault("data_preprocessor", dict(type="SegDataPreProcessor"))
    init_default_scope(cfg.get("default_scope", "mmseg"))
    model = build_model(cfg.model)
    model.cfg = cfg
    if backbone_model is not None:
        attach_backbone(model, backbone_model)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location="cpu")
    model.to(device).eval()
    return model
