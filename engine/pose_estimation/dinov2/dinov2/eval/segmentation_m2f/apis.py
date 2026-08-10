# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""MMEngine-compatible initialization for DINOv2's vendored Mask2Former."""

from copy import deepcopy
from pathlib import Path

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import load_checkpoint
from ..segmentation._compat import inference_model, upgrade_test_pipeline

from .models import DINO_MODELS, scope_dino_types


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


def init_model(config, checkpoint=None, device="cuda:0", cfg_options=None):
    """Build DINO Mask2Former through its isolated child registry."""
    cfg = upgrade_test_pipeline(_load_config(config, cfg_options))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.setdefault("data_preprocessor", dict(type="SegDataPreProcessor"))
    init_default_scope(cfg.get("default_scope", "mmseg"))
    model = DINO_MODELS.build(scope_dino_types(cfg.model))
    model.cfg = cfg
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location="cpu")
    model.to(device).eval()
    return model
