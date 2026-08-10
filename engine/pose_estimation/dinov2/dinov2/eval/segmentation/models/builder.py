# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Private model registry for DINOv2's MMSeg evaluation components.

The historical DINOv2 evaluation code defines generic names such as ``BNHead``
and ``DinoVisionTransformer``. Keeping them in a child registry prevents a
vendored evaluator from changing Sapiens/MMseg process-wide registrations.
"""

from copy import deepcopy

from mmengine.registry import MODELS as ROOT_MODELS, Registry

DINO_SEG_MODELS = Registry(
    "dinov2_seg_model", parent=ROOT_MODELS, scope="dinov2_seg"
)


def scope_dino_types(cfg):
    """Deep-copy a config and qualify types owned by this evaluator."""
    cfg = deepcopy(cfg)

    def _scope(value):
        if isinstance(value, dict):
            for key, item in list(value.items()):
                value[key] = _scope(item)
            type_name = value.get("type")
            if (
                isinstance(type_name, str)
                and "." not in type_name
                and type_name in DINO_SEG_MODELS.module_dict
            ):
                value["type"] = f"dinov2_seg.{type_name}"
        elif isinstance(value, list):
            value = [_scope(item) for item in value]
        elif isinstance(value, tuple):
            value = tuple(_scope(item) for item in value)
        return value

    return _scope(cfg)


def build_model(cfg, default_args=None):
    return DINO_SEG_MODELS.build(scope_dino_types(cfg), default_args=default_args)
