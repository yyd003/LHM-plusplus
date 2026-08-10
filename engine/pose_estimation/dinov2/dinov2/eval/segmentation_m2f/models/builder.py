# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Registries isolated from the process-wide Sapiens/MMEngine registries.

DINOv2's vendored Mask2Former classes intentionally reuse historical names
such as ``Mask2FormerHead`` and ``CrossEntropyLoss``.  Registering them into
MMSegmentation 1.x's global ``MODELS`` registry would overwrite Sapiens classes
and make the shared ``pt210`` environment import-order dependent.  A child
registry keeps DINOv2 overrides local while falling back to modern MMSeg models
for standard layers.
"""

from mmengine.registry import MODELS as ROOT_MODELS, Registry

DINO_MODELS = Registry(
    "dinov2_m2f_model", parent=ROOT_MODELS, scope="dinov2_m2f"
)
TRANSFORMER = Registry("Transformer")
MASK_ASSIGNERS = Registry("mask_assigner")
MATCH_COST = Registry("match_cost")


def build_match_cost(cfg):
    """Build Match Cost."""
    return MATCH_COST.build(cfg)


def build_assigner(cfg):
    """Build Assigner."""
    return MASK_ASSIGNERS.build(cfg)


def build_transformer(cfg):
    """Build Transformer."""
    return TRANSFORMER.build(cfg)


def build_model(cfg, default_args=None):
    """Build a DINOv2 Mask2Former component without mutating global registries."""
    return DINO_MODELS.build(cfg, default_args=default_args)


def scope_dino_types(cfg):
    """Qualify local type names before a parent MMEngine builder sees them.

    MMCV transformer layers recursively build nested attention/FFN configs via
    the process-wide root registry.  Prefixing only DINO-owned names routes
    those nested builds back to this child registry while standard MMCV/MMseg
    types continue to resolve from the parent.
    """
    from copy import deepcopy

    cfg = deepcopy(cfg)

    def _scope(value):
        if isinstance(value, dict):
            for key, item in list(value.items()):
                value[key] = _scope(item)
                # The released ADE20K Mask2Former config omits ``type`` only
                # for decoder ``ffn_cfgs``. MMCV 2.x otherwise selects its
                # global FFN, which has no DINO ``with_cp`` argument.
                if key == "ffn_cfgs" and isinstance(value[key], dict):
                    value[key].setdefault("type", "dinov2_m2f.FFN")
            type_name = value.get("type")
            if (
                isinstance(type_name, str)
                and "." not in type_name
                and type_name in DINO_MODELS.module_dict
            ):
                value["type"] = f"dinov2_m2f.{type_name}"
        elif isinstance(value, list):
            value = [_scope(item) for item in value]
        elif isinstance(value, tuple):
            value = tuple(_scope(item) for item in value)
        return value

    return _scope(cfg)
