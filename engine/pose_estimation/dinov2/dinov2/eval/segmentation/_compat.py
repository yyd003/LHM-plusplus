# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0.

"""Compatibility helpers for released MMSegmentation 0.x evaluation configs."""

from copy import deepcopy

import torch
from mmengine.structures import PixelData
from mmseg.apis import inference_model as mmseg_inference_model


def _cfg_get(cfg, key, default=None):
    """Read from plain mappings and MMEngine config objects uniformly."""
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_set(cfg, key, value):
    """Write without assuming MMEngine's attribute-style config access."""
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def upgrade_test_pipeline(cfg):
    """Translate DINOv2's released 0.x test pipeline to MMSeg 1.x.

    The original configs wrap transforms in ``MultiScaleFlipAug`` and finish
    with ``ImageToTensor``/``Collect``. The current stack expects a flat
    pipeline ending in ``PackSegInputs``. TTA parameters are retained in a
    private config entry and evaluated explicitly by :func:`inference_model`.
    """
    pipeline = cfg.get("test_pipeline")
    if not pipeline:
        return cfg
    aug_index = next(
        (i for i, t in enumerate(pipeline) if t.get("type") == "MultiScaleFlipAug"),
        None,
    )
    if aug_index is None:
        return cfg

    aug = deepcopy(pipeline[aug_index])
    base_scale = tuple(aug.pop("img_scale", aug.pop("scale", (2048, 512))))
    ratios = list(aug.pop("img_ratios", [1.0]))
    allow_flip = bool(aug.pop("flip", False))
    inner = aug.pop("transforms")

    modern = [deepcopy(t) for t in pipeline[:aug_index]]
    modern.append(dict(type="Resize", scale=base_scale, keep_ratio=True))
    for transform in inner:
        transform = deepcopy(transform)
        kind = transform.get("type")
        if kind in {"Resize", "RandomFlip", "ImageToTensor", "Collect"}:
            continue
        modern.append(transform)
    modern.append(dict(type="PackSegInputs"))

    _cfg_set(cfg, "test_pipeline", modern)
    _cfg_set(cfg, "dinov2_tta", dict(
        base_scale=base_scale,
        ratios=ratios,
        allow_flip=allow_flip,
    ))
    return cfg


def _variant_pipeline(base_pipeline, scale, flip):
    pipeline = deepcopy(base_pipeline)
    resize_index = next(
        i for i, transform in enumerate(pipeline)
        if transform.get("type") == "Resize"
    )
    pipeline[resize_index]["scale"] = tuple(scale)
    if flip:
        insert_at = resize_index + 1
        while (
            insert_at < len(pipeline)
            and pipeline[insert_at].get("type") == "ResizeToMultiple"
        ):
            insert_at += 1
        pipeline.insert(
            insert_at,
            dict(type="RandomFlip", prob=1.0, direction="horizontal"),
        )
    return pipeline


def inference_model(model, image):
    """Run single- or multi-scale inference for a released DINOv2 config."""
    if isinstance(image, (list, tuple)):
        return [inference_model(model, item) for item in image]

    tta = _cfg_get(model.cfg, "dinov2_tta")
    if not tta:
        return mmseg_inference_model(model, image)

    original_pipeline = deepcopy(_cfg_get(model.cfg, "test_pipeline"))
    results = []
    base_w, base_h = _cfg_get(tta, "base_scale")
    try:
        for ratio in _cfg_get(tta, "ratios"):
            scale = (
                max(1, round(base_w * float(ratio))),
                max(1, round(base_h * float(ratio))),
            )
            flips = (False, True) if _cfg_get(tta, "allow_flip") else (False,)
            for flip in flips:
                _cfg_set(
                    model.cfg,
                    "test_pipeline",
                    _variant_pipeline(original_pipeline, scale, flip),
                )
                results.append(mmseg_inference_model(model, image))
    finally:
        _cfg_set(model.cfg, "test_pipeline", original_pipeline)

    logits = torch.stack([result.seg_logits.data for result in results]).mean(0)
    prediction = logits.argmax(dim=0, keepdim=True)
    output = results[0]
    output.seg_logits = PixelData(data=logits)
    output.pred_sem_seg = PixelData(data=prediction)
    return output
