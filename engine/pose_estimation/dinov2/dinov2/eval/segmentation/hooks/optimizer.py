# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Optimizer hook compatibility for MMEngine.

The original DINOv2 segmentation evaluation code used MMCV 1.x's
``OptimizerHook`` and global ``HOOKS`` registry.  MMCV 2.x moved runner,
registry, and optimizer functionality to MMEngine and removed
``mmcv.runner.OptimizerHook``.  This implementation keeps the legacy config
name while using MMEngine's hook and optimizer-wrapper interfaces.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class DistOptimizerHook(Hook):
    """Backward/step hook for legacy DINOv2 segmentation configs.

    New MMEngine projects should normally configure gradient accumulation and
    AMP on ``optim_wrapper`` and let ``BaseModel.train_step`` update parameters.
    This hook exists for legacy runners whose model only returns a loss.  It
    supports both an MMEngine ``runner.optim_wrapper`` and the old
    ``runner.optimizer`` attribute without importing the removed MMCV runner.
    """

    priority = "ABOVE_NORMAL"

    def __init__(
        self,
        update_interval: int = 1,
        grad_clip: Optional[dict[str, Any]] = None,
        coalesce: bool = True,
        bucket_size_mb: int = -1,
        use_fp16: bool = False,
    ) -> None:
        if update_interval < 1:
            raise ValueError("update_interval must be a positive integer")
        self.grad_clip = grad_clip
        # Kept for config compatibility. Gradient all-reduce is handled by the
        # MMEngine model wrapper rather than by the optimizer hook.
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.update_interval = update_interval
        self.use_fp16 = use_fp16
        self._grad_scaler: Optional[torch.amp.GradScaler] = None

        if use_fp16:
            # Prefer AmpOptimWrapper. The scaler below is only used by the
            # legacy raw-optimizer fallback.
            self._grad_scaler = torch.amp.GradScaler(
                "cuda", enabled=torch.cuda.is_available()
            )

    @staticmethod
    def _zero_grad(runner: Any) -> None:
        optim_wrapper = getattr(runner, "optim_wrapper", None)
        if optim_wrapper is not None:
            optim_wrapper.zero_grad()
            return
        optimizer = getattr(runner, "optimizer", None)
        if optimizer is None:
            raise AttributeError(
                "DistOptimizerHook requires runner.optim_wrapper or "
                "runner.optimizer"
            )
        optimizer.zero_grad()

    def before_run(self, runner: Any) -> None:
        """Initialize gradients for runners that still call ``before_run``."""
        self._zero_grad(runner)

    def _clip_raw_grads(self, runner: Any) -> None:
        if self.grad_clip is None:
            return
        params = [
            parameter
            for parameter in runner.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not params:
            return
        clip_cfg = dict(self.grad_clip)
        max_norm = clip_cfg.pop("max_norm")
        norm_type = clip_cfg.pop("norm_type", 2.0)
        if clip_cfg:
            warnings.warn(
                "Unsupported legacy grad_clip keys were ignored: "
                f"{sorted(clip_cfg)}",
                stacklevel=2,
            )
        torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type)

    def after_train_iter(
        self,
        runner: Any,
        batch_idx: Optional[int] = None,
        data_batch: Any = None,
        outputs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Backpropagate the current loss and update at the configured interval."""
        del batch_idx, data_batch
        outputs = outputs if outputs is not None else getattr(runner, "outputs", None)
        if not isinstance(outputs, dict) or "loss" not in outputs:
            raise KeyError("DistOptimizerHook expected an outputs dict containing 'loss'")

        loss = outputs["loss"] / self.update_interval
        optim_wrapper = getattr(runner, "optim_wrapper", None)

        if optim_wrapper is not None:
            # AmpOptimWrapper.backward performs native torch AMP scaling.  A
            # plain OptimWrapper is also valid; `use_fp16` is then only a
            # legacy config flag and cannot retrofit autocast after forward.
            if self.use_fp16 and "AmpOptimWrapper" not in type(optim_wrapper).__name__:
                warnings.warn(
                    "use_fp16=True is best paired with MMEngine "
                    "AmpOptimWrapper; continuing with the configured wrapper.",
                    stacklevel=2,
                )
            optim_wrapper.backward(loss)
            if self.every_n_train_iters(runner, self.update_interval):
                if self.grad_clip is not None:
                    # Let a wrapper configured with its own clip policy handle
                    # clipping in ``step``. Otherwise preserve the legacy hook
                    # setting, including correct AMP unscaling before clipping.
                    if getattr(optim_wrapper, "clip_grad_kwargs", None):
                        warnings.warn(
                            "Both DistOptimizerHook.grad_clip and "
                            "OptimWrapper.clip_grad are configured; the "
                            "OptimWrapper policy takes precedence.",
                            stacklevel=2,
                        )
                    else:
                        loss_scaler = getattr(optim_wrapper, "loss_scaler", None)
                        if loss_scaler is not None:
                            loss_scaler.unscale_(optim_wrapper.optimizer)
                        self._clip_raw_grads(runner)
                optim_wrapper.step()
                optim_wrapper.zero_grad()
            return

        optimizer = getattr(runner, "optimizer", None)
        if optimizer is None:
            raise AttributeError(
                "DistOptimizerHook requires runner.optim_wrapper or "
                "runner.optimizer"
            )

        if self._grad_scaler is not None and self._grad_scaler.is_enabled():
            self._grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.every_n_train_iters(runner, self.update_interval):
            if self._grad_scaler is not None and self._grad_scaler.is_enabled():
                self._grad_scaler.unscale_(optimizer)
            self._clip_raw_grads(runner)
            if self._grad_scaler is not None and self._grad_scaler.is_enabled():
                self._grad_scaler.step(optimizer)
                self._grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
