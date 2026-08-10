# Copyright (c) SenseTime Research. All rights reserved.
# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
"""Optional TensorFlow 1.x backend used by legacy StyleGAN checkpoints.

The active LHM++ PyTorch pipeline does not need this module.  The original
implementation depends on ``tensorflow.contrib``, which only exists in
TensorFlow 1.x; TensorFlow 1.x has no supported Python 3.11 build.  Keep the
namespace importable so PyTorch-only code can inspect legacy files, but fail
with a precise compatibility message if a TensorFlow operation is requested.
"""

from __future__ import annotations

_TF1_COMPATIBILITY_ERROR = (
    "dnnlib.tflib requires TensorFlow 1.x with tensorflow.contrib. "
    "TensorFlow 1.x is not available for the pt210 Python 3.11 environment. "
    "The LHM++ PyTorch inference path does not require this backend. Convert "
    "legacy TensorFlow StyleGAN checkpoints with a separate legacy conversion "
    "environment before using them here."
)

try:
    import tensorflow as _tensorflow
except ModuleNotFoundError:
    _tensorflow = None

_TF1_AVAILABLE = bool(
    _tensorflow is not None
    and str(getattr(_tensorflow, "__version__", "")).startswith("1.")
    and hasattr(_tensorflow, "contrib")
)


def is_available() -> bool:
    """Return whether the required TensorFlow 1.x runtime is available."""

    return _TF1_AVAILABLE


def require_tensorflow1() -> None:
    """Raise a clear error when the optional TensorFlow 1.x backend is absent."""

    if not _TF1_AVAILABLE:
        raise RuntimeError(_TF1_COMPATIBILITY_ERROR)


if _TF1_AVAILABLE:
    from . import autosummary, custom_ops, network, optimizer, tfutil
    from .custom_ops import get_plugin
    from .network import Network
    from .optimizer import Optimizer
    from .tfutil import *  # noqa: F403
else:
    def __getattr__(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        raise RuntimeError(f"Cannot access dnnlib.tflib.{name}: {_TF1_COMPATIBILITY_ERROR}")
