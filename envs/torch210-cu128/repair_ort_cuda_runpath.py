#!/usr/bin/env python3
"""Make ONNX Runtime GPU discover the CUDA 12 libraries already used by pt210.

ORT and PyTorch install CUDA/cuDNN shared libraries below ``site-packages/nvidia``.
ORT 1.26 can preload them from Python, but a RUNPATH also makes bare
``InferenceSession`` calls reliable and does not introduce a second CUDA runtime.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-model",
        type=Path,
        help="Optionally create and execute this ONNX model on CUDA after patching.",
    )
    args = parser.parse_args()

    import onnxruntime as ort

    package_root = Path(ort.__file__).resolve().parent
    provider = package_root / "capi" / "libonnxruntime_providers_cuda.so"
    if not provider.is_file():
        raise FileNotFoundError(f"ORT CUDA provider not found: {provider}")

    site_packages = package_root.parent
    nvidia_root = site_packages / "nvidia"
    library_dirs = sorted(path for path in nvidia_root.glob("*/lib") if path.is_dir())
    if not library_dirs:
        raise FileNotFoundError(f"No NVIDIA CUDA wheel libraries found under {nvidia_root}")

    patchelf = shutil.which("patchelf")
    if not patchelf:
        sibling = Path(sys.executable).resolve().parent / "patchelf"
        if sibling.is_file():
            patchelf = str(sibling)
    if not patchelf:
        raise RuntimeError(
            "patchelf is required; install it without dependency resolution: "
            f"{sys.executable} -m pip install --no-deps patchelf"
        )

    origin = provider.parent
    desired = ["$ORIGIN"] + [
        "$ORIGIN/" + os.path.relpath(path, origin) for path in library_dirs
    ]
    existing = [entry for entry in run(patchelf, "--print-rpath", str(provider)).split(":") if entry]
    merged = list(dict.fromkeys(desired + existing))
    subprocess.check_call([patchelf, "--set-rpath", ":".join(merged), str(provider)])

    unresolved = [
        line.strip()
        for line in run("ldd", str(provider)).splitlines()
        if "not found" in line
    ]
    if unresolved:
        raise RuntimeError("Unresolved ORT CUDA libraries:\n" + "\n".join(unresolved))

    print(f"[pt210] ORT {ort.__version__}: {provider}")
    print(f"[pt210] RUNPATH: {run(patchelf, '--print-rpath', str(provider))}")
    print("[pt210] dynamic library audit: PASS")

    if args.verify_model:
        import numpy as np

        model = args.verify_model.expanduser().resolve()
        session = ort.InferenceSession(
            str(model),
            providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
        )
        model_input = session.get_inputs()[0]
        shape = [1 if not isinstance(dim, int) else dim for dim in model_input.shape]
        outputs = session.run(None, {model_input.name: np.zeros(shape, np.float32)})
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"ORT CUDA provider did not activate: {session.get_providers()}")
        if not outputs or not all(np.isfinite(value).all() for value in outputs):
            raise RuntimeError("ORT CUDA verification produced invalid outputs")
        print(f"[pt210] CUDA inference: PASS ({model})")


if __name__ == "__main__":
    main()
