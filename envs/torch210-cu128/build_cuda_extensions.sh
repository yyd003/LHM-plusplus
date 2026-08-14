#!/usr/bin/env bash
# Build and install CUDA extensions for the active pt210 environment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-$(command -v python)}}"
MAX_JOBS="${MAX_JOBS:-4}"
if [[ -n "${BUILD_ROOT:-}" ]]; then
  mkdir -p "$BUILD_ROOT"
else
  BUILD_ROOT="$(mktemp -d -t lhmpp-pt210-build.XXXXXX)"
  trap 'rm -rf "$BUILD_ROOT"' EXIT
fi

DIFF_GAUSSIAN_COMMIT=9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0
SIMPLE_KNN_COMMIT=86710c2d4b46680c02301765dd79e465819c8f19
FLASH_ATTN_TAG=v2.8.3.post1

[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python is not executable: $PYTHON_BIN" >&2; exit 2; }
"$PYTHON_BIN" - <<'PY'
import sys, torch
assert sys.version_info[:2] == (3, 11), sys.version
assert torch.__version__ == "2.10.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
PY

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  TORCH_CUDA_ARCH_LIST="$($PYTHON_BIN - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("No NVIDIA GPU is visible; set TORCH_CUDA_ARCH_LIST explicitly for a cross-build")
major, minor = torch.cuda.get_device_capability()
print(f"{major}.{minor}")
PY
)"
fi
export TORCH_CUDA_ARCH_LIST
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-${TORCH_CUDA_ARCH_LIST//./}}"
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS//;/}"
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS// /}"

echo "Building and installing for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

if [[ ! -d "$BUILD_ROOT/diff-gaussian-rasterization/.git" ]]; then
  git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization.git \
    "$BUILD_ROOT/diff-gaussian-rasterization"
fi
git -C "$BUILD_ROOT/diff-gaussian-rasterization" fetch origin "$DIFF_GAUSSIAN_COMMIT"
git -C "$BUILD_ROOT/diff-gaussian-rasterization" checkout --detach "$DIFF_GAUSSIAN_COMMIT"
git -C "$BUILD_ROOT/diff-gaussian-rasterization" submodule update --init --recursive
MAX_JOBS="$MAX_JOBS" "$PYTHON_BIN" -m pip install --force-reinstall \
  --no-deps --no-build-isolation "$BUILD_ROOT/diff-gaussian-rasterization"

if [[ ! -d "$BUILD_ROOT/simple-knn/.git" ]]; then
  git clone --recursive https://github.com/camenduru/simple-knn.git "$BUILD_ROOT/simple-knn"
fi
git -C "$BUILD_ROOT/simple-knn" fetch origin "$SIMPLE_KNN_COMMIT"
git -C "$BUILD_ROOT/simple-knn" checkout --detach "$SIMPLE_KNN_COMMIT"
MAX_JOBS="$MAX_JOBS" "$PYTHON_BIN" -m pip install --force-reinstall \
  --no-deps --no-build-isolation "$BUILD_ROOT/simple-knn"

if [[ ! -d "$BUILD_ROOT/flash-attention/.git" ]]; then
  git clone --recursive https://github.com/Dao-AILab/flash-attention.git "$BUILD_ROOT/flash-attention"
fi
git -C "$BUILD_ROOT/flash-attention" fetch origin tag "$FLASH_ATTN_TAG"
git -C "$BUILD_ROOT/flash-attention" checkout --detach "$FLASH_ATTN_TAG"
git -C "$BUILD_ROOT/flash-attention" submodule update --init --recursive
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS="$FLASH_ATTN_CUDA_ARCHS" \
NVCC_THREADS="${NVCC_THREADS:-1}" MAX_JOBS="$MAX_JOBS" \
  "$PYTHON_BIN" -m pip install --force-reinstall \
    --no-deps --no-build-isolation "$BUILD_ROOT/flash-attention"

MAX_JOBS="$MAX_JOBS" "$PYTHON_BIN" -m pip install --force-reinstall \
  --no-deps --no-build-isolation "$REPO_ROOT/lib/pointops"
"$PYTHON_BIN" -m pip install --no-deps gsplat==1.5.3 einops==0.8.2

echo "CUDA extensions were compiled from source and installed into: $($PYTHON_BIN -c 'import sys; print(sys.prefix)')"
