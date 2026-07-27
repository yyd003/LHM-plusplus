#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE_ROOT=$(cd "$REPO_ROOT/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/dreams/.conda/envs/lhmpp-pt210-cu128/bin/python}
BUILD_ROOT=${BUILD_ROOT:-$WORKSPACE_ROOT/.build/pt210-cu128}
WHEELHOUSE_ROOT=${WHEELHOUSE_ROOT:-$WORKSPACE_ROOT/wheelhouse/pt210-cu128}

DIFF_GAUSSIAN_COMMIT=9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0
SIMPLE_KNN_COMMIT=86710c2d4b46680c02301765dd79e465819c8f19
FLASH_ATTN_TAG=v2.8.3.post1

mkdir -p "$BUILD_ROOT" "$WHEELHOUSE_ROOT"

if [[ ! -d "$BUILD_ROOT/diff-gaussian-rasterization/.git" ]]; then
  git clone --recursive \
    https://github.com/graphdeco-inria/diff-gaussian-rasterization.git \
    "$BUILD_ROOT/diff-gaussian-rasterization"
fi
git -C "$BUILD_ROOT/diff-gaussian-rasterization" fetch origin "$DIFF_GAUSSIAN_COMMIT"
git -C "$BUILD_ROOT/diff-gaussian-rasterization" checkout "$DIFF_GAUSSIAN_COMMIT"
git -C "$BUILD_ROOT/diff-gaussian-rasterization" submodule update --init --recursive
TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 "$PYTHON_BIN" -m pip wheel \
  --no-deps --no-build-isolation \
  --wheel-dir "$WHEELHOUSE_ROOT" \
  "$BUILD_ROOT/diff-gaussian-rasterization"

if [[ ! -d "$BUILD_ROOT/simple-knn/.git" ]]; then
  git clone --recursive https://github.com/camenduru/simple-knn.git \
    "$BUILD_ROOT/simple-knn"
fi
git -C "$BUILD_ROOT/simple-knn" fetch origin "$SIMPLE_KNN_COMMIT"
git -C "$BUILD_ROOT/simple-knn" checkout "$SIMPLE_KNN_COMMIT"
TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 "$PYTHON_BIN" -m pip wheel \
  --no-deps --no-build-isolation \
  --wheel-dir "$WHEELHOUSE_ROOT" \
  "$BUILD_ROOT/simple-knn"

if [[ ! -d "$BUILD_ROOT/flash-attention/.git" ]]; then
  git clone --recursive --branch "$FLASH_ATTN_TAG" \
    https://github.com/Dao-AILab/flash-attention.git \
    "$BUILD_ROOT/flash-attention"
fi
git -C "$BUILD_ROOT/flash-attention" fetch origin tag "$FLASH_ATTN_TAG"
git -C "$BUILD_ROOT/flash-attention" checkout "$FLASH_ATTN_TAG"
git -C "$BUILD_ROOT/flash-attention" submodule update --init --recursive
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTN_CUDA_ARCHS=120 \
NVCC_THREADS=1 \
MAX_JOBS=6 \
  "$PYTHON_BIN" -m pip wheel \
    --no-deps --no-build-isolation \
    --wheel-dir "$WHEELHOUSE_ROOT" \
    "$BUILD_ROOT/flash-attention"

"$PYTHON_BIN" -m pip install --force-reinstall --no-deps \
  "$WHEELHOUSE_ROOT"/diff_gaussian_rasterization-*.whl \
  "$WHEELHOUSE_ROOT"/simple_knn-*.whl \
  "$WHEELHOUSE_ROOT"/flash_attn-*.whl

"$PYTHON_BIN" -m pip install gsplat==1.5.3 einops==0.8.2

echo "CUDA extension wheels are available in $WHEELHOUSE_ROOT"
