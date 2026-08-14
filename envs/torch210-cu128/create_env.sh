#!/usr/bin/env bash
# Create the portable shared pt210 environment used by this workspace.
set -euo pipefail

ENV_NAME="${PT210_ENV_NAME:-pt210}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
[[ -n "$CONDA_BIN" ]] || { echo "ERROR: conda/mamba is required." >&2; exit 2; }

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_BIN" create -n "$ENV_NAME" python=3.11 pip -y
fi

run() { "$CONDA_BIN" run -n "$ENV_NAME" "$@"; }
run python -m pip install --upgrade pip setuptools wheel ninja cmake packaging
run python -m pip install \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
run python -m pip install xformers==0.0.35 \
  --index-url https://download.pytorch.org/whl/cu128

run python - <<'PY'
import sys, torch
assert sys.version_info[:2] == (3, 11), sys.version
assert torch.__version__ == "2.10.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
print("created pt210:", sys.version.split()[0], torch.__version__, "CUDA", torch.version.cuda)
print("GPU visible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
PY

echo "Activate with: conda activate $ENV_NAME"
