# LHM++ on PyTorch 2.10 / CUDA 12.8

This environment targets NVIDIA Blackwell GPUs, and has been probed on an
RTX 5090 (compute capability 12.0).

## Version contract

- Python 3.11
- PyTorch 2.10.0 + CUDA 12.8
- torchvision 0.25.0 + CUDA 12.8
- torchaudio 2.10.0 + CUDA 12.8
- xFormers 0.0.35

The repository root `requirements.txt` has been rebuilt from the verified shared
`pt210` runtime. Install it only with `--no-deps`; local source projects and CUDA
extensions must additionally use `--no-build-isolation` so dependency resolution
cannot replace Torch, OpenCV, ORT, or the Sapiens forks.

## Create the environment

```bash
conda create -n pt210 python=3.11 pip -y
conda activate pt210

python -m pip install \
  torch==2.10.0 \
  torchvision==0.25.0 \
  torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  xformers==0.0.35 \
  --index-url https://download.pytorch.org/whl/cu128
```

On this machine the shared environment already exists at
the active Conda environment named `pt210`. Do not recreate it per project.

## Legacy CUDA extensions

Prefer wheels from
[MiroPsota/torch_packages_builder](https://github.com/MiroPsota/torch_packages_builder)
for packages whose upstream projects no longer publish current wheels. Wheels
must match all of Python, PyTorch, and CUDA; never reuse a wheel built for a
different PyTorch minor version.

### PyTorch3D

MiroPsota provides a matching CPython 3.11 / PyTorch 2.10 / CUDA 12.8 wheel:

```bash
python -m pip install --no-cache-dir \
  'pytorch3d==0.7.9+pt2.10.0cu128' \
  --extra-index-url https://miropsota.github.io/torch_packages_builder
```

### torch-scatter

`torch-scatter` is not currently present in the MiroPsota package index. Use
the matching PyG wheel:

```bash
python -m pip install --no-cache-dir torch_scatter \
  -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
```

The verified extension versions are now recorded in the repository root
`requirements.txt`; install matching public wheels with `--no-deps` and compile
source extensions with `--no-deps --no-build-isolation`.

### Locally built CUDA extensions

The MiroPsota index does not currently publish PyTorch 2.10 / CUDA 12.8 wheels
for `flash-attn`, `diff-gaussian-rasterization`, or `simple-knn`. They are therefore compiled from pinned source revisions on each destination machine:

- `diff-gaussian-rasterization`: commit `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`
- `simple-knn`: commit `86710c2d4b46680c02301765dd79e465819c8f19`
- `flash-attn`: tag `v2.8.3.post1`

The build script detects the active GPU architecture. Do not copy the resulting
binaries between machines with different Python, PyTorch, CUDA, or GPU architectures.

Build and install the extensions, then install gsplat:

```bash
bash envs/torch210-cu128/build_cuda_extensions.sh
```

The compiled packages are installed directly into the active Conda environment.
The script uses and removes a temporary build directory by default, so no
machine-specific binary cache is kept in the workspace. Set `BUILD_ROOT` only
when an explicit persistent source cache is desired.

`spconv-cu128` is maintained through the
[rathaROG/cumm-spconv package index](https://ratharog.github.io/cumm-spconv/)
and is the fallback source because it is not present in the MiroPsota index.

```bash
python -m pip install --no-cache-dir \
  spconv-cu128==2.4.1 \
  --extra-index-url https://ratharog.github.io/cumm-spconv/
```

Repository-local extensions such as `lib/pointops` must be rebuilt inside the
PyTorch 2.10 environment. PointOps imports PyTorch from `setup.py`, so install it
without build isolation:

```bash
python -m pip install --force-reinstall --no-deps --no-build-isolation ./lib/pointops
```

The checked-in build script performs this source installation automatically.


## ONNX Runtime CUDA loader

ORT 1.26 and PyTorch 2.10 share the existing CUDA 12 wheel libraries. Project
entry points call `onnxruntime.preload_dlls()`, and the provider RUNPATH can be
repaired after any ORT reinstall without adding another CUDA runtime:

```bash
python envs/torch210-cu128/repair_ort_cuda_runpath.py \
  --verify-model pretrained_models/u2net/u2net.onnx
```

## Verify the attention backends

Run the checked-in probe from the repository root:

```bash
conda run -n pt210 \
  python envs/torch210-cu128/verify_attention.py

conda run -n pt210 \
  python envs/torch210-cu128/verify_extensions.py
```

Expected on an RTX 5090:

- xFormers `memory_efficient_attention`: FP16 and BF16 pass.
- PyTorch Flash SDPA: FP16 and BF16 pass.
- FP32 xFormers attention may remain unsupported and is not the LHM++ inference
  path.

## Known wheel sources

1. PyTorch CUDA wheels: `https://download.pytorch.org/whl/cu128`
2. Legacy extension priority: `https://miropsota.github.io/torch_packages_builder`
3. PyG extensions: `https://data.pyg.org/whl/torch-2.10.0+cu128.html`
4. spconv fallback: `https://ratharog.github.io/cumm-spconv/`

## Portable rebuild audit (2026-08-13)

Use `create_env.sh` rather than a machine-specific Conda prefix. Public package
indexes currently provide the pinned PyTorch/xFormers base, torch-scatter,
PyTorch3D community wheels, spconv, MediaPipe 0.10.35, pyrender 0.1.45,
Albumentations 2.0.8, cuDSS 0.8.0.10 and ONNX Runtime GPU 1.26.0.

Public Python packages are installed with `--no-deps` when their dependency metadata
would otherwise replace the validated Torch/OpenCV/NumPy stack. ABI-sensitive and
CUDA packages are compiled from source on the destination machine.

| component | public source / rebuild status |
|---|---|
| CUDA Gaussian extensions and PointOps | fully reproducible with `build_cuda_extensions.sh` |
| PyTorch3D 0.7.9 | public source tag `v0.7.9`; matching community wheel also documented |
| Sapiens OpenMMLab forks | clone `https://github.com/PoliteYoung/sapiens.git` and pin `4ad09e7017d9ed9ff78e58c557200677d04eb4fd`; compatibility changes are on default `main` |
| Theseus | clone `https://github.com/PoliteYoung/theseus.git` and pin `aa219c4ac582b9a662f7d8991c3181a7d568ab56`; build `0.2.3+pt210` with cuDSS from default `main` |
| MediaPipe 0.10.35 | install the public wheel with `--no-deps` so it does not replace the selected OpenCV build |
| pyrender / Albumentations | install public releases with `--no-deps` |
| chumpy / xtcocotools | compile the maintained source snapshots for the destination Python/NumPy ABI |

Sapiens and Theseus are rebuilt from their public pinned forks. CUDA extensions and
ABI-sensitive compatibility sources are rebuilt locally rather than copied as binaries.
