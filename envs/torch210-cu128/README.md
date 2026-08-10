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
`/home/dreams/.conda/envs/pt210`. Do not recreate it per project.

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
`requirements.txt`; install existing wheels with `--no-deps` and build local
extensions with `--no-deps --no-build-isolation`.

### Locally built CUDA extensions

The MiroPsota index does not currently publish PyTorch 2.10 / CUDA 12.8 wheels
for `flash-attn`, `diff-gaussian-rasterization`, or `simple-knn`. Matching
wheels were therefore built locally from the same source revisions used by
the MiroPsota packages where available:

- `diff-gaussian-rasterization`: commit `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`
- `simple-knn`: commit `86710c2d4b46680c02301765dd79e465819c8f19`
- `flash-attn`: tag `v2.8.3.post1`

The FlashAttention wheel is deliberately compiled only for `sm_120`. It is an
RTX 50-series/Blackwell-specific wheel and must not be installed on older GPU
architectures.

Build and install the three wheels, then install gsplat:

```bash
bash envs/torch210-cu128/build_cuda_extensions.sh
```

The default wheelhouse is outside the Git repository at:

```text
../wheelhouse/pt210-cu128/
```

The generated wheels are not committed because they are large, Python/ABI
specific binary artifacts. On this machine their SHA-256 hashes are:

```text
fed9ec54bd81cc0e21fca102643ac831583c59d3cac6ee881d0301f6a44c60cd  diff_gaussian_rasterization-0.0.0-cp311-cp311-linux_x86_64.whl
16fcf7d7cdecf78478c8f8a8eedf4832d8bb26b24f3db7a1f98a78ab627716f4  simple_knn-0.0.0-cp311-cp311-linux_x86_64.whl
408087fd5cfa0643d7902e446fc95940625b587dced25de67b4dbc348c9c061e  flash_attn-2.8.3.post1-cp311-cp311-linux_x86_64.whl
```

`spconv-cu128` is maintained through the
[rathaROG/cumm-spconv package index](https://ratharog.github.io/cumm-spconv/)
and is the fallback source because it is not present in the MiroPsota index.

```bash
python -m pip install --no-cache-dir \
  spconv-cu128==2.4.1 \
  --extra-index-url https://ratharog.github.io/cumm-spconv/
```

Repository-local extensions such as `lib/pointops` must be rebuilt inside the
PyTorch 2.10 environment. PointOps imports PyTorch from `setup.py`, so an isolated
PEP 517 build can select a different PyTorch/CUDA stack. Build it without build
isolation, save the resulting wheel in the local wheelhouse, and then let
`requirements.txt` install that wheel:

```bash
/home/dreams/.conda/envs/pt210/bin/python -m pip wheel \
  --no-build-isolation --no-deps \
  /home/dreams/yaodong/LHM-plusplus/lib/pointops \
  --wheel-dir /home/dreams/yaodong/wheelhouse/pt210-cu128

/home/dreams/.conda/envs/pt210/bin/python -m pip install \
  --no-index --find-links /home/dreams/yaodong/wheelhouse/pt210-cu128 \
  --force-reinstall --no-deps pointops==0.0.0
```

The pip option is `--no-build-isolation` (there is no
`--no-build-isolation` option). A requirements file cannot attach this
build option to only one dependency, which is why the checked-in requirements
resolve the prebuilt wheel through `--find-links` instead of rebuilding PointOps.


## ONNX Runtime CUDA loader

ORT 1.26 and PyTorch 2.10 share the existing CUDA 12 wheel libraries. Project
entry points call `onnxruntime.preload_dlls()`, and the provider RUNPATH can be
repaired after any ORT reinstall without adding another CUDA runtime:

```bash
/home/dreams/.conda/envs/pt210/bin/python \
  envs/torch210-cu128/repair_ort_cuda_runpath.py \
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
