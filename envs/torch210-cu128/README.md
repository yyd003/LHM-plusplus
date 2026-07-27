# LHM++ on PyTorch 2.10 / CUDA 12.8

This environment targets NVIDIA Blackwell GPUs, and has been probed on an
RTX 5090 (compute capability 12.0).

## Version contract

- Python 3.11
- PyTorch 2.10.0 + CUDA 12.8
- torchvision 0.25.0 + CUDA 12.8
- torchaudio 2.10.0 + CUDA 12.8
- xFormers 0.0.35

The repository's root `requirements.txt` remains the upstream PyTorch 2.3 /
CUDA 12.1 contract. Do not install it verbatim after creating this environment,
because it would downgrade PyTorch and xFormers.

## Create the environment

```bash
conda create -n lhmpp-pt210-cu128 python=3.11 pip -y
conda activate lhmpp-pt210-cu128

python -m pip install \
  torch==2.10.0 \
  torchvision==0.25.0 \
  torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  xformers==0.0.35 \
  --index-url https://download.pytorch.org/whl/cu128
```

The same pinned core set is available as:

```bash
python -m pip install -r envs/torch210-cu128/requirements-core.txt
```

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

After installing the core requirements, both currently verified extensions can
also be installed together:

```bash
python -m pip install --no-cache-dir \
  -r envs/torch210-cu128/requirements-extensions.txt
```

### Other compiled extensions

At the time this environment was created, the MiroPsota index did not contain
PyTorch 2.10 / CUDA 12.8 wheels for `flash-attn`,
`diff-gaussian-rasterization`, or `simple-knn`. Install a matching wheel when
one becomes available, or build the packages against this environment. Do not
install the available PyTorch 2.8 wheels into this environment.

`flash-attn==2.8.3.post1` recognizes CUDA 12.8 and generates `sm_120` code when
built from source, but its build currently compiles kernels for several older
architectures as well. A full local build is therefore expensive and is not
part of the reproducible core setup. Until a matching PyTorch 2.10 wheel is
available, Sonata will use its existing optional non-Flash fallback when the
package is absent. PyTorch SDPA and the xFormers path remain accelerated.

`spconv-cu128` is maintained through the
[rathaROG/cumm-spconv package index](https://ratharog.github.io/cumm-spconv/)
and is the fallback source because it is not present in the MiroPsota index.

```bash
python -m pip install --no-cache-dir \
  spconv-cu128==2.4.1 \
  --extra-index-url https://ratharog.github.io/cumm-spconv/
```

Repository-local extensions such as `lib/pointops` must be rebuilt inside the
PyTorch 2.10 environment.

## Verify the attention backends

Run the checked-in probe from the repository root:

```bash
conda run -n lhmpp-pt210-cu128 \
  python envs/torch210-cu128/verify_attention.py

conda run -n lhmpp-pt210-cu128 \
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
