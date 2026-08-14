# LHM++ installation: `pt210` / CUDA 12.8

## Supported runtime

- Linux x86-64
- Conda Python 3.11
- PyTorch `2.10.0+cu128`, torchvision `0.25.0+cu128`, torchaudio `2.10.0+cu128`
- xFormers `0.0.35`
- NVIDIA driver capable of running CUDA 12.8 binaries
- CUDA Toolkit 12.8, C/C++ compiler, CMake and Ninja when compiling CUDA extensions

The PyTorch wheel contains the CUDA runtime. A system CUDA Toolkit is required only for
source-built extensions. GPU architecture is detected at build time; wheels built only
for Blackwell `sm_120` must not be copied to a different GPU architecture.

## 1. Clone and create the environment

```bash
git clone https://github.com/aigc3d/LHM-plusplus.git
cd LHM-plusplus
bash envs/torch210-cu128/create_env.sh
conda activate pt210
```

## 2. Install public binary dependencies

```bash
python -m pip install --no-deps \
  'pytorch3d==0.7.9+pt2.10.0cu128' \
  --extra-index-url https://miropsota.github.io/torch_packages_builder
python -m pip install --no-deps torch-scatter==2.1.2 \
  -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
python -m pip install --no-deps spconv-cu128==2.4.1 \
  --extra-index-url https://ratharog.github.io/cumm-spconv/
```

PyTorch3D can instead be cloned from `https://github.com/facebookresearch/pytorch3d.git`
at tag `v0.7.9` and built with `FORCE_CUDA=1`, `--no-deps`, and
`--no-build-isolation`.

## 3. Build dependencies without public matching wheels

```bash
bash envs/torch210-cu128/build_cuda_extensions.sh
```

The script clones and pins:

| package | source revision |
|---|---|
| diff-gaussian-rasterization | `graphdeco-inria/diff-gaussian-rasterization`, commit `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0` |
| simple-knn | `camenduru/simple-knn`, commit `86710c2d4b46680c02301765dd79e465819c8f19` |
| flash-attn | `Dao-AILab/flash-attention`, tag `v2.8.3.post1` |
| pointops | this repository's `lib/pointops` |

By default the script uses a temporary source/build directory and removes it when
finished. The compiled extensions are installed directly into the active Conda environment.
Set `BUILD_ROOT` only if a persistent source cache is desired; `PYTHON_BIN`,
`TORCH_CUDA_ARCH_LIST`, and `MAX_JOBS` are also configurable.

## 4. Sapiens source dependency

LHM++ imports the Meta Sapiens forks of mmengine/mmcv/mmseg/mmpose. The
Python 3.11 / PyTorch 2.10 compatibility changes are published on the maintained
fork. Clone and pin it as a sibling checkout:

```bash
git clone https://github.com/PoliteYoung/sapiens.git ../../third_party/sapiens
git -C ../../third_party/sapiens checkout 4ad09e7017d9ed9ff78e58c557200677d04eb4fd
for pkg in engine cv seg pose; do
  python -m pip install --no-deps --no-build-isolation -e "../../third_party/sapiens/$pkg"
done
```

The fixed commit is on the fork's default `main` branch. Sapiens source contains
no model weights; obtain official checkpoints from
`https://huggingface.co/facebook/sapiens` as required by the selected config.

## 5. Install LHM++ Python requirements

```bash
python -m pip install --no-deps -r requirements.txt
```

`requirements.txt` contains only portable public package references and repository-relative
editable paths. Install public packages with `--no-deps` when dependency metadata would replace the
validated OpenCV/NumPy stack. ABI-sensitive packages are built from source on the target machine.
`megfile==5.0.15` is installed from its public pure-Python wheel together with its supported
public dependency `paramiko==3.5.1`; no local repack is retained.

## 6. Download weights and assets

```bash
python scripts/download_all.py
# or independently:
python scripts/download_pretrained_models.py --prior
python scripts/download_pretrained_models.py --models
python scripts/download_motion_video.py
```

| asset | source | destination |
|---|---|---|
| LHMPP-700M / LHMPP-700MC / LHMPPS-700M | Hugging Face or ModelScope; script falls back between mirrors | `pretrained_models/` |
| LHMPP-Prior (human models, voxel grid, ArcFace, BiRefNet, etc.) | `Damo_XR_Lab/LHMPP-Prior` / corresponding HF mirror | `pretrained_models/` |
| motion examples | `Damo_XR_Lab/LHMPP-Assets` | `motion_video/` |

Some SMPL/SMPL-X/MANO assets have separate licenses. Do not redistribute them outside
the terms accepted at the official model sites. Confirm that the downloaded prior contains
the required `human_model_files`; otherwise obtain them from the official SMPL-X/MANO sites.

## 7. Verify

```bash
python envs/torch210-cu128/verify_attention.py
python envs/torch210-cu128/verify_extensions.py
python -m pip check
```

See `envs/torch210-cu128/README.md` for the full dependency and reproducibility audit.
