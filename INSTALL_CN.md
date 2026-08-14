# LHM++ 安装说明：`pt210` / CUDA 12.8

## 运行环境

- Linux x86-64
- Conda Python 3.11
- PyTorch `2.10.0+cu128`、torchvision `0.25.0+cu128`、torchaudio `2.10.0+cu128`
- xFormers `0.0.35`
- 能运行 CUDA 12.8 二进制的 NVIDIA 驱动
- 编译 CUDA 扩展时另需 CUDA Toolkit 12.8、C/C++ 编译器、CMake、Ninja

PyTorch wheel 自带 CUDA runtime；只有编译扩展时才需要系统 CUDA Toolkit。构建脚本会按
当前 GPU 架构编译，不能把只含 `sm_120` 的 Blackwell wheel 复制到其他架构使用。

## 1. 克隆并建立环境

```bash
git clone https://github.com/aigc3d/LHM-plusplus.git
cd LHM-plusplus
bash envs/torch210-cu128/create_env.sh
conda activate pt210
```

## 2. 安装有公开 wheel 的二进制依赖

```bash
python -m pip install --no-deps \
  'pytorch3d==0.7.9+pt2.10.0cu128' \
  --extra-index-url https://miropsota.github.io/torch_packages_builder
python -m pip install --no-deps torch-scatter==2.1.2 \
  -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
python -m pip install --no-deps spconv-cu128==2.4.1 \
  --extra-index-url https://ratharog.github.io/cumm-spconv/
```

PyTorch3D 也可从 `https://github.com/facebookresearch/pytorch3d.git` 的 `v0.7.9`
tag clone，并使用 `FORCE_CUDA=1 --no-deps --no-build-isolation` 编译。

## 3. 编译没有匹配公开 wheel 的依赖

```bash
bash envs/torch210-cu128/build_cuda_extensions.sh
```

脚本会固定源码版本：

| 包 | 来源和版本 |
|---|---|
| diff-gaussian-rasterization | `graphdeco-inria/diff-gaussian-rasterization`，commit `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0` |
| simple-knn | `camenduru/simple-knn`，commit `86710c2d4b46680c02301765dd79e465819c8f19` |
| flash-attn | `Dao-AILab/flash-attention`，tag `v2.8.3.post1` |
| pointops | 本仓库 `lib/pointops` |

脚本默认使用临时源码/编译目录，并在结束时自动删除；编译结果直接安装到当前 Conda
环境。只有需要长期保存源码缓存时才设置 `BUILD_ROOT`，也可设置 `PYTHON_BIN`、
`TORCH_CUDA_ARCH_LIST`、`MAX_JOBS`。

## 4. Sapiens 源码依赖

Python 3.11 / PyTorch 2.10 兼容修改已经发布到维护 fork。换机器时克隆并固定到以下提交：

```bash
git clone https://github.com/PoliteYoung/sapiens.git ../../third_party/sapiens
git -C ../../third_party/sapiens checkout 4ad09e7017d9ed9ff78e58c557200677d04eb4fd
for pkg in engine cv seg pose; do
  python -m pip install --no-deps --no-build-isolation -e "../../third_party/sapiens/$pkg"
done
```

固定提交已在 fork 的默认 `main` 分支中。Sapiens 源码不包含模型权重；所需官方权重请从
`https://huggingface.co/facebook/sapiens` 获取，并按所选配置放置。

## 5. 安装项目依赖

```bash
python -m pip install --no-deps -r requirements.txt
```

`requirements.txt` 已移除机器绝对路径，只使用公开包和仓库相对路径。没有匹配公开 wheel 的
扩展由安装脚本在目标机器从源码编译。
其中 `megfile==5.0.15` 直接使用 PyPI 的纯 Python wheel，并固定其公开支持的
`paramiko==3.5.1`；不再维护本地改包。

## 6. 下载权重和资产

```bash
python scripts/download_all.py
# 或分开下载：
python scripts/download_pretrained_models.py --prior
python scripts/download_pretrained_models.py --models
python scripts/download_motion_video.py
```

| 资产 | 来源 | 目标目录 |
|---|---|---|
| LHMPP-700M / LHMPP-700MC / LHMPPS-700M | Hugging Face 或 ModelScope，脚本自动回退 | `pretrained_models/` |
| LHMPP-Prior（人体模型、voxel grid、ArcFace、BiRefNet 等） | `Damo_XR_Lab/LHMPP-Prior` / 对应 HF 镜像 | `pretrained_models/` |
| 动作示例 | `Damo_XR_Lab/LHMPP-Assets` | `motion_video/` |

SMPL/SMPL-X/MANO 受各自许可限制。若 prior 中缺少 `human_model_files`，请从官方
SMPL-X/MANO 站点在接受许可后获取，不要擅自再分发。

## 7. 验证

```bash
python envs/torch210-cu128/verify_attention.py
python envs/torch210-cu128/verify_extensions.py
python -m pip check
```

完整依赖和可复现性审计见 `envs/torch210-cu128/README.md`。
