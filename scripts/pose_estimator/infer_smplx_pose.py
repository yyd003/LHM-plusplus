# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : CLI for SMPL-X pose estimation from single image

import argparse
import sys
from pathlib import Path

import torch

sys.path.append("./")

from engine.pose_estimation.pose_estimator import PoseEstimator


def get_parse() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments object.
    """
    parser = argparse.ArgumentParser(description="infer smplx pose demo")
    parser.add_argument(
        "-i",
        "--input",
        default="assets/pose_img.jpg",
        type=str,
        help="input image path",
    )
    parser.add_argument(
        "--model-dir",
        default="pretrained_models/human_model_files",
        type=str,
        help="directory containing pose_estimate/ and the SMPL-X model files",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, for example cuda, cuda:1, or cpu",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="optional .pt file used to save the estimator output",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parse()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    pose_estimator = PoseEstimator(args.model_dir, device=device)
    with torch.no_grad():
        smplx_output = pose_estimator(args.input, is_shape=False)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(smplx_output, output_path)
        print(f"Saved SMPL-X estimator output to {output_path}")
    elif isinstance(smplx_output, dict):
        summary = {
            key: tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            for key, value in smplx_output.items()
        }
        print(summary)
    else:
        print(smplx_output)
