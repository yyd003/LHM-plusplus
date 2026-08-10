#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate get_motion_information only (no LHM++ model required).

Usage:
    # motion_path: path to smplx_params directory (e.g. motion_video/Dance_I/smplx_params)
    python scripts/test/test_get_motion_information.py --motion_path ./motion_video/Dance_I/smplx_params

    # or derive from motion video path (same logic as app.py prepare_input_and_output)
    python scripts/test/test_get_motion_information.py --motion_video ./motion_video/Dance_I/Dance_I.mp4

Note: The motion helpers must not contain interactive debugger breakpoints.
      for uninterrupted run if present.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from omegaconf import OmegaConf

from core.utils.app_utils import get_motion_information
from scripts.download_motion_video import motion_video_check


def _resolve_motion_path(motion_path: str = None, motion_video: str = None) -> str:
    """Resolve motion_path (smplx_params dir) from args."""
    if motion_path:
        return os.path.abspath(motion_path)
    if motion_video and os.path.isfile(motion_video):
        base_vid = os.path.basename(motion_video).split(".")[0]
        motion_dir = os.path.dirname(motion_video)
        return os.path.join(motion_dir, "smplx_params")
    raise ValueError(
        "Provide --motion_path (path to smplx_params dir) or --motion_video (path to motion .mp4)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate get_motion_information only (no LHM model)"
    )
    parser.add_argument(
        "--motion_path",
        type=str,
        default=None,
        help="Path to smplx_params directory",
    )
    parser.add_argument(
        "--motion_video",
        type=str,
        default=None,
        help="Path to motion video (derive motion_path from parent/smplx_params)",
    )
    parser.add_argument(
        "--motion_size",
        type=int,
        default=120,
        help="Number of motion frames",
    )
    parser.add_argument(
        "--render_size",
        type=int,
        default=420,
        help="Render size (default 420)",
    )
    args = parser.parse_args()

    motion_video_check(save_dir=".")

    motion_path = _resolve_motion_path(args.motion_path, args.motion_video)
    if not os.path.isdir(motion_path):
        raise FileNotFoundError(f"smplx_params dir not found: {motion_path}")

    cfg = OmegaConf.create({
        "render_size": args.render_size,
        "motion_img_need_mask": False,
        "vis_motion": False,
    })

    print(f"[1/2] Calling get_motion_information(motion_path={motion_path}, motion_size={args.motion_size})")
    motion_name, motion_seqs = get_motion_information(
        motion_path, cfg, motion_size=args.motion_size
    )
    video_size = len(motion_seqs["motion_seqs"])

    print(f"[2/2] OK. motion_name={motion_name}, frames={video_size}")
    print(f"      Keys: {list(motion_seqs.keys())}")


if __name__ == "__main__":
    main()
