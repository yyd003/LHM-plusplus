# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-10-20 20:00:00
# @Function      : LHM++ Unified Inference Code. Supports both video generation and benchmark evaluation.

import argparse
import glob
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from PIL import Image

# Add project root to path
sys.path.append("./")

# Disable torch dynamo for compatibility
torch._dynamo.config.disable = True

from tqdm import tqdm

from core.runners.infer.utils import prepare_motion_seqs, prepare_motion_seqs_eval
from core.structures.bbox import Bbox
from core.utils.hf_hub import wrap_model_hub
from scripts.inference.utils import get_smplx_params, obtain_motion_sequence

# Dataset configuration mapping
DATASETS_CONFIG = {
    "eval": {
        "root_dirs": "./benchmark/dataset_evaluation/LHM_video_dataset/",
        "meta_path": "./benchmark/clean_labels/valid_LHM_dataset_train_val_100.json",
    },
    "dataset5": {
        "root_dirs": "./benchmark/dataset_evaluation/selected_dataset_v5_tar/",
        "meta_path": "./benchmark/clean_labels/valid_selected_datasetv5_val_filter460-self-rotated-69.json",
    },
    "dataset6": {
        "root_dirs": "./benchmark/dataset_evaluation/selected_dataset_v6_tar/",
        "meta_path": "./benchmark/clean_labels/valid_selected_datasetv6_test_100-self-rotated-25.json",
    },
    "in_the_wild": {
        "root_dirs": "./benchmark/dataset_evaluation/PFLHM-Causal-Video/",
        "meta_path": "./benchmark/clean_labels/eval_sparse_lhm_wild.json",
    },
    "in_the_wild_rec_mv": {
        "root_dirs": "./benchmark/dataset_evaluation/rec_mv_dataset",
        "meta_path": "./benchmark/dataset_evaluation/rec_mv_dataset/rec_mv_dataset.json",
    },
    "in_the_wild_mvhumannet": {
        "root_dirs": "./benchmark/dataset_evaluation/mvhumannet/",
        "meta_path": "./benchmark/dataset_evaluation/mvhumannet/mvhumannet.json",
    },
    "web_dresscode": {
        "root_dirs": "./benchmark/dataset_evaluation/Dresscode/",
        "meta_path": None,
    },
    "hweb_hero": {
        "root_dirs": "./benchmark/dataset_evaluation/flux_generation/",
        "meta_path": None,
    },
}

# Constants
DEFAULT_FPS = 30
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_PIXEL_FORMAT = "yuv420p"
DEFAULT_VIDEO_BITRATE = "10M"
MACRO_BLOCK_SIZE = 16
SMPLX_EXPRESSION_DIM = 100
GRID_SIZE = 4  # Number of images in the grid (2x2)
DEFAULT_SRC_HEAD_SIZE = 112
DEFAULT_BATCH_SIZE = 40


def resize_with_padding(
    images: List[np.ndarray],
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Combine 4 images into a 2x2 grid, resize with aspect ratio, and pad.

    Args:
        images: List of 4 numpy arrays, each of shape (H, W, C) with dtype uint8.
        target_size: Target output dimensions as (height, width).

    Returns:
        Combined and padded image of shape (target_height, target_width, 3).

    Raises:
        ValueError: If number of images is not 4 or images have different dimensions.
    """
    if len(images) != GRID_SIZE:
        raise ValueError(f"Exactly {GRID_SIZE} images are required, got {len(images)}")

    height, width = images[0].shape[:2]
    if not all(img.shape[:2] == (height, width) for img in images):
        raise ValueError("All images must have the same dimensions")

    # Arrange images in 2x2 grid
    top_row = np.hstack([images[0], images[1]])
    bottom_row = np.hstack([images[2], images[3]])
    combined = np.vstack([top_row, bottom_row])

    combined_h, combined_w = combined.shape[:2]
    target_h, target_w = target_size

    # Compute scale to preserve aspect ratio
    scale = min(target_h / combined_h, target_w / combined_w)
    new_h = int(combined_h * scale)
    new_w = int(combined_w * scale)

    # Resize combined image
    resized = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create white canvas and center the resized image
    padded = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    top_offset = (target_h - new_h) // 2
    left_offset = (target_w - new_w) // 2
    padded[top_offset : top_offset + new_h, left_offset : left_offset + new_w] = resized

    return padded


def _build_model(cfg: DictConfig) -> torch.nn.Module:
    """Build and load the LHM model from pretrained weights.

    Args:
        cfg: Configuration object containing model_name and other parameters.

    Returns:
        Loaded LHM model ready for inference.
    """
    from core.models import model_dict

    model_cls = wrap_model_hub(model_dict["human_lrm_a4o"])
    model = model_cls.from_pretrained(cfg.model_name)
    return model


@torch.no_grad()
def inference_results(
    lhm: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    smplx_params: Dict[str, torch.Tensor],
    motion_seq: Dict[str, Any],
    camera_size: int = 40,
    ref_imgs_bool: Optional[torch.Tensor] = None,
    ori_space: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str = "cuda",
) -> np.ndarray:
    """Run inference on a motion sequence with batching to prevent OOM.

    Args:
        lhm: LHM model for human animation.
        batch: Input batch containing source images and metadata.
        smplx_params: SMPL-X parameters for the initial pose.
        motion_seq: Dictionary containing motion sequence data.
        camera_size: Total number of frames to render.
        ref_imgs_bool: Boolean mask indicating which reference images to use.
        ori_space: If False, crops output to subject bounds with 10% padding.
        batch_size: Number of frames to process in each batch.
        device: Device to run inference on.

    Returns:
        Rendered RGB frames as numpy array of shape (T, H, W, 3).
    """
    offset_list = motion_seq.get("offset_list")
    ori_h, ori_w = motion_seq.get("ori_size", (512, 512))
    output_rgb = torch.ones((ori_h, ori_w, 3))

    # Run single-view inference to extract features
    model_outputs = lhm.infer_single_view(
        batch["source_rgbs"].unsqueeze(0).to(device),
        None,
        None,
        render_c2ws=motion_seq["render_c2ws"].to(device),
        render_intrs=motion_seq["render_intrs"].to(device),
        render_bg_colors=motion_seq["render_bg_colors"].to(device),
        smplx_params={k: v.to(device) for k, v in smplx_params.items()},
        ref_imgs_bool=ref_imgs_bool,
    )

    # Unpack model outputs (handle different output formats)
    if len(model_outputs) == 7:
        (
            gs_model_list,
            query_points,
            transform_mat_neutral_pose,
            gs_hidden_features,
            image_latents,
            motion_emb,
            pos_emb,
        ) = model_outputs
    else:
        (
            gs_model_list,
            query_points,
            transform_mat_neutral_pose,
            gs_hidden_features,
            image_latents,
            motion_emb,
        ) = model_outputs
        pos_emb = None

    # Prepare batch-invariant parameters
    batch_smplx_params = {
        "betas": smplx_params["betas"].to(device),
        "transform_mat_neutral_pose": transform_mat_neutral_pose,
    }

    # Parameters that vary across frames
    frame_varying_keys = [
        "root_pose",
        "body_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "lhand_pose",
        "rhand_pose",
        "trans",
        "focal",
        "princpt",
        "img_size_wh",
        "expr",
    ]

    # Process motion sequence in batches
    batch_rgb_list = []
    batch_mask_list = []
    num_batches = (camera_size + batch_size - 1) // batch_size

    for batch_idx in range(0, camera_size, batch_size):
        current_batch = batch_idx // batch_size + 1
        print(f"Processing batch {current_batch}/{num_batches}")

        # Update frame-varying parameters for current batch
        batch_smplx_params.update(
            {
                key: motion_seq["smplx_params"][key][
                    :, batch_idx : batch_idx + batch_size
                ].to(device)
                for key in frame_varying_keys
            }
        )

        # Get masks for current batch if available
        mask_seqs = (
            motion_seq.get("masks", [])[batch_idx : batch_idx + batch_size]
            if "masks" in motion_seq
            else None
        )

        # Prepare animation infer kwargs
        anim_kwargs = {
            "gs_model_list": gs_model_list,
            "query_points": query_points,
            "smplx_params": batch_smplx_params,
            "render_c2ws": motion_seq["render_c2ws"][
                :, batch_idx : batch_idx + batch_size
            ].to(device),
            "render_intrs": motion_seq["render_intrs"][
                :, batch_idx : batch_idx + batch_size
            ].to(device),
            "render_bg_colors": motion_seq["render_bg_colors"][
                :, batch_idx : batch_idx + batch_size
            ].to(device),
            "gs_hidden_features": gs_hidden_features,
            "image_latents": image_latents,
            "motion_emb": motion_emb,
        }

        # Add optional parameters
        if pos_emb is not None:
            anim_kwargs["pos_emb"] = pos_emb
        if offset_list is not None:
            anim_kwargs["offset_list"] = offset_list[batch_idx : batch_idx + batch_size]
        if mask_seqs is not None:
            anim_kwargs["mask_seqs"] = mask_seqs
        if output_rgb is not None:
            anim_kwargs["output_rgb"] = output_rgb

        # Run animation inference
        batch_rgb, batch_mask = lhm.animation_infer(**anim_kwargs)

        # Convert to uint8 and store
        batch_rgb_list.append((batch_rgb.clamp(0, 1) * 255).to(torch.uint8).numpy())
        batch_mask_list.append((batch_mask.clamp(0, 1) * 255).to(torch.uint8).numpy())

    # Optionally crop to subject bounds
    if not ori_space and batch_mask_list:
        # Find bounding box of subject across all frames
        mask_numpy = np.concatenate(batch_mask_list, axis=0)
        h_indices, w_indices = np.where(mask_numpy > 0.25)[1:]

        if len(h_indices) > 0 and len(w_indices) > 0:
            top, bottom = h_indices.min(), h_indices.max()
            left, right = w_indices.min(), w_indices.max()

            # Add 10% padding around subject
            center_y, center_x = (top + bottom) / 2, (left + right) / 2
            height, width = bottom - top, right - left
            new_height, new_width = height * 1.1, width * 1.1

            # Compute padded bounds
            top_new = max(0, int(center_y - new_height / 2))
            bottom_new = int(center_y + new_height / 2)
            left_new = max(0, int(center_x - new_width / 2))
            right_new = int(center_x + new_width / 2)

            # Crop to bounds
            rgb = np.concatenate(batch_rgb_list, axis=0)
            output = rgb[:, top_new:bottom_new, left_new:right_new]
        else:
            output = np.concatenate(batch_rgb_list, axis=0)
    else:
        output = np.concatenate(batch_rgb_list, axis=0)

    return output


@torch.no_grad()
def inference_video_mode(
    lhm: Optional[torch.nn.Module],
    save_path: str,
    view: int = 16,
    cfg: Optional[Dict] = None,
    motion_path: Optional[str] = None,
    exp_name: str = "eval",
    debug: bool = False,
    split: int = 1,
    gpus: int = 0,
    ori_space: bool = False,
) -> None:
    """Run inference in video mode with external motion sequences.

    Args:
        lhm: LHM model for inference.
        save_path: Directory path where output videos will be saved.
        view: Number of input views to use for inference.
        cfg: Configuration dictionary with rendering parameters.
        motion_path: Path to directory containing motion sequence data.
        exp_name: Experiment name that determines which dataset to load.
        debug: If True, processes only 100 frames for faster iteration.
        split: Number of splits for distributed processing.
        gpus: GPU index for distributed processing.
        ori_space: If False, crops output around subject with padding.
    """
    # Prepare model
    if lhm is not None:
        lhm.cuda().eval()

    if motion_path is None:
        raise ValueError("motion_path must be provided for video mode")

    cfg = cfg or {}

    # Create output directories
    parent_dir = os.path.dirname(save_path)
    os.makedirs(os.path.join(parent_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, "mask"), exist_ok=True)

    # Select appropriate dataset class based on experiment name
    dataset_config = DATASETS_CONFIG[exp_name]
    dataset_kwargs = {}

    if "in_the_wild" in exp_name:
        from core.datasets.video_in_the_wild_dataset import (
            VideoInTheWildEval as VideoDataset,
        )

        dataset_kwargs["heuristic_sampling"] = True
    elif "hweb" in exp_name:
        from core.datasets.video_in_the_wild_web_dataset import (
            WebInTheWildHeurEval as VideoDataset,
        )

        dataset_kwargs["heuristic_sampling"] = False
    elif "web" in exp_name:
        from core.datasets.video_in_the_wild_web_dataset import (
            WebInTheWildEval as VideoDataset,
        )

        dataset_kwargs["heuristic_sampling"] = False
    else:
        from core.datasets.video_human_dataset_a4o import (
            VideoHumanA4ODatasetEval as VideoDataset,
        )

    # Initialize dataset
    dataset = VideoDataset(
        root_dirs=dataset_config["root_dirs"],
        meta_path=dataset_config["meta_path"],
        sample_side_views=7,
        render_image_res_low=420,
        render_image_res_high=420,
        render_region_size=(682, 420),
        source_image_res=512,
        debug=False,
        use_flame=True,
        ref_img_size=view,
        womask=True,
        is_val=True,
        processing_pipeline=[
            {
                "name": "PadRatioWithScale",
                "target_ratio": 5 / 3,
                "tgt_max_size_list": [840],
            },
            {"name": "ToTensor"},
        ],
        **dataset_kwargs,
    )

    # Load motion sequences
    smplx_path = os.path.join(motion_path, "smplx_params")
    mask_path = os.path.join(motion_path, "samurai_seg")
    motion_files = sorted(glob.glob(os.path.join(smplx_path, "*.json")))
    if not motion_files:
        raise FileNotFoundError(f"No SMPL-X JSON files found under {smplx_path}")
    motion_ids = [os.path.basename(f).replace(".json", "") for f in motion_files]
    mask_paths = [os.path.join(mask_path, f"{mid}.png") for mid in motion_ids]
    bbox_json_path = os.path.join(motion_path, "bbox", "bbox.json")
    bbox_dict = {}
    if os.path.isfile(bbox_json_path):
        import json

        with open(bbox_json_path) as reader:
            bbox_dict = json.load(reader)
    bbox_list = []
    for motion_id in motion_ids:
        bbox = bbox_dict.get(motion_id)
        bbox_list.append(Bbox(bbox, mode="xywh").to_whwh() if bbox is not None else None)

    # Prepare motion sequences with or without masks
    motion_size = 100 if debug else 1000
    if os.path.exists(mask_path):
        motion_seqs = prepare_motion_seqs_eval(
            obtain_motion_sequence(smplx_path),
            mask_paths=mask_paths,
            bbox_list=bbox_list,
            bg_color=1.0,
            aspect_standard=5.0 / 3,
            enlarge_ratio=[1.0, 1.0],
            tgt_size=cfg.get("render_size", 420),
            render_image_res=cfg.get("render_size", 420),
            need_mask=cfg.get("motion_img_need_mask", False),
            vis_motion=cfg.get("vis_motion", False),
            motion_size=motion_size,
            specific_id_list=None,
        )
    else:
        motion_seqs = prepare_motion_seqs(
            obtain_motion_sequence(smplx_path),
            None,
            bg_color=1.0,
            aspect_standard=5.0 / 3,
            enlarge_ratio=[1.0, 1.0],
            render_image_res=cfg.get("render_size", 420),
            need_mask=cfg.get("motion_img_need_mask", False),
            vis_motion=cfg.get("vis_motion", False),
            motion_size=motion_size,
            specific_id_list=None,
        )

    # Distribute dataset across GPUs
    dataset_size = len(dataset)
    samples_per_gpu = int(np.ceil(dataset_size / split))
    start_idx = samples_per_gpu * gpus
    end_idx = samples_per_gpu * (gpus + 1)

    # Process assigned samples
    for idx in tqdm(range(start_idx, end_idx)):

        try:
            item = dataset.__getitem__(idx, view)
            uid = item["uid"]
        except:
            continue

        print(f"Processing {uid}, idx: {idx}")

        # Create output directory
        video_dir = os.path.join(save_path, f"view_{view:03d}")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"{uid}.mp4")

        # Update motion sequence with subject-specific shape parameters
        motion_seqs["smplx_params"]["betas"] = item["betas"].unsqueeze(0)

        # Run inference
        rgbs = inference_results(
            lhm,
            item,
            motion_seqs["smplx_params"],
            motion_seqs,
            camera_size=motion_seqs["smplx_params"]["root_pose"].shape[1],
            ref_imgs_bool=item["ref_imgs_bool"].unsqueeze(0),
            ori_space=ori_space,
        )

        # Prepare reference images (first 4 views)
        source_imgs = item["source_rgbs"][:GRID_SIZE]
        source_imgs = (
            (source_imgs * 255)
            .detach()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        source_imgs_list = list(source_imgs)

        # Pad if fewer than 4 views available
        while len(source_imgs_list) < GRID_SIZE:
            source_imgs_list.append(source_imgs_list[-1])

        # Create reference image grid and concatenate with rendered frames
        ref_imgs = resize_with_padding(source_imgs_list, rgbs.shape[1:3])
        concat_rgbs = [np.concatenate([ref_imgs, rgb], axis=1) for rgb in rgbs]

        # Save video
        iio.imwrite(
            video_path,
            concat_rgbs,
            fps=DEFAULT_FPS,
            codec=DEFAULT_VIDEO_CODEC,
            pixelformat=DEFAULT_PIXEL_FORMAT,
            bitrate=DEFAULT_VIDEO_BITRATE,
            macro_block_size=MACRO_BLOCK_SIZE,
        )
        print(f"Saved video to {video_path}")


@torch.no_grad()
def inference_benchmark_mode(
    lhm: Optional[torch.nn.Module],
    save_path: str,
    view: int = 16,
    cfg: Optional[DictConfig] = None,
    exp_name: str = "eval",
    debug: bool = False,
) -> None:
    """Run inference in benchmark mode using dataset's own motion sequences.

    Args:
        lhm: LHM model for inference.
        save_path: Directory path where output images will be saved.
        view: Number of input views to use.
        cfg: Configuration dictionary with rendering parameters.
        exp_name: Experiment name that determines which dataset to load.
        debug: If True, saves debug visualizations of input images.
    """
    # Prepare model
    if lhm is not None:
        lhm.cuda().eval()

    # Load dataset configuration
    dataset_config = DATASETS_CONFIG[exp_name]
    root_dirs = dataset_config["root_dirs"]
    meta_path = dataset_config["meta_path"]

    # Select dataset class
    if exp_name == "eval":
        from core.datasets.video_human_dataset_a4o import (
            VideoHumanA4ODatasetEval as VideoDataset,
        )
    else:
        raise NotImplementedError(
            f"Dataset for exp_name='{exp_name}' not implemented for benchmark mode"
        )

    # Initialize dataset
    dataset = VideoDataset(
        root_dirs=root_dirs,
        meta_path=meta_path,
        sample_side_views=7,
        render_image_res_low=384,
        render_image_res_high=384,
        render_region_size=(682, 384),
        source_image_res=384,
        debug=False,
        use_flame=True,
        ref_img_size=view,
        womask=True,
        is_val=True,
        processing_pipeline=[
            {
                "name": "PadRatioWithScale",
                "target_ratio": 5 / 3,
                "tgt_max_size_list": [840],
            },
            {"name": "ToTensor"},
        ],
    )

    # Create output directories
    parent_dir = os.path.dirname(save_path)
    os.makedirs(os.path.join(parent_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, "mask"), exist_ok=True)

    # Process each item in dataset
    for idx in range(len(dataset)):
        print(f"Processing item {idx}/{len(dataset)}")

        item = dataset.__getitem__(idx)

        # Prepare SMPL-X parameters
        smplx_params = get_smplx_params(item, device=torch.device("cuda"))
        smplx_params["focal"] = item["focal"].unsqueeze(0)
        smplx_params["princpt"] = item["princpt"].unsqueeze(0)
        smplx_params["img_size_wh"] = item["img_size_wh"].unsqueeze(0)

        # Prepare motion sequences
        motion_seqs = {
            "render_c2ws": item["c2ws"].unsqueeze(0),
            "render_intrs": item["intrs"].unsqueeze(0),
            "render_bg_colors": item["render_bg_colors"].unsqueeze(0),
            "smplx_params": smplx_params,
        }

        source_rgbs = item["source_rgbs"]
        uid = item["uid"]

        # Debug visualization
        if debug:
            debug_dir = "./debug/heuristic_sampling"
            os.makedirs(debug_dir, exist_ok=True)
            debug_rgbs = source_rgbs.permute(0, 2, 3, 1)
            debug_rgbs = (debug_rgbs.detach().cpu().numpy() * 255).astype(np.uint8)
            for img_idx, img in enumerate(debug_rgbs):
                Image.fromarray(img).save(os.path.join(debug_dir, f"{img_idx:05d}.png"))

        print(f"Processing {uid}, index: {idx}")

        # Run inference
        rgb = inference_results(
            lhm,
            item,
            smplx_params,
            motion_seqs,
            smplx_params["root_pose"].shape[1],
            ref_imgs_bool=item["ref_imgs_bool"].unsqueeze(0),
            ori_space=True,
        )

        # Save results
        pred_path = os.path.join(save_path, f"view_{view:03d}", uid)
        os.makedirs(pred_path, exist_ok=True)

        for pred_idx, pred_image in enumerate(rgb):
            pred_name = f"{pred_idx:05d}.png"
            save_img_path = os.path.join(pred_path, pred_name)
            Image.fromarray(pred_image).save(save_img_path)
            print(f"Saved: {save_img_path}")


def parse_configs() -> Tuple[DictConfig, str]:
    """Parse configuration from environment variables and model config files.

    Returns:
        Tuple containing merged configuration object and model name string.

    Raises:
        ValueError: If MODEL_PATH environment variable is not set.
    """
    cli_cfg = OmegaConf.create()
    cfg = OmegaConf.create()

    # Extract model path from environment
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise ValueError("MODEL_PATH environment variable is required")

    model_path = model_path.rstrip("/")
    model_name = model_path.split("/")[-2]
    cli_cfg.model_name = model_path

    # Load training configuration if provided
    model_config_path = os.environ.get("APP_MODEL_CONFIG")
    if model_config_path:
        cfg_train = OmegaConf.load(model_config_path)

        # Extract basic configuration parameters
        cfg.source_size = cfg_train.dataset.source_image_res
        cfg.src_head_size = getattr(
            cfg_train.dataset, "src_head_size", DEFAULT_SRC_HEAD_SIZE
        )
        cfg.render_size = cfg_train.dataset.render_image.high

        # Construct output paths based on experiment configuration
        exp_id = os.path.basename(cli_cfg.model_name).split("_")[-1]
        relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            exp_id,
        )

        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", relative_path)
        cfg.image_dump = os.path.join("exps", "images", relative_path)
        cfg.video_dump = os.path.join("exps", "videos", relative_path)

    cfg.motion_video_read_fps = DEFAULT_FPS
    cfg.merge_with(cli_cfg)
    cfg.setdefault("logger", "INFO")

    if not cfg.get("model_name"):
        raise ValueError("model_name is required in configuration")

    return cfg, model_name


def get_parse() -> argparse.Namespace:
    """Parse command-line arguments for unified inference script.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="LHM++ Unified Inference Script - Supports video and benchmark modes"
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "-w", "--ckpt", required=True, help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "-v",
        "--view",
        type=int,
        default=16,
        help="Number of input views to use (default: 16 for video, 4 for benchmark)",
    )
    parser.add_argument(
        "-p",
        "--pre",
        type=str,
        required=True,
        help="Experiment name (must be key in DATASETS_CONFIG)",
    )
    parser.add_argument(
        "-m",
        "--motion",
        type=str,
        help="Path to directory containing motion sequence data (required for video mode)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "benchmark"],
        default="video",
        help='Inference mode: "video" for video generation, "benchmark" for benchmark evaluation',
    )
    parser.add_argument(
        "-s",
        "--split",
        type=int,
        default=1,
        help="Number of dataset splits for distributed inference (video mode only)",
    )
    parser.add_argument(
        "-g",
        "--gpus",
        type=int,
        default=0,
        help="GPU index for current process (video mode only)",
    )
    parser.add_argument(
        "--ori_space",
        action="store_true",
        help="Output in original video space instead of cropped space",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (process only 100 frames or save debug visualizations)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for LHM++ unified inference script.

    Supports two modes:
    - video: Generate videos with external motion sequences
    - benchmark: Evaluate on dataset's own sequences and save image frames
    """
    # Parse command-line arguments
    args = get_parse()

    # Configure environment variables
    os.environ.update(
        {
            "APP_ENABLED": "1",
            "APP_MODEL_CONFIG": args.config,
            "APP_TYPE": "infer.human_lrm_a4o",
            "NUMBA_THREADING_LAYER": "omp",
            "MODEL_PATH": args.ckpt,
        }
    )

    # Extract experiment information
    exp_name = args.pre

    # Validate experiment name
    if exp_name not in DATASETS_CONFIG:
        raise ValueError(
            f"Unknown experiment name '{exp_name}'. "
            f"Must be one of: {list(DATASETS_CONFIG.keys())}"
        )

    # Initialize accelerator and parse configuration
    accelerator = Accelerator()
    cfg, model_name = parse_configs()

    # Build model (skip in debug mode for benchmark)
    if args.mode == "benchmark" and args.debug:
        lhm = None
    else:
        lhm = _build_model(cfg)

    # Run inference based on mode
    if args.mode == "video":
        # Video generation mode
        if args.motion is None:
            raise ValueError("--motion is required for video mode")

        motion_path = args.motion.rstrip("/")
        motion_name = os.path.basename(motion_path)

        # Construct output path
        save_path = os.path.join(
            "./exps/reattached", f"{exp_name}-heursample", model_name, motion_name
        )
        os.makedirs(save_path, exist_ok=True)

        # Run video inference
        inference_video_mode(
            lhm=lhm,
            save_path=save_path,
            view=args.view,
            cfg=cfg,
            motion_path=motion_path,
            exp_name=exp_name,
            debug=args.debug,
            split=args.split,
            gpus=args.gpus,
            ori_space=args.ori_space,
        )

    elif args.mode == "benchmark":
        # Benchmark evaluation mode
        save_path = os.path.join(
            "./exps/validation", f"{exp_name}-heursample", model_name, "benchmark"
        )
        os.makedirs(save_path, exist_ok=True)

        # Run benchmark inference
        inference_benchmark_mode(
            lhm=lhm,
            save_path=save_path,
            view=args.view,  # Default to 4 for benchmark
            cfg=cfg,
            exp_name=exp_name,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
