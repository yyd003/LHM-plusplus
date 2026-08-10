# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-07-03 20:11:35
# @Function      : PF-LHM Inference Demo Code.
#                  Given specific motion sequences.
#                  We reenattach the motion sequences to the corresponding human body.

import os
import sys

sys.path.append("./")
import time

import cv2
import numpy as np
import torch

torch._dynamo.config.disable = True
import glob
import json
from typing import Dict, Optional, Tuple

import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from core.runners.infer.utils import (
    prepare_motion_seqs_eval,
)
from core.structures.bbox import Bbox
from core.utils.hf_hub import wrap_model_hub


def resize_with_padding(images, target_size):
    """
    Combine 4 images into a 2x2 grid, then resize with aspect ratio preserved,
    and pad with white to match the target size.

    Args:
        images: List[np.ndarray], each of shape (H, W), dtype usually uint8
        target_size: tuple (H1, W1)

    Returns:
        np.ndarray: Output image of shape (H1, W1), dtype uint8, padded with white (255)
    """
    assert len(images) == 4, "Exactly 4 images are required"

    H, W = images[0].shape[:2]
    assert all(
        img.shape[:2] == (H, W) for img in images
    ), "All images must have the same shape (H, W)"

    # 1. Concatenate into a 2H x 2W image (2x2 grid)
    top_row = np.hstack([images[0], images[1]])  # Shape: (H, 2W, C)
    bottom_row = np.hstack([images[2], images[3]])  # Shape: (H, 2W, C)
    combined = np.vstack([top_row, bottom_row])  # Shape: (2H, 2W, C)

    Hc, Wc, _ = combined.shape  # = (2H, 2W)

    target_h, target_w = target_size

    # 2. Resize with preserved aspect ratio
    scale_h = target_h / Hc
    scale_w = target_w / Wc
    scale = min(scale_h, scale_w)  # Use the smaller scale to avoid overflow

    new_h = int(Hc * scale)
    new_w = int(Wc * scale)

    # Resize the combined image
    resized = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 3. Pad with white (255) to reach target size (centered)
    padded = np.full((target_h, target_w, 3), 255, dtype=np.uint8)  # White background

    # Calculate padding offsets (center the resized image)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2

    # Place the resized image in the center
    padded[top : top + new_h, left : left + new_w] = resized

    return padded


DATASETS_CONFIG = {
    "eval": dict(
        root_dirs="/mnt/workspaces/rmbg/papers/heyuan/stablenorml/d6f8c7e1a2b3c4d5e6f7a8b9c0d1e2f3/tmp/video_human_datasets/LHM_video_dataset/",
        meta_path="./train_data/ClothVideo/label/valid_LHM_dataset_train_val_100.json",
    ),
    "dataset5": dict(
        root_dirs="/mnt/workspaces/dataset/video_human_datasets/selected_dataset_v5_tar/",
        meta_path="/mnt/workspaces/dataset/video_human_datasets/clean_labels/valid_selected_datasetv5_val_filter460-self-rotated-69.json",
    ),
    "dataset6": dict(
        root_dirs="/mnt/workspaces/dataset/video_human_datasets/selected_dataset_v6_tar/",
        meta_path="/mnt/workspaces/dataset/video_human_datasets/clean_labels/valid_selected_datasetv6_test_100-self-rotated-25.json",
    ),
    "synthetic": dict(
        root_dirs="/mnt/workspaces/dataset/video_human_datasets/synthetic_data_tar/",
        meta_path="/mnt/workspaces/dataset/video_human_datasets/clean_labels/valid_LHM_synthetic_dataset_val_17.json",
    ),
    "dataset5_train": dict(
        root_dirs="/mnt/workspaces/dataset/video_human_datasets/selected_dataset_v5_tar/",
        meta_path="/mnt/workspaces/dataset/video_human_datasets/clean_labels/valid_selected_datasetv5_train_filter40K-self-rotated-7147.json",
    ),
    "dataset6_train": dict(
        root_dirs="/mnt/workspaces/dataset/video_human_datasets/selected_dataset_v6_tar/",
        meta_path="/mnt/workspaces/dataset/video_human_datasets/clean_labels/valid_selected_datasetv6_train_5W-self-rotated-11341.json",
    ),
    "eval_train": dict(
        root_dirs="/mnt/workspaces/rmbg/papers/heyuan/stablenorml/d6f8c7e1a2b3c4d5e6f7a8b9c0d1e2f3/tmp/video_human_datasets/LHM_video_dataset/",
        meta_path="/mnt/workspaces/rmbg/papers/heyuan/stablenorml/d6f8c7e1a2b3c4d5e6f7a8b9c0d1e2f3/tmp/video_human_datasets/clean_labels/valid_LHM_dataset_train_filter_16W.json",
    ),
    "in_the_wild": dict(
        root_dirs="/mnt/workspaces/rmbg/papers/heyuan/stablenorml/d6f8c7e1a2b3c4d5e6f7a8b9c0d1e2f3/PFLHM-Causal-Video/",
        meta_path="/mnt/workspaces/rmbg/papers/heyuan/stablenorml/d6f8c7e1a2b3c4d5e6f7a8b9c0d1e2f3/tmp/video_human_datasets/clean_labels/eval_sparse_lhm_wild.json",
    ),
    "in_the_wild_people_snapshot": dict(
        root_dirs="/mnt/workspaces/dataset/people_snapshot/",
        meta_path="/mnt/workspaces/dataset/people_snapshot/peoplesnapshot.json",
    ),
    "in_the_wild_rec_mv": dict(
        root_dirs="/mnt/workspaces/rmbg/papers/heyuan/stablenorml/d6f8c7e1a2b3c4d5e6f7a8b9c0d1e2f3/rec_mv_dataset",
        meta_path="/mnt/workspaces/rmbg/papers/heyuan/stablenorml/d6f8c7e1a2b3c4d5e6f7a8b9c0d1e2f3/rec_mv_dataset/rec_mv_dataset.json",
    ),
    "in_the_wild_mvhumannet": dict(
        root_dirs="/mnt/workspaces/rmbg/papers/heyuan/stablenorml/d6f8c7e1a2b3c4d5e6f7a8b9c0d1e2f3/mvhumannet/",
        meta_path="/mnt/workspaces/rmbg/papers/heyuan/stablenorml/d6f8c7e1a2b3c4d5e6f7a8b9c0d1e2f3/mvhumannet/mvhumannet.json",
    ),
    "dataset_train_real": dict(
        root_dirs="/mnt/workspaces/rmbg/papers/heyuan/stablenorml/d6f8c7e1a2b3c4d5e6f7a8b9c0d1e2f3/tmp/video_human_datasets/LHM_video_dataset/",
        meta_path="/mnt/workspaces/rmbg/papers/heyuan/stablenorml/d6f8c7e1a2b3c4d5e6f7a8b9c0d1e2f3/tmp/video_human_datasets/clean_labels/valid_LHM_dataset_train_filter_16W.json",
    ),
    "dataset5_train_real": dict(
        root_dirs="/mnt/workspaces/dataset/video_human_datasets/selected_dataset_v5_tar/",
        meta_path="/mnt/workspaces/dataset/video_human_datasets/clean_labels/valid_selected_datasetv5_train_filter40K.json",
    ),
    "dataset6_train_real": dict(
        root_dirs="/mnt/workspaces/dataset/video_human_datasets/selected_dataset_v6_tar/",
        meta_path="/mnt/workspaces/dataset/video_human_datasets/clean_labels/valid_selected_datasetv6_train_5W.json",
    ),
    "web_dresscode": dict(
        root_dirs="/mnt/workspaces/datasets/lhm_human_datasets/DressCode/",  # heyuan generation dataset.
        meta_path=None,
    ),
    "hweb_hero": dict(
        root_dirs="/mnt/workspaces/datasets/lhm_human_datasets/pinterest_download_0903_gen_full_fixed_view_random_pose_2_filtered",  # heyuan generation dataset.
        meta_path=None,
    ),
}


def obtain_motion_sequence(motion_seqs):
    motion_seqs = sorted(glob.glob(os.path.join(motion_seqs, "*.json")))

    smplx_list = []

    for motion in motion_seqs:

        with open(motion) as reader:
            smplx_params = json.load(reader)

        flame_path = motion.replace("smplx_params", "flame_params")
        if os.path.exists(flame_path):
            with open(flame_path) as reader:
                flame_params = json.load(reader)
            smplx_params["expr"] = torch.FloatTensor(flame_params["expcode"])

            # replace with flame's jaw_pose
            smplx_params["jaw_pose"] = torch.FloatTensor(flame_params["posecode"][3:])
            smplx_params["leye_pose"] = torch.FloatTensor(flame_params["eyecode"][:3])
            smplx_params["reye_pose"] = torch.FloatTensor(flame_params["eyecode"][3:])
        else:
            smplx_params["expr"] = torch.FloatTensor([0.0] * 100)

        smplx_list.append(smplx_params)

    return smplx_list


def prepare_canonical_motion_sequence(
    motion_path: str,
    cfg: Dict,
    *,
    debug: bool = False,
) -> Dict:
    """Load the current motion directory layout with the current inference API."""
    smplx_path = os.path.join(motion_path, "smplx_params")
    mask_path = os.path.join(motion_path, "samurai_seg")
    bbox_json_path = os.path.join(motion_path, "bbox", "bbox.json")

    smplx_files = sorted(glob.glob(os.path.join(smplx_path, "*.json")))
    if not smplx_files:
        raise FileNotFoundError(f"No SMPL-X JSON files found under {smplx_path}")

    bbox_dict = {}
    if os.path.isfile(bbox_json_path):
        with open(bbox_json_path) as reader:
            bbox_dict = json.load(reader)

    motion_ids = [os.path.splitext(os.path.basename(path))[0] for path in smplx_files]
    mask_paths = [os.path.join(mask_path, f"{motion_id}.png") for motion_id in motion_ids]
    bbox_list = []
    for motion_id in motion_ids:
        bbox = bbox_dict.get(motion_id)
        bbox_list.append(Bbox(bbox, mode="xywh").to_whwh() if bbox is not None else None)

    valid_motion_ids = [
        int(motion_id) if motion_id.isdigit() else motion_id
        for motion_id, current_mask in zip(motion_ids, mask_paths)
        if os.path.isfile(current_mask)
    ]
    motion = prepare_motion_seqs_eval(
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
        motion_size=100 if debug else 1000,
        specific_id_list=None,
    )
    motion["motion_id"] = valid_motion_ids[: motion["render_c2ws"].shape[1]]
    return motion


def _build_model(cfg):
    from core.models import model_dict

    hf_model_cls = wrap_model_hub(model_dict["human_lrm_a4o"])
    model = hf_model_cls.from_pretrained(cfg.model_name)

    return model


def get_smplx_params(data, device):
    smplx_params = {}
    smplx_keys = [
        "root_pose",
        "body_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "lhand_pose",
        "rhand_pose",
        "expr",
        "trans",
        "betas",
    ]
    for k, v in data.items():
        if k in smplx_keys:
            # print(k, v.shape)
            smplx_params[k] = data[k].unsqueeze(0).to(device)
    return smplx_params


def animation_infer(
    renderer,
    gs_model_list,
    query_points,
    smplx_params,
    render_c2ws,
    render_intrs,
    render_bg_colors,
) -> dict:
    """Render animation frames in parallel without redundant computations.

    Args:
        renderer: The rendering engine
        gs_model_list: List of Gaussian models
        query_points: 3D query points
        smplx_params: SMPL-X parameters
        render_c2ws: Camera-to-world matrices
        render_intrs: Intrinsic camera parameters
        render_bg_colors: Background colors

    Returns:
        Dictionary of rendered results (rgb, mask, depth, etc.)
    """

    render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(
        render_intrs[0, 0, 0, 2] * 2
    )
    # render target views
    render_res_list = []
    num_views = render_c2ws.shape[1]

    start_time = time.time()

    # render target views
    render_res_list = [
        renderer.forward_animate_gs(
            gs_model_list,
            query_points,
            renderer.get_single_view_smpl_data(smplx_params, view_idx),
            render_c2ws[:, view_idx : view_idx + 1],
            render_intrs[:, view_idx : view_idx + 1],
            render_h,
            render_w,
            render_bg_colors[:, view_idx : view_idx + 1],
        )
        for view_idx in range(num_views)
    ]

    # Calculate and print timing information
    avg_time = (time.time() - start_time) / num_views
    print(f"Average time per frame: {avg_time:.4f}s")

    # Aggregate and format results
    out = defaultdict(list)
    for res in render_res_list:
        for k, v in res.items():
            out[k].append(v.detach().cpu() if isinstance(v, torch.Tensor) else v)

    # Post-process tensor outputs
    for k, v in out.items():
        if isinstance(v[0], torch.Tensor):
            out[k] = torch.concat(v, dim=1)
            if k in {"comp_rgb", "comp_mask", "comp_depth"}:
                out[k] = out[k][0].permute(
                    0, 2, 3, 1
                )  # [1, Nv, 3, H, W] -> [Nv, H, W, 3]

    return out


@torch.no_grad()
def inference_results(
    lhm: torch.nn.Module,
    batch: Dict,
    smplx_params: Dict,
    motion_seq: Dict,
    camera_size: int = 40,
    ref_imgs_bool=None,
    batch_size: int = 40,
    device: str = "cuda",
) -> np.ndarray:
    """Perform inference on a motion sequence in batches to avoid OOM."""

    offset_list = motion_seq["offset_list"]
    ori_h, ori_w = motion_seq["ori_size"]

    output_rgb = torch.ones((ori_h, ori_w, 3))  # draw board

    # Extract model outputs
    (
        gs_model_list,
        query_points,
        transform_mat_neutral_pose,
        gs_hidden_features,
        image_latents,
        motion_emb,
        pos_emb,
    ) = lhm.infer_single_view(
        batch["source_rgbs"].unsqueeze(0).to(device),
        None,
        None,
        render_c2ws=motion_seq["render_c2ws"].to(device),
        render_intrs=motion_seq["render_intrs"].to(device),
        render_bg_colors=motion_seq["render_bg_colors"].to(device),
        smplx_params={k: v.to(device) for k, v in smplx_params.items()},
        ref_imgs_bool=ref_imgs_bool,
    )

    # Prepare parameters that are constant across batches
    batch_smplx_params = {
        "betas": smplx_params["betas"].to(device),
        "transform_mat_neutral_pose": transform_mat_neutral_pose,
    }

    keys = [
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

    batch_list = []
    batch_mask_list = []
    for batch_i in range(0, camera_size, batch_size):
        print(
            f"Processing batch {batch_i//batch_size + 1}/{(camera_size + batch_size - 1)//batch_size}"
        )

        # Update batch-specific parameters
        batch_smplx_params.update(
            {
                key: motion_seq["smplx_params"][key][
                    :, batch_i : batch_i + batch_size
                ].to(device)
                for key in keys
            }
        )

        # Perform inference
        batch_rgb, batch_mask = lhm.animation_infer(
            gs_model_list,
            query_points,
            batch_smplx_params,
            render_c2ws=motion_seq["render_c2ws"][:, batch_i : batch_i + batch_size].to(
                device
            ),
            render_intrs=motion_seq["render_intrs"][
                :, batch_i : batch_i + batch_size
            ].to(device),
            render_bg_colors=motion_seq["render_bg_colors"][
                :, batch_i : batch_i + batch_size
            ].to(device),
            gs_hidden_features=gs_hidden_features,
            image_latents=image_latents,
            motion_emb=motion_emb,
            pos_emb=pos_emb,
            offset_list=offset_list[batch_i : batch_i + batch_size],
            mask_seqs=motion_seq["masks"][batch_i : batch_i + batch_size],
            output_rgb=output_rgb,
        )

        # Convert and store results
        batch_list.append((batch_rgb.clamp(0, 1) * 255).to(torch.uint8).numpy())
        batch_mask_list.append((batch_mask.clamp(0, 1) * 255).to(torch.uint8).numpy())

    return np.concatenate(batch_list, axis=0), np.concatenate(batch_mask_list, axis=0)


@torch.no_grad()
def inference_gs_model(
    lhm: torch.nn.Module,
    batch: Dict,
    smplx_params: Dict,
    motion_seq: Dict,
    camera_size: int = 40,
    ref_imgs_bool=None,
    batch_size: int = 40,
    device: str = "cuda",
) -> np.ndarray:
    """Perform inference on a motion sequence in batches to avoid OOM."""

    # Extract model outputs
    (
        gs_model_list,
        query_points,
        transform_mat_neutral_pose,
        gs_hidden_features,
        image_latents,
        motion_emb,
    ) = lhm.infer_single_view(
        batch["source_rgbs"].unsqueeze(0).to(device),
        None,
        None,
        render_c2ws=motion_seq["render_c2ws"].to(device),
        render_intrs=motion_seq["render_intrs"].to(device),
        render_bg_colors=motion_seq["render_bg_colors"].to(device),
        smplx_params={k: v.to(device) for k, v in smplx_params.items()},
        ref_imgs_bool=ref_imgs_bool,
    )

    # Prepare parameters that are constant across batches
    batch_smplx_params = {
        "betas": smplx_params["betas"].to(device),
        "transform_mat_neutral_pose": transform_mat_neutral_pose,
    }

    keys = [
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

    batch_list = []
    for batch_i in range(0, camera_size, batch_size):
        print(
            f"Processing batch {batch_i//batch_size + 1}/{(camera_size + batch_size - 1)//batch_size}"
        )

        # Update batch-specific parameters
        batch_smplx_params.update(
            {
                key: motion_seq["smplx_params"][key][
                    :, batch_i : batch_i + batch_size
                ].to(device)
                for key in keys
            }
        )

        # Perform inference
        gs_model = lhm.inference_gs(
            gs_model_list,
            query_points,
            batch_smplx_params,
            render_c2ws=motion_seq["render_c2ws"][:, batch_i : batch_i + batch_size].to(
                device
            ),
            render_intrs=motion_seq["render_intrs"][
                :, batch_i : batch_i + batch_size
            ].to(device),
            render_bg_colors=motion_seq["render_bg_colors"][
                :, batch_i : batch_i + batch_size
            ].to(device),
            gs_hidden_features=gs_hidden_features,
            image_latents=image_latents,
            motion_emb=motion_emb,
        )
        return gs_model


@torch.no_grad()
def lhm_validation_inference(
    lhm: Optional[torch.nn.Module],
    save_path: str,
    view: int = 16,
    cfg: Optional[Dict] = None,
    motion_path: Optional[str] = None,
    exp_name: str = "eval",
    debug: bool = False,
    split: int = 1,
    gpus: int = 0,
) -> None:
    """Run validation inference on the model."""
    if lhm is not None:
        lhm.cuda().eval()

    assert motion_path is not None
    cfg = cfg or {}

    # Setup paths
    gt_save_path = os.path.join(os.path.dirname(save_path), "gt")
    gt_mask_save_path = os.path.join(os.path.dirname(save_path), "mask")
    os.makedirs(gt_save_path, exist_ok=True)
    os.makedirs(gt_mask_save_path, exist_ok=True)

    # Load dataset
    dataset_config = DATASETS_CONFIG[exp_name]
    kwargs = {}
    if exp_name == "eval" or exp_name == "eval_train":
        from core.datasets.video_human_lhm_dataset_a4o import (
            VideoHumanLHMA4ODatasetEval as VideoDataset,
        )
    elif "in_the_wild" in exp_name:
        from core.datasets.video_in_the_wild_dataset import (
            VideoInTheWildEval as VideoDataset,
        )

        kwargs["heuristic_sampling"] = True
    elif "hweb" in exp_name:
        from core.datasets.video_in_the_wild_web_dataset import (
            WebInTheWildHeurEval as VideoDataset,
        )

        kwargs["heuristic_sampling"] = False
    elif "web" in exp_name:
        from core.datasets.video_in_the_wild_web_dataset import (
            WebInTheWildEval as VideoDataset,
        )

        kwargs["heuristic_sampling"] = False
    else:
        from core.datasets.video_human_dataset_a4o import (
            VideoHumanA4ODatasetEval as VideoDataset,
        )

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
            dict(name="PadRatioWithScale", target_ratio=5 / 3, tgt_max_size_list=[840]),
            dict(name="ToTensor"),
        ],
        **kwargs,
    )

    motion_seqs = prepare_canonical_motion_sequence(motion_path, cfg, debug=debug)

    motion_id = motion_seqs["motion_id"]

    # split_dataset
    dataset_size = len(dataset)
    bins = int(np.ceil(dataset_size / split))

    for idx in range(bins * gpus, bins * (gpus + 1)):
        try:
            item = dataset.__getitem__(idx, view)
        except:
            continue

        uid = item["uid"]
        save_folder = os.path.join(save_path, uid)
        os.makedirs(save_folder, exist_ok=True)

        print(f"Processing {uid}, idx: {idx}")

        video_dir = os.path.join(save_path, f"view_{view:03d}")
        os.makedirs(video_dir, exist_ok=True)

        # Update parameters and perform inference
        motion_seqs["smplx_params"]["betas"] = item["betas"].unsqueeze(0)
        try:
            rgbs, masks = inference_results(
                lhm,
                item,
                motion_seqs["smplx_params"],
                motion_seqs,
                camera_size=motion_seqs["smplx_params"]["root_pose"].shape[1],
                ref_imgs_bool=item["ref_imgs_bool"].unsqueeze(0),
            )
        except:
            print("Error in infering")
            continue

        for rgb, mask, mi in zip(rgbs, masks, motion_id):
            pred_image = Image.fromarray(rgb)
            pred_mask = Image.fromarray(mask)
            idx = mi
            pred_name = f"rgb_{idx:05d}.png"
            mask_pred_name = f"mask_{idx:05d}.png"
            save_img_path = os.path.join(save_folder, pred_name)
            save_mask_path = os.path.join(save_folder, mask_pred_name)
            pred_image.save(save_img_path)
            pred_mask.save(save_mask_path)


@torch.no_grad()
def lhm_validation_inference_gs(
    lhm: Optional[torch.nn.Module],
    save_path: str,
    view: int = 16,
    cfg: Optional[Dict] = None,
    motion_path: Optional[str] = None,
    exp_name: str = "eval",
    debug: bool = False,
    split: int = 1,
    gpus: int = 0,
) -> None:
    """Run validation inference on the model."""
    if lhm is not None:
        lhm.cuda().eval()

    assert motion_path is not None
    cfg = cfg or {}

    # Load dataset
    dataset_config = DATASETS_CONFIG[exp_name]
    kwargs = {}
    if exp_name == "eval" or exp_name == "eval_train":
        from core.datasets.video_human_lhm_dataset_a4o import (
            VideoHumanLHMA4ODatasetEval as VideoDataset,
        )
    elif "in_the_wild" in exp_name:
        from core.datasets.video_in_the_wild_dataset import (
            VideoInTheWildEval as VideoDataset,
        )

        kwargs["heuristic_sampling"] = True
    else:
        from core.datasets.video_human_dataset_a4o import (
            VideoHumanA4ODatasetEval as VideoDataset,
        )

    dataset = VideoDataset(
        root_dirs=dataset_config["root_dirs"],
        meta_path=dataset_config["meta_path"],
        sample_side_views=7,
        render_image_res_low=420,
        render_image_res_high=420,
        render_region_size=(700, 420),
        source_image_res=420,
        debug=False,
        use_flame=True,
        ref_img_size=view,
        womask=True,
        is_val=True,
        **kwargs,
    )

    motion_seqs = prepare_canonical_motion_sequence(motion_path, cfg, debug=True)

    # split_dataset

    dataset_size = len(dataset)
    bins = int(np.ceil(dataset_size / split))

    for idx in range(bins * gpus, bins * (gpus + 1)):

        try:
            item = dataset.__getitem__(idx, view)
        except:
            continue

        uid = item["uid"]
        print(f"Processing {uid}, idx: {idx}")

        gs_dir = os.path.join(save_path, f"view_{view:03d}")
        os.makedirs(gs_dir, exist_ok=True)
        gs_file_path = os.path.join(gs_dir, f"{uid}.ply")

        if os.path.exists(gs_file_path):
            continue

        # Update parameters and perform inference
        motion_seqs["smplx_params"]["betas"] = item["betas"]

        gs_model = inference_gs_model(
            lhm,
            item,
            motion_seqs["smplx_params"],
            motion_seqs,
            camera_size=motion_seqs["smplx_params"]["root_pose"].shape[1],
        )
        print(f"generated GS Model!")

        gs_model.save_ply(gs_file_path)


def parse_configs() -> Tuple[DictConfig, str]:
    """Parse configuration from environment variables and model config files.

    Returns:
        Tuple containing:
            - Merged configuration object
            - Model name extracted from MODEL_PATH
    """
    cli_cfg = OmegaConf.create()
    cfg = OmegaConf.create()

    # Parse from environment variables
    model_path = os.environ.get("MODEL_PATH", "").rstrip("/")
    if not model_path:
        raise ValueError("MODEL_PATH environment variable is required")

    cli_cfg.model_name = model_path
    model_name = model_path.split("/")[-2]  # Assuming standard path structure

    # Load model config if available
    if model_config := os.environ.get("APP_MODEL_CONFIG"):
        cfg_train = OmegaConf.load(model_config)

        # Set basic configuration parameters
        cfg.update(
            {
                "source_size": cfg_train.dataset.source_image_res,
                "src_head_size": getattr(cfg_train.dataset, "src_head_size", 112),
                "render_size": cfg_train.dataset.render_image.high,
                "motion_video_read_fps": 30,
                "logger": "INFO",
            }
        )

        # Set paths based on experiment configuration
        exp_id = os.path.basename(model_path).split("_")[-1]
        relative_path = os.path.join(
            cfg_train.experiment.parent, cfg_train.experiment.child, exp_id
        )

        cfg.update(
            {
                "save_tmp_dump": os.path.join("exps", "save_tmp", relative_path),
                "image_dump": os.path.join("exps", "images", relative_path),
                "video_dump": os.path.join("exps", "videos", relative_path),
            }
        )

    # Merge CLI config and validate
    cfg.merge_with(cli_cfg)
    if not cfg.get("model_name"):
        raise ValueError("model_name is required in configuration")

    return cfg, model_name


def get_parse():
    import argparse

    parser = argparse.ArgumentParser(description="Inference_Config")
    parser.add_argument("-c", "--config", required=True, help="config file path")
    parser.add_argument("-w", "--ckpt", required=True, help="model checkpoint")
    parser.add_argument("-v", "--view", help="input views", default=16, type=int)
    parser.add_argument("-p", "--pre", help="exp_name", type=str)
    parser.add_argument("-m", "--motion", help="motion_path", type=str)
    parser.add_argument(
        "--dataset-root",
        type=str,
        help="override DATASETS_CONFIG[--pre].root_dirs",
    )
    parser.add_argument(
        "--dataset-meta",
        type=str,
        help="override DATASETS_CONFIG[--pre].meta_path",
    )
    parser.add_argument(
        "-s",
        "--split",
        help="split_dataset, used for distribution inference.",
        type=int,
        default=1,
    )
    parser.add_argument("-g", "--gpus", help="current gpu id", type=int, default=0)
    parser.add_argument("--gs", help="output gaussian model", action="store_true")
    parser.add_argument(
        "--output", help="output_cano_folder", default="./debug/cano_output", type=str
    )
    parser.add_argument("--debug", help="motion_path", action="store_true")
    args = parser.parse_args()
    return args


def main():

    args = get_parse()

    os.environ.update(
        {
            "APP_ENABLED": "1",
            "APP_MODEL_CONFIG": args.config,  # choice from MODEL_CARD
            "APP_TYPE": "infer.human_lrm_a4o",
            "APP_TYPE": "infer.human_lrm_a4o",
            "NUMBA_THREADING_LAYER": "omp",
            "MODEL_PATH": args.ckpt,
        }
    )

    exp_name = args.pre
    if not exp_name:
        raise ValueError("--pre must name a dataset configuration")
    if not args.motion:
        raise ValueError("--motion must point to a motion directory")
    motion = args.motion.rstrip("/")
    if not motion:
        raise ValueError("--motion cannot be the filesystem root")
    cfg, model_name = parse_configs()
    # save_path = os.path.join('./exps/validation/public_benchmark', model_name)
    if exp_name not in DATASETS_CONFIG:
        raise ValueError(
            f"Unknown dataset config {exp_name!r}; choose one of {sorted(DATASETS_CONFIG)}"
        )
    if args.dataset_root is not None:
        DATASETS_CONFIG[exp_name]["root_dirs"] = args.dataset_root
    if args.dataset_meta is not None:
        DATASETS_CONFIG[exp_name]["meta_path"] = args.dataset_meta

    dataset_config = DATASETS_CONFIG[exp_name]
    if not dataset_config.get("root_dirs"):
        raise ValueError("A dataset root is required; pass --dataset-root")
    if not os.path.exists(dataset_config["root_dirs"]):
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_config['root_dirs']}. "
            "Pass --dataset-root to override the legacy path."
        )
    if dataset_config.get("meta_path") and not os.path.isfile(dataset_config["meta_path"]):
        raise FileNotFoundError(
            f"Dataset metadata does not exist: {dataset_config['meta_path']}. "
            "Pass --dataset-meta to override the legacy path."
        )
    if not os.path.isdir(motion):
        raise FileNotFoundError(f"Motion directory does not exist: {motion}")

    output_path = args.output

    lhm = _build_model(cfg)

    if not args.gs:
        os.makedirs(output_path, exist_ok=True)
        lhm_validation_inference(
            lhm,
            output_path,
            args.view,
            cfg,
            motion,
            exp_name,
            debug=args.debug,
            split=args.split,
            gpus=args.gpus,
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
