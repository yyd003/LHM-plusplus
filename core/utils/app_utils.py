# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : LHM++ Gradio App helper utilities

import glob
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v3 as iio
import numpy as np
from omegaconf import DictConfig
from PIL import Image

from core.datasets.data_utils import src_center_crop_according_to_mask
from core.runners.infer.utils import prepare_motion_seqs_eval
from core.structures.bbox import Bbox
from scripts.inference.utils import obtain_motion_sequence

# Default constants for gallery/examples
MV_IMAGES_THRESHOLD = 8
MV_IMAGES_AIGC = 4  # example_aigc_images has 4 images per case
NUM_VIEWS = 2

# First-page pinned cases (in order)
PINNED_CASES = [
    "00000_yuliang",
    "anran_purble",
    "exstrimalik_crop",
    "11a71df60779",
    "c81e728d9d4c",
    "testimg_000045",
    "IMG_9334",
    "testimg_000038",
    "8f14e45fceea",
]


def prepare_aigc_example(
    assets_dir: str = "assets/example_aigc_images",
    num_views: int = NUM_VIEWS,
    max_images: int = MV_IMAGES_AIGC,
) -> List[Image.Image]:
    """Prepare AIGC example images (4 images per case) for the gallery.

    Same format as prepare_multi_example but with max_images=4, supports .jpg.
    """
    if not os.path.exists(assets_dir):
        return []
    files = [f for f in os.listdir(assets_dir) if not f.startswith("_")]
    multi_case = list(
        set(
            [
                "_".join(i.rsplit("_", 1)[:-1])
                for i in files
                if re.search(r"_\d{3}\.(png|jpg|jpeg)$", i, re.I)
            ]
        )
    )
    multi_case = sorted(multi_case)
    images = []
    for case in multi_case:
        _images = []
        for i in range(0, max_images):
            path = None
            for ext in (".png", ".jpg", ".jpeg"):
                p = f"{assets_dir}/{case}_{i:03d}{ext}"
                if os.path.exists(p):
                    path = p
                    break
            if path is not None:
                img = Image.open(path)
                img = np.array(img.convert("RGB"))
                alpha = np.ones_like(img[:, :, 0:1]) * 255
                alpha[:, 0, 0] = 0
                _images.append(np.concatenate([img, alpha], axis=2))
        if len(_images) > 0:
            _images_count = len(_images)
            padding_needed = (
                num_views - (_images_count % num_views)
                if _images_count % num_views != 0
                else 0
            )
            h, w, c = _images[0].shape
            for _ in range(padding_needed):
                _images.append(np.ones((h, w, c), dtype=np.uint8) * 255)
            row_imgs = []
            each_row = (_images_count + padding_needed) // num_views
            for i in range(0, len(_images), each_row):
                row_imgs.append(np.concatenate(_images[i : i + each_row], axis=1))
            cat_img = np.concatenate(row_imgs, axis=0)
            images.append(Image.fromarray(cat_img))
    return images


def prepare_examples_ordered() -> List[Image.Image]:
    """Prepare examples with pinned first 6, rest shuffled.

    First 6 cases (in order): 00000_yuliang, anran_purble, exstrimalik_crop,
    11a71df60779, c81e728d9d4c, 8f14e45fceea. Remaining cases are randomized.
    """

    def _with_cases(assets_dir: str, max_images: int) -> List[Tuple[str, Image.Image]]:
        if not os.path.exists(assets_dir):
            return []
        files = [f for f in os.listdir(assets_dir) if not f.startswith("_")]
        multi_case = sorted(
            set(
                [
                    "_".join(i.rsplit("_", 1)[:-1])
                    for i in files
                    if re.search(r"_\d{3}\.(png|jpg|jpeg)$", i, re.I)
                ]
            )
        )
        result = []
        for case in multi_case:
            _images = []
            for i in range(max_images):
                path = None
                for ext in (".png", ".jpg", ".jpeg"):
                    p = f"{assets_dir}/{case}_{i:03d}{ext}"
                    if os.path.exists(p):
                        path = p
                        break
                if path:
                    img = Image.open(path)
                    arr = np.array(img.convert("RGB"))
                    alpha = np.ones_like(arr[:, :, 0:1]) * 255
                    alpha[:, 0, 0] = 0
                    _images.append(np.concatenate([arr, alpha], axis=2))
            if _images:
                n, nv = len(_images), NUM_VIEWS
                pad = (nv - n % nv) if n % nv else 0
                h, w, c = _images[0].shape
                for _ in range(pad):
                    _images.append(np.ones((h, w, c), dtype=np.uint8) * 255)
                each_row = (n + pad) // nv
                rows = [
                    np.concatenate(_images[i : i + each_row], axis=1)
                    for i in range(0, len(_images), each_row)
                ]
                result.append((case, Image.fromarray(np.concatenate(rows, axis=0))))
        return result

    multi = _with_cases("assets/example_multi_images", MV_IMAGES_THRESHOLD)
    aigc = _with_cases("assets/example_aigc_images", MV_IMAGES_AIGC)
    all_items = multi + aigc
    by_case = {c: img for c, img in all_items}
    pinned = [by_case[c] for c in PINNED_CASES if c in by_case]
    rest_cases = [c for c, _ in all_items if c not in PINNED_CASES]
    random.seed(220019047)
    random.shuffle(rest_cases)
    rest = [by_case[c] for c in rest_cases]
    return pinned + rest


def prior_check() -> None:
    """Check if prior model exists, download if needed."""
    pass


def prepare_multi_example(
    assets_dir: str = "assets/example_multi_images",
    num_views: int = NUM_VIEWS,
    max_images: int = MV_IMAGES_THRESHOLD,
) -> List[Image.Image]:
    """Prepare multi-view example images for the gallery.

    Loads example images from assets dir, organizes them into multi-view
    arrangements, and creates concatenated preview images.

    Returns:
        List of PIL Images containing multi-view arrangements.
    """
    if not os.path.exists(assets_dir):
        return []
    multi_case = list(
        set(["_".join(i.split("_")[:-1]) for i in os.listdir(assets_dir)])
    )
    multi_case = sorted(multi_case)
    images = []
    for case in multi_case:
        _images = []
        for i in range(0, max_images):
            path = f"{assets_dir}/{case}_{i:03d}.png"
            if os.path.exists(path):
                img = Image.open(path)
                img = np.array(img)
                alpha = np.ones_like(img[:, :, 0:1]) * 255
                alpha[:, 0, 0] = 0
                _images.append(np.concatenate([img, alpha], axis=2))
        if len(_images) > 0:
            _images_count = len(_images)
            padding_needed = (
                num_views - (_images_count % num_views)
                if _images_count % num_views != 0
                else 0
            )
            h, w, c = _images[0].shape
            for _ in range(padding_needed):
                _images.append(np.ones((h, w, c), dtype=np.uint8) * 255)
            row_imgs = []
            each_row = (_images_count + padding_needed) // num_views
            for i in range(0, len(_images), each_row):
                row_imgs.append(np.concatenate(_images[i : i + each_row], axis=1))
            cat_img = np.concatenate(row_imgs, axis=0)
            images.append(Image.fromarray(cat_img))
    return images


def split_image(
    image: Image.Image,
    num_views: int = NUM_VIEWS,
) -> List[Image.Image]:
    """Split a multi-view concatenated image into separate view images.

    Args:
        image: A concatenated image containing multiple views.
        num_views: Number of views to split horizontally.

    Returns:
        List of individual preprocessed view images.
    """
    image_array = np.array(image)
    h, w, c = image_array.shape
    img_list = np.split(image_array, num_views, axis=0)
    image_array = np.concatenate(img_list, axis=1)
    alpha = image_array[..., 3]
    rgb = image_array[..., :3]
    start_pos = np.where(alpha[0, :-1] & alpha[0, 1:] == 0)[0].tolist()
    end_pos = start_pos[1::2]
    start_pos = start_pos[::2]
    end_pos.append(start_pos[-1] + (end_pos[0] - start_pos[0]))
    images = []
    for s, e in zip(start_pos, end_pos):
        images.append(Image.fromarray(rgb[:, s : e + 1]))

    return images


def obtain_ref_imgs(
    imgs: List[Any],
    ref_view: int = 8,
) -> List[np.ndarray]:
    """Obtain reference images from selected input images via uniform sampling.

    Args:
        imgs: List of image objects (or tuples) from gallery selection.
        ref_view: Number of reference images to select.

    Returns:
        List of numpy arrays representing the evenly sampled reference images.
    """
    patches = [np.asarray(img[0]) for img in imgs]
    n = len(patches)
    assert n > 0, "No images provided"
    ref_view = min(ref_view, n)
    if ref_view == 1:
        return [patches[0]]
    indices = np.linspace(0, n - 1, ref_view, dtype=int)
    return [patches[i] for i in indices]


def obtain_ref_imgs_from_videos(
    video_path: str,
    ref_view: int,
    dataset_pipeline: Any,
    cfg: Optional[DictConfig] = None,
) -> List[np.ndarray]:
    """Extract evenly sampled reference frames from video.

    Args:
        video_path: Path to the input video file.
        ref_view: Number of frames to extract.
        dataset_pipeline: Pipeline for processing images.
        cfg: Optional configuration object.

    Returns:
        List of numpy arrays with shape (H, W, 3).

    Raises:
        RuntimeError: If video cannot be read.
        Exception: If no frames are found in the video.
    """
    u2net_home = os.path.abspath(os.path.join(".", "pretrained_models", "u2net"))
    os.makedirs(u2net_home, exist_ok=True)
    os.environ.setdefault("U2NET_HOME", u2net_home)
    from rembg import remove

    def human_centers_crop(
        imgs: List[np.ndarray],
        cfg: Optional[DictConfig],
    ) -> List[np.ndarray]:
        rgbs = []
        for img in imgs:
            bgr_img = img[:, :, ::-1]
            try:
                mask = remove(bgr_img)
                img, mask, _, _, _ = src_center_crop_according_to_mask(
                    img,
                    mask[..., -1] / 255.0,
                    aspect_standard=5.0 / 3,
                    enlarge_ratio=[1.0, 1.0],
                    head_bbox=None,
                )
            except Exception:
                pass
            rgbs.append(img / 255.0)
        rgbs = dataset_pipeline(rgbs)
        rgbs = [(rgb * 255).astype(np.uint8) for rgb in rgbs]
        return rgbs

    try:
        video_reader = iio.imiter(video_path)
        frames = [frame for frame in video_reader]
    except Exception as e:
        raise RuntimeError(f"Could not read video {video_path}: {e}") from e

    num_frames = len(frames)
    if num_frames == 0:
        raise ValueError("No frames found in video upload.")

    if num_frames < ref_view:
        indices = list(range(num_frames))
        indices += [num_frames - 1] * (ref_view - num_frames)
    else:
        indices = [int(i * num_frames / ref_view) for i in range(ref_view)]

    selected_frames = [frames[idx].copy() for idx in indices]
    selected_frames = human_centers_crop(selected_frames, cfg)
    return selected_frames


def prepare_input_and_output(
    image: Optional[List[Any]],
    video: Optional[str],
    ref_view: int,
    video_params: str,
    working_dir: Any,
    dataset_pipeline: Any,
    cfg: Optional[DictConfig] = None,
) -> Tuple[List[np.ndarray], str, str, str, str]:
    """Prepare input images and output paths for processing.

    Returns:
        Tuple of (imgs, sample_path, motion_seqs_dir, dump_image_dir, dump_video_path).
    """
    if video is not None:
        imgs = obtain_ref_imgs_from_videos(video, ref_view, dataset_pipeline, cfg)
    else:
        imgs = obtain_ref_imgs(image, ref_view)

    sample_imgs = np.concatenate(imgs, axis=1)
    save_sample_imgs = os.path.join(working_dir.name, "raw.png")

    with Image.fromarray(sample_imgs) as img:
        img.save(save_sample_imgs)

    base_vid = os.path.basename(video_params).split(".")[0]
    smplx_params_dir = os.path.join(
        "./motion_video/", base_vid, "smplx_params"
    )

    dump_video_path = os.path.join(working_dir.name, "output.mp4")
    dump_image_path = os.path.join(working_dir.name, "output.png")

    omit_prefix = os.path.dirname(save_sample_imgs)
    image_name = os.path.basename(save_sample_imgs)
    uid = image_name.split(".")[0]
    subdir_path = os.path.dirname(save_sample_imgs).replace(omit_prefix, "")
    subdir_path = subdir_path[1:] if subdir_path.startswith("/") else subdir_path

    motion_seqs_dir = smplx_params_dir
    dump_image_dir = os.path.dirname(dump_image_path)
    os.makedirs(dump_image_dir, exist_ok=True)

    return imgs, save_sample_imgs, motion_seqs_dir, dump_image_dir, dump_video_path


def get_motion_information(
    motion_path: str,
    cfg: DictConfig,
    motion_size: int = 120,
) -> Tuple[str, Dict[str, Any]]:
    """Extract motion information from SMPL-X parameters.

    Args:
        motion_path: Path to the motion sequence directory.
        cfg: Configuration object.
        motion_size: Number of frames to extract.

    Returns:
        Tuple of (motion_name, motion_seqs).
    """

    motion_name = Path(motion_path).parent.name
    smplx_path = Path(motion_path).parent / "smplx_params"
    mask_path = Path(motion_path).parent / "samurai_seg"
    bbox_path = Path(motion_path).parent / "bbox"
    bbox_json_path = os.path.join(bbox_path, "bbox.json")
    if os.path.isfile(bbox_json_path):
        with open(bbox_json_path, "r") as f:
            bbox_dict = json.load(f)
    else:
        bbox_dict = None


    motion_seqs = sorted(glob.glob(os.path.join(smplx_path, "*.json")))
    motion_id_seqs = [
        motion_seq.split("/")[-1].replace(".json", "") for motion_seq in motion_seqs
    ]
    mask_paths = [
        os.path.join(mask_path, motion_id_seq + ".png")
        for motion_id_seq in motion_id_seqs
    ]

    # according mask_paths name to get bbox_dict
    bbox_list = []
    for mask_path in mask_paths:
        mask_name = os.path.basename(mask_path).split(".")[0]
        bbox = bbox_dict.get(mask_name)

        if bbox is not None:
            bbox_list.append(Bbox(bbox, mode="xywh").to_whwh())
        else:
            bbox_list.append(None)

    motion_seqs = prepare_motion_seqs_eval(
        obtain_motion_sequence(str(smplx_path)),
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
        res_scale=1.0,
    )

    return motion_name, motion_seqs