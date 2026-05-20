# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : SMPL-X pose estimation model

import os
import pdb
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from engine.ouputs import BaseOutput
from engine.pose_estimation.model import Model

IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]


@dataclass
class SMPLXOutput(BaseOutput):
    beta: np.ndarray
    is_full_body: bool


def normalize_rgb_tensor(img, imgenet_normalization=True):
    img = img / 255.0
    if imgenet_normalization:
        img = (
            img - torch.tensor(IMG_NORM_MEAN, device=img.device).view(1, 3, 1, 1)
        ) / torch.tensor(IMG_NORM_STD, device=img.device).view(1, 3, 1, 1)
    return img


def load_model(ckpt_path, model_path, device=torch.device("cuda")):
    """Open a checkpoint, build Multi-HMR using saved arguments, load the model weigths."""
    # Model

    assert os.path.isfile(ckpt_path), f"{ckpt_path} not found"

    # Load weights
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    # Get arguments saved in the checkpoint to rebuild the model
    kwargs = {}
    for k, v in vars(ckpt["args"]).items():
        kwargs[k] = v
    print(ckpt["args"].img_size)
    # Build the model.
    if isinstance(ckpt["args"].img_size, list):
        kwargs["img_size"] = ckpt["args"].img_size[0]
    else:
        kwargs["img_size"] = ckpt["args"].img_size
    kwargs["smplx_dir"] = model_path
    print("Loading model...")
    model = Model(**kwargs).to(device)
    print("Model loaded")
    # Load weights into model.
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.output_mesh = False
    model.eval()
    return model


def inverse_perspective_projection(points, K, distance):
    """
    This function computes the inverse perspective projection of a set of points given an estimated distance.
    Input:
        points (bs, N, 2): 2D points
        K (bs,3,3): camera intrinsics params
        distance (bs, N, 1): distance in the 3D world
    Similar to:
        - pts_l_norm = cv2.undistortPoints(np.expand_dims(pts_l, axis=1), cameraMatrix=K_l, distCoeffs=None)
    """
    # Apply camera intrinsics
    points = torch.cat([points, torch.ones_like(points[..., :1])], -1)
    points = torch.einsum("bij,bkj->bki", torch.inverse(K), points)

    # Apply perspective distortion
    if distance is None:
        return points
    points = points * distance
    return points


class PoseEstimator:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device)
        self.mhmr_model = load_model(
            os.path.join(model_path, "pose_estimate", "multiHMR_896_L.pt"),
            model_path=model_path,
            device=self.device,
        )
        self.pad_ratio = 0.2
        self.img_size = 896
        self.fov = 60

    def to(self, device):
        self.device = torch.device(device)
        self.mhmr_model.to(self.device)
        return self

    def get_camera_parameters(self):
        K = torch.eye(3)
        # Get focal length.
        focal = self.img_size / (2 * np.tan(np.radians(self.fov) / 2))
        K[0, 0], K[1, 1] = focal, focal

        K[0, -1], K[1, -1] = self.img_size // 2, self.img_size // 2

        # Add batch dimension
        K = K.unsqueeze(0).to(self.device)
        return K

    def img_center_padding(self, img_np):

        ori_w, ori_h = img_np.shape[:2]

        w = round((1 + self.pad_ratio) * ori_w)
        h = round((1 + self.pad_ratio) * ori_h)

        img_pad_np = np.zeros((w, h, 3), dtype=np.uint8)
        offset_h, offset_w = (w - img_np.shape[0]) // 2, (h - img_np.shape[1]) // 2
        img_pad_np[
            offset_h : offset_h + img_np.shape[0] :,
            offset_w : offset_w + img_np.shape[1],
        ] = img_np

        return img_pad_np, offset_w, offset_h

    def _preprocess(self, img_np):

        raw_img_size = max(img_np.shape[:2])

        img_tensor = (
            torch.Tensor(img_np).to(self.device).unsqueeze(0).permute(0, 3, 1, 2)
        )

        _, _, h, w = img_tensor.shape
        scale_factor = min(self.img_size / w, self.img_size / h)
        img_tensor = F.interpolate(
            img_tensor, scale_factor=scale_factor, mode="bilinear"
        )

        _, _, h, w = img_tensor.shape
        pad_left = (self.img_size - w) // 2
        pad_top = (self.img_size - h) // 2
        pad_right = self.img_size - w - pad_left
        pad_bottom = self.img_size - h - pad_top
        img_tensor = F.pad(
            img_tensor,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0,
        )

        resize_img = normalize_rgb_tensor(img_tensor)

        annotation = (
            pad_left,
            pad_top,
            scale_factor,
            self.img_size / scale_factor,
            raw_img_size,
        )

        return resize_img, annotation

    @torch.no_grad()
    def __call__(self, img, is_shape=True):
        # image_tensor H W C
        if isinstance(img, str):
            img_np = np.asarray(Image.open(img).convert("RGB"))
        elif isinstance(img, np.ndarray):
            img_np = img
        else:
            raise ValueError(f"Invalid input type: {type(img)}")

        raw_h, raw_w, _ = img_np.shape
        img_np, offset_w, offset_h = self.img_center_padding(img_np)
        img_tensor, annotation = self._preprocess(img_np)
        K = self.get_camera_parameters()

        with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            target_human = self.mhmr_model(
                img_tensor,
                is_training=False,
                nms_kernel_size=int(3),
                det_thresh=0.3,
                K=K,
                idx=None,
                max_dist=None,
            )

        if not is_shape:
            return target_human

        if len(target_human) < 1:
            return SMPLXOutput(
                beta=None,
                is_full_body=False,
            )

        pad_left, pad_top, scale_factor, crop_size, raw_size = annotation

        target_human["loc"] = (
            target_human["loc"] - torch.tensor([pad_left, pad_top], device=self.device)
        ) / scale_factor

        target_human["dist"] = target_human["dist"] / (crop_size / raw_size)

        transl = inverse_perspective_projection(
            target_human["loc"].view(1, 1, -1), K, target_human["dist"].view(1, 1, -1)
        )[0, 0]

        return SMPLXOutput(
            beta=target_human["shape"][0].cpu().numpy(),
            is_full_body=True,
        )


if __name__ == "__main__":
    img_path = "/home/qing/code/LHM/LHM/example_data/demo_data/imgs_png/00001.png"
    model_path = (
        "/home/qing/code/LHM/data_preprocess/HumanVideoProcessing/tmp/smplx_models"
    )
    pose_estimator = PoseEstimator(model_path)
    img_np = np.asarray(Image.open(img_path).convert("RGB"))
    output = pose_estimator(img_np)
    print(output)
