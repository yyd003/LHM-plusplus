# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-02-20 15:07:39
# @Function      : StyleGAN2 Human model warping
import math
import os
import queue
import sys

sys.path.append("./")
import threading

import torch
from torch.nn import functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image

import legacy


def avaliable_device():
    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"

    return device


class EasyStyleGAN_series_model:
    """A simple forward model"""

    def __init__(
        self,
        model_path=os.path.join(
            ROOT_DIR,
            "pretrained_models/stylegan2-human/stylegan_human_v2_1024.pkl",
        ),
    ):

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                "StyleGAN2-Human checkpoint is optional and was not found at "
                f"{model_path!r}. Provide model_path explicitly to use this "
                "legacy discriminator; LHM++ inference does not require it."
            )
        with open(model_path, "rb") as checkpoint_file:
            self.stylegan_human_D = (
                legacy.load_network_pkl(checkpoint_file)["D"]
                .to(avaliable_device())
                .float()
            )
        for para in self.stylegan_human_D.parameters():
            para.requires_grad = False
        self.stylegan_human_D.eval()

    def __call__(self, img, c=None):
        """img: [B C H W]"""

        out = self.stylegan_human_D(img.float(), c)
        return out

    def __repr__(self):
        return f"ESRStyleGAN model:\n {self.stylegan_human_D}"

    def warmup(self):
        """this is engineering trick to compile the nvidia-op :)"""
        import cv2
        import numpy as np

        img = cv2.imread("fake_img.png")
        img = img.astype(np.uint8)
        img = img[..., ::-1]

        img = (
            torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0) / 255.0 - 0.5
        ) * 2
        img = img.to(avaliable_device())
        print("warmup, score: {:.4f}".format(self(img).item()))


# fix the conflict between stylegan-2 and accelerator-library
def warmup_call():
    stylegan_model = EasyStyleGAN_series_model()
    stylegan_model.warmup()
    del stylegan_model
    torch.cuda.empty_cache()
    print("clean cuda.....")


if __name__ == "__main__":
    warmup_call()
    import cv2
    import numpy as np

    model = EasyStyleGAN_series_model()

    img = cv2.imread("fake_img.png")
    img = img.astype(np.uint8)
    img = img[..., ::-1]

    img = (torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0) / 255.0 - 0.5) * 2

    fake_img = cv2.imread("fake.png")
    fake_img = fake_img.astype(np.uint8)
    fake_img = fake_img[..., ::-1]

    fake_img = (
        torch.from_numpy(fake_img.copy()).permute(2, 0, 1).unsqueeze(0) / 255.0 - 0.5
    ) * 2
    merge_img = torch.cat([img, fake_img], dim=0)

    logit = model(merge_img.cuda(), None)
    print(logit)

    img_dir = "./thirdparty/sg2_for_LHM/SHHQ-1.0-imgs/"
    parsing_dir = "./thirdparty/sg2_for_LHM/SHHQ-1.0-parsing_imgs/"

    img_list = sorted(os.listdir(img_dir))
    parsing_list = sorted(os.listdir(parsing_dir))
    assert len(img_list) == len(parsing_list)

    cat_img_list = []
    mb = max(min(8 * min(4096 // 1024, 32), 64), 8)

    for _, (img, parse_img) in enumerate(zip(img_list, parsing_list)):

        img = cv2.imread(os.path.join(img_dir, img))
        img = img.astype(np.uint8)
        mask = cv2.imread(os.path.join(parsing_dir, parse_img), cv2.IMREAD_GRAYSCALE)
        mask = mask[..., None]
        mask = mask == 0
        img = img * (1 - mask) + mask * 255
        img = img[..., ::-1]

        img = (
            torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0) / 255.0 - 0.5
        ) * 2
        cat_img_list.append(img)
        if _ == 32:
            break
    cat_img = torch.cat(cat_img_list, dim=0)
    logit_1 = model(cat_img[:-1].cuda(), None)
    print(logit, logit.mean())
    logit_2 = model(cat_img[1:].cuda(), None)

    print("logit delta:", logit_1[1:] - logit_2[:-1])
    # logit = model(cat_img.cuda(), None)
    # # single_img = cat_img[:1]
    # # repeat_single_img = single_img.repeat(32, 1, 1, 1)
    # # logit = model(repeat_single_img.cuda(), None)
    # print(logit)
    # if logit > -6.5:
    #     img = (
    #         ((img[0] + 1) * 255.0 / 2)
    #         .detach()
    #         .cpu()
    #         .permute(1, 2, 0)
    #         .numpy()
    #         .astype(np.uint8)
    #     )
    #     Image.fromarray(img).save(os.path.join("./debug/gt_-5", f"demo_{_}.jpg"))
    #     print(logit)
