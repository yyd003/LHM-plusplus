# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : Distributed normal estimation pipeline

import sys

sys.path.append(".")

import os
import pdb
import time
import traceback

import numpy as np
from tqdm import tqdm

from engine.NormalEstimator.Sapiens.interface import interface


def basename(path):
    pre_name = os.path.basename(path).split(".")[0]

    return pre_name


def multi_process(worker, items, **kwargs):
    """
    worker worker function to process items
    """

    nodes = kwargs["nodes"]
    dirs = kwargs["dirs"]

    bucket = int(
        np.ceil(len(items) / nodes)
    )  # avoid  last node  process too many items.

    print("Total Nodes:", nodes)
    print("Save Path:", dirs)
    rank = int(os.environ.get("RANK", 0))

    print("Current Rank:", rank)

    kwargs["RANK"] = rank

    if rank == nodes - 1:
        output_dir = worker(items[bucket * rank :], **kwargs)
    else:
        output_dir = worker(items[bucket * rank : bucket * (rank + 1)], **kwargs)

    if rank == 0 and nodes > 1:
        time_sleep = kwargs.get("timesleep", 7200)
        time.sleep(time_sleep)  # one hour


def run_normal(items, **params):
    output_dir = params["dirs"]
    debug = params["debug"]
    os.makedirs(output_dir, exist_ok=True)

    if debug:
        items = items[:1]

    process_valid = []

    for item in tqdm(items, desc="Processing..."):
        if not os.path.isdir(item):
            continue

        print(f"processing img dir {item}")
        basename_folder = basename(item)
        img_folder = item
        mask_folder = item.replace("mvs_render", "mvs_sam")
        normal_folder = item.replace("mvs_render", "mvs_normal")

        if os.path.exists(normal_folder):
            continue

        interface(img_folder, mask_folder, normal_folder)

    return output_dir


def get_parse():
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input", required=True, help="input path")
    parser.add_argument("-o", "--output", required=True, help="output path")
    parser.add_argument("--nodes", default=1, type=int, help="how many workload?")
    parser.add_argument("--debug", action="store_true", help="debug tag")
    parser.add_argument("--txt", default=None, type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = get_parse()

    # catch avaliable items
    if opt.txt == None:
        available_items = os.listdir(opt.input)
        available_items = [os.path.join(opt.input, item) for item in available_items]
    else:
        available_items = []
        with open(opt.txt) as reader:
            for line in reader:
                available_items.append(line.strip())

            available_items = [
                os.path.join(opt.input, item.split(".")[0]) for item in available_items
            ]

    multi_process(
        worker=run_normal,
        items=available_items,
        dirs=opt.output,
        nodes=opt.nodes,
        debug=opt.debug,
    )
