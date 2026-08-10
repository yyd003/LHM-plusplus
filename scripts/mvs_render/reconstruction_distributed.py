# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : Distributed 3D reconstruction pipeline

import sys

sys.path.append(".")

import os
import pdb
import time
import traceback

import numpy as np
from tqdm import tqdm

from engine.MVSRecon.mvs_recon import MVSRec


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
        timesleep = kwargs.get("timesleep", 3600)
        time.sleep(timesleep)  # one hour


def run_mvsrecon(items, **params):
    """run_mvsrecon"""
    output_dir = params["dirs"]
    debug = params["debug"]
    os.makedirs(output_dir, exist_ok=True)

    if debug:
        items = items[:20]

    process_valid = []

    for item in tqdm(items, desc="Processing..."):
        if ".ply" not in item:
            continue
        gs_ply = item
        baseid = basename(gs_ply)

        gs_ply = f"./exps/output_gs/{baseid}.ply"
        data_root = f"./exps/mvs_render/{baseid}"
        mask_root = f"./exps/mvs_sam/{baseid}"
        normal_root = f"./exps/mvs_normal/{baseid}"
        mesh_root = f"./exps/smplx_output/{baseid}.obj"

        # os.makedirs('debug/recon_examples', exist_ok=True)

        # out_ply = os.path.join(output_dir, f'{baseid}' +'.obj')
        # save_ply = os.path.join('debug/recon_examples', f'{baseid}' +'.obj')

        # import shutil

        # shutil.copy(out_ply, save_ply)
        # save_gs_ply = f'./debug/exps/output_gs/{baseid}.ply'
        # save_data_root = f'./debug/exps/mvs_render/{baseid}'
        # save_mask_root = f'./debug/exps/mvs_sam/{baseid}'
        # save_normal_root = f'./debug/exps/mvs_normal/{baseid}'
        # save_mesh_root = f'./debug/exps/smplx_output/{baseid}.obj'
        # os.makedirs(os.path.dirname(save_gs_ply), exist_ok=True)
        # os.makedirs(os.path.dirname(save_data_root), exist_ok=True)
        # os.makedirs(os.path.dirname(save_mask_root), exist_ok=True)
        # os.makedirs(os.path.dirname(save_normal_root), exist_ok=True)
        # os.makedirs(os.path.dirname(save_mesh_root), exist_ok=True)
        # shutil.copy(gs_ply, save_gs_ply)
        # shutil.copy(mesh_root, save_mesh_root)
        # shutil.copytree(data_root, save_data_root)
        # shutil.copytree(mask_root, save_mask_root)
        # shutil.copytree(normal_root, save_normal_root)
        # continue

        try:
            stereo = MVSRec(
                gs_ply, data_root, mask_root, normal_root, mesh_root, radius=2.5
            )
            stereo.fitting(
                output_dir,
                save_name=f"{baseid}" + ".obj",
                Niter=200,
                vis_data_root=f"./exps/mvs_recvis/{baseid}",
            )
        except:
            traceback.print_exc()

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
                os.path.join(opt.input, item) for item in available_items
            ]

    multi_process(
        worker=run_mvsrecon,
        items=available_items,
        dirs=opt.output,
        nodes=opt.nodes,
        debug=opt.debug,
    )
