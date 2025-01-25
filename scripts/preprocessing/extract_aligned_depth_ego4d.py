import argparse
import os

import align_depth
import cv2
import numpy as np
from tqdm import tqdm
from tracking import *
from utils import sample_linearly, split_into_chunks, save_as_json
from extract_mesh_depth import normalise_depth


def parse_args():
    parser = argparse.ArgumentParser()

    # for parallel processing on cluster
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument("--chunk_id", type=int, default=0)

    parser.add_argument("--vid", type=str)

    parser.add_argument(
        "--root", type=str, default="./out"
    )

    return parser.parse_args()


def run(args):

    vid = args.vid
    root = Path(args.root)

    dir_aligned_depth = root / Path(f"{vid}/aligned_depth")
    dir_depth = root / Path(f"{vid}/depth")
    dir_depth_mesh = root / Path(f"{vid}/mesh_depth")

    print(f"Processing {vid} ...")
    outdir = os.path.join(dir_aligned_depth, "default")
    os.makedirs(outdir, exist_ok=True)

    alignment_factors_default = {}

    frames = [x for x in os.listdir(dir_depth_mesh) if "npy" in x]
    frames = sorted(["_".join(x.split(".")[0].split("_")[:2]) for x in frames])

    frames_chunk = list(split_into_chunks(frames, args.n_chunks))[args.chunk_id]

    # last_frame = sorted(os.listdir(dir_aligned_depth / 'default'))[-6].split('.')[0].split('_depth')[0]

    for i, frame in tqdm(enumerate(frames_chunk)):

        # if i < frames_chunk.index(last_frame):
        #     continue

        depth_default = load_depth(frame, dir_depth=dir_depth)
        depth_mesh = load_mesh_depth(frame, dir=dir_depth_mesh)
        
        # resize to match depth_default. Keep inf values as is
        mask = depth_mesh == np.inf
        depth_mesh[mask] = -1
        depth_mesh = cv2.resize(depth_mesh, (depth_default.shape[1], depth_default.shape[0]))
        depth_mesh[depth_mesh < 0] = np.inf

        scale, shift = align_depth.align_depths(depth_mesh, depth_default)
        alignment_factors_default[frame] = [float(scale), float(shift)]
        default_aligned = scale * depth_default + shift

        np.save(os.path.join(outdir, frame + "_depth.npy"), default_aligned)

        # NOTE ignoring normalisation of depth for now
        # x = normalise_depth(default_aligned)
        # cv2.imwrite(os.path.join(outdir, frame + "_depth.png"), x)

    save_as_json(
        os.path.join(
            dir_aligned_depth,
            f"alignment_default-{args.chunk_id + 1}_{args.n_chunks}.json",
        ),
        alignment_factors_default,
    )


if __name__ == "__main__":
    args = parse_args()
    run(args)
