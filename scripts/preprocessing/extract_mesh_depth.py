import argparse
import os

import align_depth
import cv2
import numpy as np
import pycolmap
import pyrender

# import skimage.color
import trimesh
from tqdm import tqdm

from pathlib import Path

# from tracking import *
from utils import sample_linearly, split_into_chunks


def filter_finite(x):
    return x[np.isfinite(x)]


def normalise_depth(x):
    vmin = filter_finite(x).min()
    vmax = filter_finite(x).max()
    return (x - vmin) / (vmax - vmin) * 255.0


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
    pid = vid.split('_')[0]
    print(f"Processing {vid} ...")
    root = Path(args.root)
    path_mesh = root / Path(f'mesh/{vid}/dense/mesh.ply')
    dir_colmap = root / Path(f"dense/{vid}/sparse/0")
    dir_mesh_depth = root / f"{pid}/{vid}/mesh_depth"

    mesh = trimesh.load(path_mesh)
    mesh = pyrender.Mesh.from_trimesh(mesh)

    r = pycolmap.Reconstruction(dir_colmap)
    cameras = r.cameras
    images = r.images
    frame2colmapid = {x[1].name.split(".")[0]: x[0] for x in list(images.items())}
    frames = sorted(frame2colmapid.keys())[:20000]

    frames_chunk = list(split_into_chunks(frames, args.n_chunks))[args.chunk_id]

    os.makedirs(dir_mesh_depth, exist_ok=True)
    print("Saving files ...")
    for frame in tqdm(frames_chunk):
        mesh_depth = align_depth.rasterize_mesh_with_pyrender(
                mesh, cameras=cameras, images=images, idx=frame2colmapid[frame]
            )
        # NOTE problem with normalisation of depth for visualisation, ignore for now
        # mesh_depth_n = normalise_depth(mesh_depth)
        np.save(os.path.join(dir_mesh_depth, frame + "_depth.npy"), mesh_depth)


if __name__ == "__main__":
    args = parse_args()
    run(args)
