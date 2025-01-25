import json, os
import numpy as np
import pycolmap, pyrender, trimesh
from tqdm import tqdm
os.environ["PYOPENGL_PLATFORM"] = "egl"

from align_depth import rasterize_mesh_with_pyrender, disparity2depth, align_depths


if __name__=="__main__":
    vid = 'P01_104' # 'P01_01'
    
    ############ PATHs ############
    colmap_path = f"/work/yashsb/datasets/EPIC-Fields/EPIC-FIELDS-Dense/{vid}"
    pretrained_depth_path = f"/work/yashsb/datasets/EPIC-Fields/EPIC-KITCHENS/P01/{vid}/metric_depth/"
    mesh_path = f"/work/vadim/share/yash/reconstructions/sparse/{vid}/dense/poisson-output.ply"

    ### Load colmap data using Pycolmap
    r = pycolmap.Reconstruction(colmap_path)
    cameras, images = r.cameras, r.images
    
    ### Load mesh
    mesh = pyrender.Mesh.from_trimesh(trimesh.load(mesh_path))
    
    scales, shifts, errors = {}, {}, {}
    for i, (idx, im) in tqdm(enumerate(images.items())):
        if i % 10 != 0:
            continue
        # Rasterize mesh
        mesh_depth = rasterize_mesh_with_pyrender(mesh, cameras, images, idx)

        ### Pretrained depthmap (DepthAnything)
        depth_path = os.path.join(pretrained_depth_path, im.name.replace(".jpg", "_depth.npy"))
        if not os.path.exists(depth_path):
            continue
        depth_map = np.load(depth_path)
        # depth_map = disparity2depth(depth_map) # NOTE: this is actually disparity! Convert to depth first

        # Alignment of depthmaps
        scale, shift = align_depths(mesh_depth, depth_map)
        print("[Point Cloud] Optimized scale and shift:", scale, shift)
        aligned_depth = scale * depth_map + shift
        error = np.abs(1/(mesh_depth[mesh_depth!=np.inf]+1e-6) - 1/(aligned_depth[mesh_depth!=np.inf]+1e-6)).mean()
    
        scales[im.name], shifts[im.name], errors[im.name] = scale.item(), shift.item(), error.item()

    with open("depth_alignment_metric.json", "w") as f:
        json.dump({"scales": scales, "shifts": shifts, "errors": errors}, f)