import json, os, torch
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import pycolmap, pyrender, trimesh
os.environ["PYOPENGL_PLATFORM"] = "egl"

from colmap_data_reader import read_colmap_data


def rasterize_pointcloud_to_depth(point_cloud, c2w, K, H, W):
    N = point_cloud.shape[0]
    point_cloud = point_cloud[..., :3]
    point_cloud = np.concatenate([point_cloud, np.ones((N, 1))], axis=1) # (N, 4)
    w2c = np.linalg.inv(c2w)
    point_cloud_cam = (w2c @ point_cloud.T).T[:, :3] # (N, 3)
    # convert to pixel coordinates
    pixel_coords = (K @ point_cloud_cam.T).T # (N, 3)
    pixel_coords = pixel_coords / pixel_coords[:, 2:] # (N, 3)
    pixel_coords = pixel_coords[:, :2] # (N, 2)
    # rasterize
    pixel_coords[:, 0] *= W # because intrinsic matrix "K" is normalized
    pixel_coords[:, 1] *= H 
    depth = np.ones((H, W)) * np.inf
    point_cloud_cam = point_cloud_cam[point_cloud_cam[:, 2] > 0] # remove points behind the camera
    for i in range(point_cloud_cam.shape[0]):
        x_unnorm, y_unnorm = pixel_coords[i].astype(int)
        if x_unnorm >= 0 and x_unnorm < W and y_unnorm >= 0 and y_unnorm < H:
            # we need the closest point to the camera
            if point_cloud_cam[i, 2] < depth[y_unnorm, x_unnorm]:
                depth[y_unnorm, x_unnorm] = point_cloud_cam[i, 2]
    
    return depth

def rasterize_mesh_with_pyrender(mesh, cameras, images, idx):
    scene = pyrender.Scene()
    scene.add(mesh)

    cam_data = cameras[images[idx].camera_id]
    renderer = pyrender.OffscreenRenderer(cam_data.width, cam_data.height)
    fx = np.float32(cam_data.params[0])
    fy = np.float32(cam_data.params[1])
    cx = np.float32(cam_data.params[2])
    cy = np.float32(cam_data.params[3])
    R = np.asmatrix(pycolmap.qvec_to_rotmat(images[idx].qvec)).transpose()
    T = np.identity(4)
    T[0:3,0:3] = R
    T[0:3,3] = -R.dot(images[idx].tvec)
    T[:, 1:3] *= -1 # colmap to computer graphics convention
    pyrender_camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
                                            zfar=800.0)
    cam_node = scene.add(pyrender_camera, pose=T)

    _, depth = renderer.render(scene)
    renderer.delete()

    # postprocess
    depth[depth<=0] = np.inf # set invalid depth to inf
    return depth

def align_depths(pc_depth, depth):
    mask = np.logical_and(pc_depth != np.inf, depth != np.inf)
    
    # use mean and std of depth to ignore outliers mu +- 3*sigma
    mu, sigma = np.mean(depth), np.std(depth)
    mask = np.logical_and(mask, np.abs(depth - mu) < 3*sigma)

    depth = depth[mask]
    pc_depth = pc_depth[mask]
    A = np.vstack([depth, np.ones(depth.shape[0])]).T # (N, 2)
    # scale, shift = np.linalg.lstsq(A, pc_depth, rcond=None)[0]
    ### use cvxpy with L1 regularization
    scale = cp.Variable()
    shift = cp.Variable()
    objective = cp.Minimize(cp.sum(cp.abs(A @ cp.vstack([scale, shift]) - pc_depth[..., np.newaxis])))
    problem = cp.Problem(objective)
    problem.solve()
    scale, shift = scale.value, shift.value

    return scale, shift

def disparity2depth(disp, eps=1e-6):
    ignore_mask = disp < eps
    depth = np.zeros_like(disp)
    depth[~ignore_mask] = 1.0 / disp[~ignore_mask]
    depth[ignore_mask] = np.inf
    return depth

def visualize_depths(depth_map, pc_depth, aligned_depth):
    import matplotlib
    matplotlib.use('Agg')
    FONTSIZE = 30
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    pc_depth[pc_depth == np.inf] = 1e+6
    aligned_depth[aligned_depth == np.inf] = 1e+6
    min2 = min(np.min(pc_depth), np.min(aligned_depth))
    max2 = max(np.max(pc_depth), np.max(aligned_depth))
    ax[0].imshow(1/(depth_map+1e-6))
    ax[0].set_title("Pretrained Depth", fontsize=FONTSIZE)
    ax[1].imshow(1/(pc_depth+1e-6), vmax=1/min2, vmin=1/max2)
    ax[1].set_title("Mesh Depth", fontsize=FONTSIZE)
    ax[2].imshow(1/(aligned_depth+1e-6), vmax=1/min2, vmin=1/max2)
    ax[2].set_title("Aligned Depth", fontsize=FONTSIZE)
    error = np.abs(1/(pc_depth+1e-6) - 1/(aligned_depth+1e-6))
    ax[3].imshow(error)
    ax[3].set_title("Error", fontsize=FONTSIZE)
    for a in ax: a.axis("off")
    # plt.show()
    plt.tight_layout()
    plt.savefig("depth_alignment.png")
    

if __name__=="__main__":
    FRAME = 'frame_0000006263' # change to try out different frames
    vid = 'P01_104' # 'P01_01'
    
    ############ PATHs ############
    colmap_path = f"/work/yashsb/datasets/EPIC-Fields/EPIC-FIELDS-Dense/{vid}"
    pretrained_depth_path = f"/work/yashsb/datasets/EPIC-Fields/EPIC-KITCHENS/P01/{vid}/metric_depth/{FRAME}_depth.npy"
    mesh_path = f"/work/vadim/share/yash/reconstructions/sparse/{vid}/dense/poisson-output.ply"

    ### Load the COLMAP data
    # json_path = os.path.join("/work/yashsb/datasets/EPIC-Fields/EPIC_FIELDS_JSON_DATA", vid + ".json")
    # extrinsics, intrinsics, image_names, point_cloud = read_colmap_data(json_path)
    # idx = image_names.index(f"{FRAME}.jpg")
    # K, c2w = intrinsics[idx], extrinsics[idx]

    ### Load colmap data using Pycolmap
    r = pycolmap.Reconstruction(colmap_path)
    cameras, images = r.cameras, r.images
    for idx, im in images.items():
        if im.name == f"{FRAME}.jpg":
            break
    
    ### Pretrained depthmap (DepthAnything)
    depth_map = np.load(pretrained_depth_path)
    # depth_map = disparity2depth(depth_map) # NOTE: this is actually disparity! Convert to depth first

    ### Rasterize the point cloud to depth
    # pc_depth = rasterize_pointcloud_to_depth(point_cloud, c2w, K, depth_map.shape[0], depth_map.shape[1])

    ### Rendered depthmap from Gaussian Splatting model -- because PC is too sparse
    # gsplat_depth = np.load(f"/work/yashsb/datasets/EPIC-Fields/EPIC-KITCHENS/P01/{vid}/with_visormask/test/ours_30000/depth/{FRAME}.npy")[0]
    
    ### Rasterize mesh using ppyrender
    mesh = pyrender.Mesh.from_trimesh(trimesh.load(mesh_path))
    mesh_depth = rasterize_mesh_with_pyrender(mesh, cameras, images, idx)

    # Alignment of depthmaps
    depth_3D = mesh_depth # OR could be -- gsplat_depth, pc_depth
    scale, shift = align_depths(depth_3D, depth_map)
    print("[Point Cloud] Optimized scale and shift:", scale, shift)
    aligned_depth = scale * depth_map + shift
    
    # OPTIONAL: visualize
    visualize_depths(depth_map, depth_3D, aligned_depth)
