import collections, json, os
import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm
# from scipy.optimize import minimize
# from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


def read_colmap_data(json_path, reorder=True):
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    all_extrinsics, all_intrinsics, all_image_names = [], [], []
    for idx, (image_name, quatxyz) in tqdm(enumerate(metadata["images"].items())):
        width, height = metadata["camera"]["width"], metadata["camera"]["height"]
        fx, fy, cx, cy = metadata["camera"]["params"][:4]
        intrinsics = np.eye(3, dtype=np.float32)
        intrinsics[0, 0] = fx
        intrinsics[1, 1] = fy
        intrinsics[0, 2] = cx
        intrinsics[1, 2] = cy
        intrinsics[0] /= width
        intrinsics[1] /= height
        all_intrinsics.append(intrinsics)
    
        # Read the camera extrinsics.
        qw, qx, qy, qz = quatxyz[:4]
        w2c = np.eye(4, dtype=np.float32)
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
        w2c[:3, :3] = rotation
        w2c[:3, 3] = quatxyz[4:]
        extrinsics = np.linalg.inv(w2c)
        all_extrinsics.append(extrinsics)

        all_image_names.append(image_name)

    # Since COLMAP shuffles the images, we generally want to re-order them according
    # to their file names so that they form a video again.
    if reorder:
        ordered = sorted([(name, index) for index, name in enumerate(all_image_names)])
        indices = [index for _, index in ordered]
        all_extrinsics = [all_extrinsics[index] for index in indices]
        all_intrinsics = [all_intrinsics[index] for index in indices]
        all_image_names = [all_image_names[index] for index in indices]

    # return torch.stack(all_extrinsics), torch.stack(all_intrinsics), all_image_names, torch.tensor(metadata["points"], dtype=torch.float32, device=device)[..., :3]
    return np.stack(all_extrinsics), np.stack(all_intrinsics), all_image_names, np.array(metadata["points"], dtype=np.float32)


# def depthmap_to_pointcloud(depth_orig, intrinsic, extrinsic, scale, shift):
#     """Convert depth map to point cloud using camera intrinsics and extrinsics."""
#     depth_map = depth_orig * scale + shift
#     height, width = depth_map.shape
#     K, R, t = intrinsic[:3, :3], extrinsic[:3, :3], extrinsic[:3, 3]
#     K_inv = np.linalg.inv(K)

#     # Create a point cloud.
#     point_cloud = []
#     for v in range(height):
#         for u in range(width):
#             u_norm, v_norm = u / width, v / height
#             if depth_map[v, u] > 0:  # Valid depth
#                 # point_3D_cam = project_depth_to_3D(u_norm, v_norm, depth_map[v, u], K_inv)
#                 point_3D_cam = depth_map[v, u] * (K_inv @ np.array([u_norm, v_norm, 1]))
#                 point_3D = R @ point_3D_cam + t
#                 point_cloud.append(point_3D)
    
#     point_cloud = torch.stack(point_cloud)
#     return point_cloud.cpu().numpy()


# def compute_error(params, depth_map, K_inv, c2w, point_cloud_tree):
#     """Compute the total error of transformed depth map against the point cloud."""
#     s, b = params
#     points_3D = depthmap_to_pointcloud(depth_map, K_inv, c2w, s, b)
#     # use point_cloud_tree.query
#     dist, _ = point_cloud_tree.query(points_3D)
#     return np.sum(dist)
    

# def optimize_depth_scale_shift(depth_map, K, c2w, point_cloud):
#     """Optimize scale (s) and shift (b) to align depth map with point cloud."""
#     K_inv = np.linalg.inv(K)
#     point_cloud_tree = cKDTree(point_cloud)  # Using cKDTree for nearest neighbor search

#     result = minimize(# using L-BFGS-B
#         compute_error,  # The error function
#         x0=[1.0, 0.0],  # Initial guess
#         args=(depth_map, K_inv, c2w, point_cloud_tree),
#         method="L-BFGS-B",
#         bounds=[(0.1, 100.0), (-100.0, 100.0)],  # Bounds for scale and shift
#         options={"disp": False},
#     )   

#     if result.success:
#         return result.x  # Returns the optimized [s, b]
#     else:
#         raise Exception("Optimization failed:", result.message)

# # Example usage (assuming you have loaded your depth_map, K, R, t, and point_cloud)
# # optimized_scale_shift = optimize_depth_scale_shift(depth_map, K, R, t, point_cloud)
# # print("Optimized scale and shift:", optimized_scale_shift)
# if __name__ == "__main__":
#     # Load the COLMAP data
#     json_path = os.path.join("/work/yashsb/datasets/EPIC-Fields/EPIC_FIELDS_JSON_DATA", "P01_01" + ".json")
#     extrinsics, intrinsics, image_names, point_cloud = read_colmap_data(json_path)

#     # Load the depth map and point cloud
#     depth_map = np.load("/work/yashsb/datasets/EPIC-Fields/EPIC-KITCHENS/P01/P01_01/depth/frame_0000026089_depth.npy")

#     # Optimize the depth map scale and shift
#     # find index for frame_0000026089
#     idx = image_names.index("frame_0000026089.jpg")
#     optimized_scale_shift = optimize_depth_scale_shift(depth_map, intrinsics[idx], extrinsics[idx], point_cloud)
#     print("Optimized scale and shift:", optimized_scale_shift)
#     scale, shift = optimized_scale_shift

#     # visualize the point cloud and the depth map - use different colors for the point cloud and the depth map
#     depth_points = depthmap_to_pointcloud(depth_map, intrinsics[idx], extrinsics[idx], scale, shift)
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c="b", label="Point Cloud")
#     ax.scatter(depth_points[:, 0], depth_points[:, 1], depth_points[:, 2], c="r", label="Depth Map")
#     plt.show()