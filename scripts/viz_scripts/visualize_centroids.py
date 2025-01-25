import os
import numpy as np
import pandas as pd
from PIL import Image
import pycolmap, pyrender, trimesh
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as offline

import align_depth, scenevis
from tracking import *

RANGE_POT = (4040, 4136)
LABEL_POT = 29
RANGE_DISH_CLOTH = (5580, 5690)
LABEL_DISH_CLOTH = 17


if __name__=="__main__":
    ### Paths
    mesh_path = 'reconstructions/sparse/P01_104/dense/poisson-output.ply'
    colmap_path = 'reconstructions/dense_frames/P01_104/sparse/0'
    visor_path = 'out/P01/P01_104/visor_segmaps/'
    pointcloud_path = 'reconstructions/sparse/P01_104/dense/sparse'

    trimesh_mesh = trimesh.load(mesh_path)
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)
    r = pycolmap.Reconstruction(colmap_path)
    cameras, images = r.cameras, r.images
    frame2colmapid = {x[1].name.split('.')[0]: x[0] for x in list(images.items())}
    frames = [x[:x.index('.')] for x in sorted(os.listdir(visor_path))]

    ### Object CONSTANTS
    range_selected = RANGE_POT
    label_selected = LABEL_POT

    centroids, aligned_depths = [], []
    # for frame in frames[range_selected[1]-1:range_selected[1]:]:
    for frame in frames[range_selected[0]:range_selected[1]:10]:
        im, visor, pred_depth = load_im(frame), load_visor(frame), load_depth(frame)
        # resize visor to same size as im
        visor = np.array(Image.fromarray(visor).resize((im.shape[1], im.shape[0]), Image.NEAREST))
        visor_mask = visor == label_selected
        centroid = calculate_centroid(visor_mask)
        # visor_centroid = draw_centroid(visor_mask, centroid)

        mesh_depth = align_depth.rasterize_mesh_with_pyrender(mesh, cameras=cameras, images=images, idx=frame2colmapid[frame])
        scale, shift = align_depth.align_depths(mesh_depth, pred_depth)
        print("[Point Cloud] Optimized scale and shift:", scale, shift)
        aligned_depth = scale * pred_depth + shift

        centroids.append(centroid); aligned_depths.append(aligned_depth)

    centroid_traces = []
    for centroid, aligned_depth in zip(centroids, aligned_depths):
        u, v = centroid
        Z = aligned_depth[v, u]
        x = images[frame2colmapid[frame]]
        R, t = x.rotation_matrix(), x.tvec
        pose = scenevis.compute_camera_pose_from_world(R, t)
        K = cameras[1].calibration_matrix()
        K_inv = np.linalg.inv(K)
        camera_coords = np.dot(K_inv, np.array([u, v, 1])) * Z
        world_coords = pose[:3,:3] @ camera_coords[:3] + pose[:3,3]
        object_points = scenevis.points2trace(world_coords[None, :], np.array([0, 0, 1])[None,], size=5)
        centroid_traces += [object_points]

    rec, poses, pcd_xyz, pcd_clr = scenevis.load_colmap(pointcloud_path)
    # trace_pcl = scenevis.points2trace(pcd_xyz, pcd_clr)
    trace_mesh = scenevis.mesh2trace(trimesh_mesh)
    trace_cams = scenevis.poses2traces(poses[::])

    layout = go.Layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'
        )
    )

    fig = go.Figure(data=trace_cams + [trace_mesh] + centroid_traces, layout=layout)
    fig.update_traces(showlegend=False)

    offline.plot(fig, filename='pot_mesh.html', auto_open=False)
