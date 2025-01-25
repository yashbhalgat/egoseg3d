import random

import numpy as np
import plotly.graph_objs as go
import pycolmap


def load_colmap(path_sparse_model, n_points=5000):
    # path: e.g. filtered/P01_01/sparse/0/
    rec = pycolmap.Reconstruction(path_sparse_model)

    # load poses
    poses = []
    for k in list(rec.images.keys())[::]:
        x = rec.images[k]
        R = x.rotation_matrix()
        t = x.tvec
        pose = compute_camera_pose_from_world(R, t)
        poses.append(pose)

    # then pointcloud
    indices = random.sample(list(rec.points3D), n_points)
    points = [rec.points3D[i] for i in indices]
    pcd_xyz = np.array([p.xyz for p in points])
    pcd_clr = np.array([p.color.astype(float) / 255 for p in points])

    return rec, poses, pcd_xyz, pcd_clr


def plot_3d_points(points):
    trace = go.Scatter3d(
        x=[p[0] for p in points],
        y=[p[1] for p in points],
        z=[p[2] for p in points],
        mode="markers",
    )
    layout = go.Layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
    return trace, layout


def points2trace(pcd_xyz, pcd_clr=None, size=2):
    import numpy as np
    import plotly.graph_objs as go

    x, y, z = [x[:, 0] for x in np.split(pcd_xyz, 3, axis=-1)]

    colors = pcd_clr

    # Create a scatter3d trace
    trace = go.Scatter3d(
        x=x, y=y, z=z, mode="markers", marker=dict(size=size, color=colors, opacity=0.7)
    )

    return trace


def mesh2trace(trimesh_mesh):
    import plotly.graph_objs as go

    x, y, z = [x[:, 0] for x in np.split(trimesh_mesh.vertices, 3, axis=-1)]

    i, j, k = [x[:, 0] for x in np.split(trimesh_mesh.faces, 3, axis=-1)]

    trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.5)

    return trace


def points2pointcloud(pcd_xyz, pcd_clr=None, color="red", size=2, opacity=0.7):
    import numpy as np
    import plotly.graph_objs as go

    x, y, z = [x[:, 0] for x in np.split(pcd_xyz, 3, axis=-1)]
    if pcd_clr is not None:
        color = pcd_clr
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(size=size, color=color, opacity=opacity),
    )
    return [trace]


def compute_camera_pose_from_world(R_w, t_w):
    R_c = R_w.T
    t_c = -np.dot(R_c, t_w)
    pose = np.eye(4)
    pose[:3, :3] = R_c
    pose[:3, 3] = t_c
    return pose


def poses2traces(poses, scale=0.1):

    cameras = []

    for pose in poses:

        R = np.array(pose[:3, :3])
        t = np.array(pose[:3, 3])
        X = np.array(
            [
                [1, 1, 0],
                [-1, 1, 0],
                [0, 0, 2],
                [1, 1, 0],
                [1, -1, 0],
                [-1, -1, 0],
                [-1, 1, 0],
                [0, 0, 2],
                [1, -1, 0],
                [0, 0, 2],
                [-1, -1, 0],
            ]
        )
        X = X * scale
        X[:, -1] = X[:, -1] * -1

        X = np.dot(X, R.T) + t
        X = X[:, :3]

        x, y, z = [x[:, 0] for x in np.split(X, 3, axis=-1)]

        camera_traces = []
        camera_traces = go.Scatter3d(
            x=x, y=y, z=z, mode="lines", line=dict(color="red", width=2)
        )
        cameras += [camera_traces]
    return cameras


def make_open3d_bbox(poses, pcd):
    # calculate bbox from poses and pointcloud
    bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(
            [x[:3, -1] for x in poses] + [z for z in pcd_filtered.points]
        )
    )


def make_plotly_bbox(center=(0, 0, 0), extent=(1, 1, 1)):
    # Define the coordinates of the bounding box
    x = [0, 0, 1, 1, 0]
    y = [0, 1, 1, 0, 0]
    z = [0, 0, 0, 0, 0]

    x = x + [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    y = y + [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    z = z + [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]

    bbox = np.stack([x, y, z])

    # shift to (0,0,0) center
    bbox = bbox - 0.5

    # scale extent
    bbox = bbox * np.asarray(extent[:, None])

    # shift to selected center
    bbox = bbox + np.asarray(center)[:, None]

    return bbox


def bbox2trace(bbox):

    x, y, z = [x[0] for x in np.split(bbox, 3, 0)]

    scatter = go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="red", width=2))

    return scatter


import numpy as np
import plotly.graph_objs as go


def poses2cameras(poses, scale=0.1):

    cameras = []

    for pose in poses:

        R = np.array(pose[:3, :3])
        t = np.array(pose[:3, 3])
        X = np.array(
            [
                [1, 1, 0],
                [-1, 1, 0],
                [0, 0, 2],
                [1, 1, 0],
                [1, -1, 0],
                [-1, -1, 0],
                [-1, 1, 0],
                [0, 0, 2],
                [1, -1, 0],
                [0, 0, 2],
                [-1, -1, 0],
            ]
        )
        X = X * scale
        X[:, -1] = X[:, -1] * -1

        X = np.dot(X, R.T) + t
        X = X[:, :3]

        x, y, z = [x[:, 0] for x in np.split(X, 3, axis=-1)]

        lines = [[X[0], X[1]], [X[1], X[2]], [X[2], X[3]], [X[3], X[0]]]
        lines += [[X[0], X[-1]], [X[1], X[-1]], [X[2], X[-1]], [X[3], X[-1]]]
        colors = ["red"] * len(lines)

        camera_traces = []
        camera_traces = go.Scatter3d(
            x=x, y=y, z=z, mode="lines", line=dict(color="red", width=2)
        )
        cameras += [camera_traces]

    return cameras


def mesh2plotly(mesh):
    # requires mesh from `trimesh`

    x, y, z = mesh.vertices.swapaxes(1, 0)
    faces = mesh.faces
    facecolors = mesh.visual.face_colors
    vertexcolors = mesh.visual.vertex_colors

    plotly_mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=[f[0] for f in faces],
        j=[f[1] for f in faces],
        k=[f[2] for f in faces],
        facecolor=facecolors,
        vertexcolor=vertexcolors,
    )

    return plotly_mesh
