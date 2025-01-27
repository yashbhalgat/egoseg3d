import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pycolmap
import scenevis
from PIL import Image
from scipy.optimize import linear_sum_assignment
import skimage
import utils
from global_vars import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = '/work/vadim/workspace/experiments/OSNOM-Lang/out'

def dist_squared(x, y):
    return ((x - y) ** 2).sum()


def similarity_appearance(v_prev, v_curr, beta_v):
    """p.7 for function, p.12 for beta_v."""
    # return 1 / (beta_v * dist_squared(v_prev, v_curr)) # INCORRECT
    return 1 / (1 + beta_v * dist_squared(v_prev, v_curr))
    # return 1 # HACK: deactivated for now


def similarity_loc(loc_prev, loc_curr, beta_l):
    """p.7 for function, p.12 for beta_l."""
    return 1 / beta_l * np.exp(-dist_squared(loc_prev, loc_curr))
    # return 1 # HACK: deactivated for now


def calc_cost(sigma_l, sigma_v):
    return -np.log(sigma_l) - np.log(sigma_v)


def calc_cost_new(sigma_l, sigma_v, sigma_c, sigma_s):
    # NOTE: very small random noise added to handle "ties" in hungarian algorithm
    return -np.log(sigma_l) - np.log(sigma_v) + sigma_c + sigma_s + np.random.normal(0, 0.1)


def apply_hungarian_algorithm(
    tracks_l, obs_l, tracks_a, obs_a, alpha, beta_v, beta_l, beta_c, beta_s, observations=None, tracks=None
):
    """
    p. 8: Matching with Hungarian, filter matches with alpha (p. 8, 10).
    Chosen alpha=50 for visualising pointcloud, for eval they use 10.
    """
    nb_tracks = len(tracks_l)
    nb_observations = len(obs_l)
    cost_matrix = np.zeros((nb_tracks, nb_observations))

    # filling cost matrix
    for i in range(nb_tracks):
        for j in range(nb_observations):
            loc_sim = similarity_loc(tracks_l[i], obs_l[j], beta_l)
            vis_sim = similarity_appearance(
                tracks_a[i] * 0.999, obs_a[j],
                beta_v
            )

            unique, counts = np.unique(tracks[i].categories, return_counts=True)
            track_category = unique[np.argmax(counts)]
            obs_category = observations[j].category
            unequal_category = obs_category != track_category
            sigma_c = unequal_category * beta_c

            # track_instance = tracks[i].labels[-1]
            window_labels = tracks[i].labels[-100:]
            unique, counts = np.unique(window_labels, return_counts=True)
            track_instance = unique[np.argmax(counts)]
            obs_instance = observations[j].label
            unequal_instance = obs_instance != track_instance
            sigma_s = unequal_instance * beta_s

            # cost_matrix[i, j] = calc_cost(loc_sim, vis_sim)
            cost_matrix[i, j] = calc_cost_new(loc_sim, vis_sim, sigma_c, sigma_s)

            # import ext
            # ll = ext.get('_mylist')
            # ll.append([loc_sim, vis_sim, sigma_c, sigma_s])

    # hungarian algorithm
    matched_rows, matched_cols = linear_sum_assignment(cost_matrix)

    # filter out matches with costs exceeding the threshold
    filtered_matches = [
        (r, c) for r, c in zip(matched_rows, matched_cols) if cost_matrix[r, c] <= alpha
    ]

    # return matched tracks (rows) to observations (cols)
    return (
        filtered_matches,
        {match: cost_matrix[match[0], match[1]] for match in filtered_matches},
    )


class Tracker:

    def __init__(self, alpha, beta_v, beta_l, beta_c, beta_s):
        self._tracks = []
        self.alpha = alpha
        self.beta_v = beta_v
        self.beta_l = beta_l
        self.beta_c = beta_c
        self.beta_s = beta_s

    def add_observations(self, observations):

        # initialise (when tracks are empty)
        if len(self._tracks) == 0:
            self._tracks += [Track(obs) for obs in observations]
            return

        last_locations = [x.last_loc3d for x in self._tracks]
        last_appearances = [x.last_appearance for x in self._tracks]

        curr_locations = [x.loc3d for x in observations]
        curr_appearances = [x.appearance for x in observations]

        matches, costs = apply_hungarian_algorithm(
            last_locations,
            curr_locations,
            last_appearances,
            curr_appearances,
            alpha=self.alpha,
            beta_v=self.beta_v,
            beta_l=self.beta_l,
            beta_c=self.beta_c,
            beta_s=self.beta_s,
            observations=observations,
            tracks=self._tracks,
        )

        for track_idx, obs_idx in matches:
            self._tracks[track_idx].add_observation(observations[obs_idx])

        all_obs = set(range(len(observations)))
        matched_obs = set([x[1] for x in matches])
        unmatched_obs = all_obs.difference(matched_obs)

        self._tracks += [Track(observations[i]) for i in unmatched_obs]

    def __getitem__(self, i):
        return self._tracks[i]

    def __len__(self):
        return len(self._tracks)

    def calc_instances(self, deva_loader):

        instances3d = {}
        inst2cat = {}
        locations3d = defaultdict(dict)

        frame = list(deva_loader.frame2id)[0]
        instance_map_template = deva_loader.load_instances(frame)

        for tid in range(0, len(self)):
            # the tracks start from index 0, but the instance labels from 1 (0 for BG)
            instance_id = tid + 1
            print(f"Track: {tid}.")

            for obs in utils.tqdm(list(self[tid].observations.values())):

                if obs.frame not in instances3d:
                    instances3d[obs.frame] = np.zeros_like(instance_map_template)

                locations3d[obs.frame][instance_id] = obs.loc3d

                instances2d = deva_loader.load_instances(obs.frame)
                label = obs.label
                instance = instances2d == label

                instances3d[obs.frame][instance] = instance_id
                inst2cat[instance_id] = obs.category
        return instances3d, inst2cat, locations3d

    def export(self, dst, instances_3d, inst2cat, locations3d, catname2id, valid_frames):

        print('Exporting tracks ... ')
        pred_json_dict = make_pred_json(instances_3d, inst2cat, locations3d, catname2id, valid_frames)
        dir_annotations = os.path.join(dst, "Annotations", "Raw")
        path_json = os.path.join(dst, "pred.json")
        os.makedirs(dir_annotations, exist_ok=True)

        frame = list(instances_3d)[0]
        empty_np = np.zeros_like(instances_3d[frame])

        for frame in utils.tqdm(valid_frames):
            if frame in instances_3d:
                np.save(os.path.join(dir_annotations, frame + ".npy"), instances_3d[frame])
            else:
                np.save(os.path.join(dir_annotations, frame + ".npy"), empty_np)

        utils.save_as_json(path_json, pred_json_dict)
        print(f"Exported `Annotations/Raw` and `pred.json` to {dst}.")


class Track:
    """
    p. 6: Each track represents the set of observations belonging to
    the same object.

    We use "frames" instead of "time" for indexing

    NOTE: we use for now "offline" manner for ease of implementation. They use online, p. 6: "We process the egocentric video E in an online manner. While an offline approach could also be pursued, we opt to replicate the human's spatial cognition - i.e. a person only knows of an object's location when first encountered and this is when it is kept in mind."
    """

    def __init__(self, obs, gamma=100):

        # evolving visual appearance, p. 7
        self.gamma = gamma
        self.observations = {obs.frame: obs}
        self.appearances = {obs.frame: obs.appearance[0]}
        self.last_loc3d = obs.loc3d
        self.last_appearance = obs.appearance[0]
        self.labels = [obs.label]
        self.categories = [obs.category]
        self.frames = [obs.frame]

    def loc2d(self, frame):
        return self.observations[frame].loc2d

    def loc3d(self, frame):
        """p. 7 we refer to the location of T^j at time t by L(T_t^j)."""
        return self.observations[frame].loc3d

    def appearance(self, frame):
        return self.appearances[frame]

    def add_observation(self, obs):
        """
        "track update" eq. 3
        p. 7: Additionally, the track has an evolving appearance representation over time.
        It is calculated at time t using the visual appearance of the most
        recent \gamma visual features assigned to the track.
        """
        self.observations[obs.frame] = obs
        self.frames.append(obs.frame)
        last_appearances = list(self.appearances.values())[-self.gamma :]
        evolved_appearance = np.concatenate(
            [np.array(last_appearances), obs.appearance], axis=0
        ).mean(axis=0)

        self.appearances[obs.frame] = evolved_appearance

        self.categories += [obs.category]
        self.labels += [obs.label]

        self.last_loc3d = obs.loc3d
        self.last_appearance = self.appearances[obs.frame]


class Scene:
    def __init__(self, vid, root=ROOT):
        self.vid = vid
        self.pid = vid.split('_')[0]
        self.root = Path(root)
        self.dir_images = self.root / Path(f'mesh/{self.vid}/images')
        # path_colmap = self.root / Path(f"dense/{vid}/sparse/0")
        path_colmap = self.root / Path(f"mesh/{self.vid}/dense/sparse")
        print('Loading COLMAP model ...')
        self.colmap = pycolmap.Reconstruction(path_colmap)
        self.cameras = self.colmap.cameras


        assert len(self.cameras) == 1

        self.images = self.colmap.images
        self.frame2colmapid = {
            x[1].name.split(".")[0]: x[0] for x in list(self.images.items())
        }
        frames = sorted(self.frame2colmapid)

        self.frames = frames

        self.calibmat = self.cameras[1].calibration_matrix()
        self.calibmat_inv = np.linalg.inv(self.calibmat)

    def rmat(self, frame):
        x = self.images[self.frame2colmapid[frame]]
        return x.rotation_matrix()

    def tvec(self, frame):
        x = self.images[self.frame2colmapid[frame]]
        return x.tvec

    def campose(self, frame):
        x = self.images[self.frame2colmapid[frame]]
        return x.inverse_projection_matrix()

    def load_pointcloud(self, nb_samples=2000):
        indices = sample_linearly(list(self.colmap.points3D), nb_samples)
        points = [self.colmap.points3D[i] for i in indices]
        xyz = np.array([p.xyz for p in points])
        rgb = np.array([p.color.astype(float) / 255 for p in points])
        return xyz, rgb


def lift_observation(centroid, depth, R, t, K):
    u, v = centroid
    Z = depth[v, u]
    pose = scenevis.compute_camera_pose_from_world(R, t)
    K_inv = np.linalg.inv(K)
    camera_coords = np.dot(K_inv, np.array([u, v, 1])) * Z
    world_coords = pose[:3, :3] @ camera_coords[:3] + pose[:3, 3]
    return world_coords


class Observations:
    def __init__(self, scene=None, deva_loader=None, dir_fts=None, scaling_factor=1):
        self.scene = scene
        self._observations = defaultdict(list)
        self.labels = set()
        self.categories = set()
        self.with_pretrained = True
        self.deva_loader = deva_loader
        self.vid = scene.vid
        self.pid = scene.pid
        self.root = scene.root
        self.dir_fts = dir_fts
        if self.dir_fts is None:
            # self.dir_fts = self.root / Path(f"{self.pid}/{self.vid}/features/dino_s5")
            self.dir_fts = self.root / Path(f"{self.vid}/features/dino_s5")
        # self.dir_visor = self.root / Path(f"{self.pid}/{self.vid}/visor_DEVA100_segmaps")
        self.dir_visor = self.root / Path(f"{self.vid}/visor_DEVA100_segmaps")
        # self.dir_depth = self.root / Path(f"{self.pid}/{self.vid}/aligned_depth/default")
        self.dir_depth = self.root / Path(f"{self.vid}/aligned_depth/default")
        self.scaling_factor = scaling_factor

    def add(self, observation):
        self._observations[observation.frame] += [observation]

    def add_by_frame(self, frame, visualise=False, label_selected=None):
        """Add observations for given frame. Visualise for debuggin with
        selected label."""

        if frame in self._observations:
            return

        im = load_im(frame, self.scene.dir_images)
        aligned_depth = load_depth(frame, dir_depth=self.dir_depth)

        # NOTE from previous version with VISOR for calculating centroids
        if not self.with_pretrained:

            visor = load_visor(frame)

            # exclude background label
            visor_labels = set(np.unique(visor)[1:])

        else:
            visor = self.deva_loader.load_instances(frame)
            categories = self.deva_loader.load_categories(frame)
            visor_labels = set(np.unique(visor)[1:])

        fts = load_fts(frame, self.dir_fts)

        inst2cat = self.deva_loader.inst2cat
        for label in visor_labels:

            category = inst2cat[label]

            obs = Observation(frame)
            obs.init_loc2d(visor, label)
            obs.init_loc3d(
                aligned_depth,
                self.scene.rmat(frame),
                self.scene.tvec(frame),
                self.scene.calibmat,
                scaling_factor=self.scaling_factor,
            )
            obs.init_appearance(fts[label])
            obs.init_label(label)
            obs.init_category(category)
            self.add(obs)
            self.labels.add(label)
            self.categories.add(category)

        if label_selected in np.unique(visor) and visualise:

            f, ax = plt.subplots(1, 4, figsize=(1 * 10, 4 * 10))
            ax[0].imshow(im)
            ax[1].imshow(skimage.color.label2rgb(visor, bg_label=-1))
            visor_selected = visor == label_selected
            centroid = calculate_centroid(visor_selected)
            visor_centroid = draw_centroid(visor_selected, centroid)
            ax[2].imshow(visor_centroid)
            ax[2].set_title("object+centroid")
            ax[3].imshow(aligned_depth)
            ax[3].set_title("d_aligned")
            [x.axis("off") for x in ax]
            fig = utils.fig2im(f, is_notebook=False)

            return fig

    def __getitem__(self, frame):
        return self._observations[frame]

    @property
    def frames(self):
        return sorted(self._observations.keys())

    def query(self, labels=None, frames=None):
        """Query observations for given labels."""
        queried = []
        if frames is None:
            frames = self.frames
        # if no labels given, then select all labels currently available
        if labels is None:
            labels = self.labels
        for f in frames:
            for obs in self._observations[f]:
                if obs.label in labels:
                    queried += [obs]
        return queried


class Observation:
    """Represents w_n from p.6, includes frame f_n, location l_n, visual features v_n"""

    def __init__(self, frame):
        # initialise m_n
        # self.mask = None
        self.loc2d = None
        self.loc3d = None
        self.appearance = None
        self.frame = frame

        # include label for evaluation (not used for LMK)
        self.label = None
        self.category = None

    def init_loc2d(self, mask, label):
        mask = mask == label
        centroid = calculate_centroid(mask)
        self.loc2d = centroid

    def init_loc3d(self, depth, rmat, tvec, calibmat, scaling_factor):
        loc3d = lift_observation(self.loc2d, depth, rmat, tvec, calibmat)
        self.loc3d = loc3d * scaling_factor

    def init_appearance(self, fts):
        self.appearance = fts

    def init_label(self, label):
        self.label = label

    def init_category(self, category):
        self.category = category


def load_im(frame, dir_images):
    return plt.imread(dir_images / (frame + ".jpg"))


def load_visor(frame, dir_visor):
    x = np.load(dir_visor / (frame + ".npy"), allow_pickle=True).astype(np.int32)
    x = np.array(Image.fromarray(x).resize((IMAGE_HW[1], IMAGE_HW[0]), Image.NEAREST))

    # exclude hands
    x[x == 300] = -1
    x[x == 301] = -1

    return x


def load_fts(frame, dir_fts):

    fts = np.load(dir_fts / (frame + "_feat.npy"), allow_pickle=True).item()
    return fts


def load_depth(frame, dir_depth=None):
    return np.load(Path(dir_depth) / (frame + "_depth.npy"), allow_pickle=True)


def load_mesh_depth(frame, dir=None):
    if dir is None:
        dir = DIR_MESH_DEPTH
    return np.load(Path(dir) / (frame + "_depth.npy"), allow_pickle=True)


def calculate_centroid(segmentation):
    # get the coordinates of all pixels that are part of the object
    indices = np.where(segmentation == 1)
    y_coords, x_coords = indices[0], indices[1]

    # calculate the mean of the coordinates
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)

    return (int(centroid_x), int(centroid_y))


def draw_centroid(im, centroid, size=7):
    # im must have range [0,1]
    x = im.copy().astype(float)
    if len(im.shape) == 2:
        x = x[:, :, None].repeat(3, 2)
    x[
        centroid[1] - size : centroid[1] + size, centroid[0] - size : centroid[0] + size
    ] = (1, 0, 0)
    return x


class DevaLoader:
    def __init__(self, root):
        self.root = root
        pred_json = utils.read_json(f"{self.root}/pred.json")

        # mapping from Deva
        self.catname2id = pred_json["category_name2id"]
        self.id2catname = {v: k for k, v in self.catname2id.items()}

        frame2id = {}
        inst2cat = {}
        cat2insts = defaultdict(set)
        for i, x in enumerate(pred_json["annotations"]):
            frame2id[x["file_name"].split(".")[0]] = i
            for segment in x["segments_info"]:
                inst2cat[segment["id"]] = segment["category_id"]
                cat2insts[segment["category_id"]].update([segment["id"]])

        self.inst2cat = inst2cat
        self.cat2insts = cat2insts
        self.instance_labels = list(inst2cat)
        self.category_labels = list(cat2insts)
        self.frame2id = frame2id

    def load_deva(self, frame):
        return skimage.io.imread(f"{self.root}/Annotations/{frame}.png", as_gray=True)

    def load_instances(self, frame, visualise=False):
        instances = np.load(f"{self.root}/Annotations/Raw/{frame}.npy")

        uniq_labels, counts = np.unique(instances, return_counts=True)
        counts = counts[uniq_labels != 0]
        uniq_labels = uniq_labels[uniq_labels != 0]

        valid_instances = counts > SEG_VALID_COUNTS
        uniq_labels = uniq_labels[valid_instances]
        instances_filtered = instances.copy()
        instances_filtered[~np.isin(instances, uniq_labels)] = 0
        instances = instances_filtered

        ignore_inst = []
        for cat in IGNORE_CAT:
            if cat not in self.catname2id:
                continue
            catid = self.catname2id[cat]
            for inst in self.cat2insts[catid]:
                ignore_inst += [inst]

        ignore_inst = np.isin(instances, ignore_inst)

        instances[ignore_inst] = 0

        if visualise:
            for i, l in enumerate(self.instance_labels):
                instances[0, i] = l
        return instances

    def load_categories(self, frame, instances=None, visualise=False):
        if instances is None:
            instances = self.load_instances(frame)
        categories = np.zeros_like(instances)

        for k in np.unique(instances)[1:]:
            mask = instances == k
            categories[mask] = self.inst2cat[k]

        if visualise:
            for i, l in enumerate(self.category_labels):
                categories[0, i] = l

        return categories




# convert tracks to "pred.json" format for eval_deva


def make_segments_dict(category_id, instance_id, location):
    return {
        "category_id": category_id,
        "id": instance_id,
        "loc3d": location,
    }


def make_pred_json(instances_3d, inst2cat, locations3d, category_name2id, valid_frames):
    data = {"annotations": []}

    # for frame in sorted(instances_3d):
    for frame in sorted(valid_frames):
        if frame in instances_3d:
            unique_labels = np.unique(instances_3d[frame])[1:].tolist()
            frame_dict = {
                "file_name": frame + ".jpg",
                "segments_info": [
                    # category, instance, location
                    make_segments_dict(inst2cat[l], l, tuple(locations3d[frame][l]))
                    for l in unique_labels
                ],
            }
        else:
            frame_dict = {
                "file_name": frame + ".jpg",
                "segments_info": [],
            }
        data["annotations"].append(frame_dict)

    data["category_name2id"] = category_name2id

    return data


def remove_segments(segmap, labels, background_label):
    # -1 for VISOR
    ignore = np.isin(segmap, labels)
    segmap = segmap.copy()
    segmap[ignore] = background_label
    return segmap
