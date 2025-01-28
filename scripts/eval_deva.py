"""
Postprocess DEVA outputs and evaluate HOTA metric(s).
"""

import argparse, json, os, torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from hota import HOTA
from tracking import Scene
from identity import Identity
from clear import CLEAR

from global_vars import SEG_VALID_COUNTS, IGNORE_CAT


def init_visor_class_name2id(video_id, root):
    visor_class_name2id = {}
    all_annotations = torch.load(str(root / Path(f"visor/{video_id}_classes.pt")))["visor"]
    for annotations in all_annotations:
        for ann in annotations["annotations"]:
            visor_class_name2id[ann["name"]] = ann["class_id"]
    return visor_class_name2id


def get_segments_for_class(segments, preds_dict, class_id):
    """
    segments: dict of np.arrays of shape (H, W)
    Output (segments_cls): dict of np.arrays of shape (H, W)
    """
    segments_cls = {}
    for anno in preds_dict["annotations"]:
        fname = anno["file_name"].split(".")[0]
        if fname not in segments:
            continue

        ins_id_list = []  # instance IDs for given class
        for seg in anno["segments_info"]:
            if seg["category_id"] == class_id:
                ins_id_list.append(seg["id"])
        # keep instances if the id is in ins_id_list, otherwise set to 0
        segments_cls[fname] = np.where(
            np.isin(segments[fname], ins_id_list), segments[fname], 0
        )
    return segments_cls

def read_deva_and_gt_ins_seg(deva_output_dir, dir_gt):
    deva_fnames = sorted([
            f for f in os.listdir(os.path.join(deva_output_dir, "Annotations", "Raw"))
            if f.endswith(".npy")
    ])
    gt_fnames = sorted([f for f in os.listdir(dir_gt) if f.endswith(".npy")])
    common_fnames = sorted(list(set(deva_fnames).intersection(set(gt_fnames))))

    deva_ins_seg = {}
    for f in tqdm(common_fnames, desc="Loading DEVA segmaps"):
        deva_ins_seg[f.split(".")[0]] = np.load(os.path.join(deva_output_dir, "Annotations", "Raw", f))

    gt_ins_seg = {}
    for f in tqdm(common_fnames, desc="Loading GT segmaps"):
        gt_ins_seg[f.split(".")[0]] = np.load(os.path.join(dir_gt, f))

    return deva_ins_seg, gt_ins_seg

def postprocess_deva(video_id, class_name, deva_preds, deva_ins_seg, gt_ins_seg, root):
    """
    Output:
        data (dict): contains the following fields
            [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
            [gt_ids, tracker_ids]: list (for each timestep) of 1D NDArrays (for each det).
            [gt_dets, tracker_dets]: list (for each timestep) of lists of detection masks.
            [similarity_scores]: list (for each timestep) of 2D NDArrays.
    """
    deva_ins_seg = get_segments_for_class(
        deva_ins_seg, deva_preds, class_id=deva_preds["category_name2id"][class_name]
    )

    visor_class_name2id = init_visor_class_name2id(video_id, root)
    # NOTE: assuming only one instance per class in GT
    gt_ins_seg = {
        k: (v == visor_class_name2id[class_name]).astype(np.int32)
        for k, v in gt_ins_seg.items()
    }

    # HACK: only keep frames where VISOR says there's an object
    gt_ins_seg = {k: v for k, v in gt_ins_seg.items() if v.sum() > 0}
    deva_ins_seg = {k: v for k, v in deva_ins_seg.items() if k in gt_ins_seg}
    print(f"Using {len(deva_ins_seg)} frames for evaluation.")

    # resize gt_ins_seg to match deva_ins_seg
    frame = list(deva_ins_seg.keys())[0]
    deva_shape = deva_ins_seg[frame].shape
    for k in gt_ins_seg.keys():
        gt_ins_seg[k] = np.array(
            Image.fromarray(gt_ins_seg[k]).resize(
                deva_shape[::-1], Image.NEAREST
            )
        )
    print("Number of frames: ", len(deva_ins_seg))

    data = {}
    data["num_timesteps"] = len(deva_ins_seg)
    uniq_gt_ids = np.unique(np.concatenate(list(gt_ins_seg.values())))
    data["num_gt_ids"] = len(uniq_gt_ids[uniq_gt_ids != 0])  # exclude background
    uniq_tracker_ids = np.unique(np.concatenate(list(deva_ins_seg.values())))
    data["num_tracker_ids"] = len(uniq_tracker_ids[uniq_tracker_ids != 0])  # exclude background
    data["gt_ids"] = []
    data["tracker_ids"] = []
    data["similarity_scores"] = []
    data["gt_dets"] = []
    data["tracker_dets"] = []

    for frame in deva_ins_seg.keys():
        unique_labels = np.unique(deva_ins_seg[frame])
        for label in unique_labels:
            mask = deva_ins_seg[frame] == label
            if mask.sum() <= SEG_VALID_COUNTS:
                deva_ins_seg[frame][mask] = 0

    num_gt_dets, num_tracker_dets = 0, 0
    for t, fname in enumerate(tqdm(deva_ins_seg.keys())):
        gt_id_values = np.unique(gt_ins_seg[fname])
        data["gt_ids"].append(gt_id_values[gt_id_values != 0])
        tr_id_values = np.unique(deva_ins_seg[fname])
        data["tracker_ids"].append(tr_id_values[tr_id_values != 0])
        num_gt_dets += len(data["gt_ids"][-1])
        num_tracker_dets += len(data["tracker_ids"][-1])

        gt_masks, tracker_masks = [], []
        for gt_id in data["gt_ids"][-1]:
            gt_masks.append(gt_ins_seg[fname] == gt_id)
        for tracker_id in data["tracker_ids"][-1]:
            tracker_masks.append(deva_ins_seg[fname] == tracker_id)
        data["gt_dets"].append(gt_masks)
        data["tracker_dets"].append(tracker_masks)

        # compute similarity scores (intersection over union)
        similarity_scores = np.zeros(
            (len(data["gt_ids"][-1]), len(data["tracker_ids"][-1]))
        )
        for i, gt_det in enumerate(data["gt_dets"][-1]):
            for j, tracker_det in enumerate(data["tracker_dets"][-1]):
                intersection = np.sum(np.logical_and(gt_det, tracker_det))
                union = np.sum(np.logical_or(gt_det, tracker_det))
                similarity_scores[i, j] = intersection / union
        data["similarity_scores"].append(similarity_scores)

    unique_gt_ids = np.unique(np.concatenate(list(gt_ins_seg.values())))
    unique_gt_ids = unique_gt_ids[unique_gt_ids != 0]
    unique_tracker_ids = np.unique(np.concatenate(list(deva_ins_seg.values())))
    unique_tracker_ids = unique_tracker_ids[unique_tracker_ids != 0]
    gt_id_map = {k: v for k, v in zip(unique_gt_ids, range(0, len(unique_gt_ids)))} # [1] --> [0]
    tracker_id_map = {
        k: v for k, v in zip(unique_tracker_ids, range(0, len(unique_tracker_ids)))
    }
    for t in range(data["num_timesteps"]):
        for i in range(len(data["gt_ids"][t])):
            data["gt_ids"][t][i] = gt_id_map[data["gt_ids"][t][i]]
        for i in range(len(data["tracker_ids"][t])):
            data["tracker_ids"][t][i] = tracker_id_map[data["tracker_ids"][t][i]]

    data["num_gt_dets"] = num_gt_dets
    data["num_tracker_dets"] = num_tracker_dets

    return data


def count_ID_metric(data):
    """
    There is only one segmentation to consider in GT.
    For that segment, find the best matching predicted segment (for each timestep).
    Count the number of IDs in the selected predicted track.
    """
    ids = np.ones(data["num_timesteps"]) * -1
    for t in range(data["num_timesteps"]):
        if len(data["gt_ids"][t]) > 0 and len(data["tracker_ids"][t]) > 0:
            ious = data["similarity_scores"][t][
                0
            ]  # only one GT segment, so take the first row
            ids[t] = data["tracker_ids"][t][np.argmax(ious)]
    # id_switches = np.sum(ids[:-1] != ids[1:])
    return np.unique(ids[ids != -1])  # , id_switches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="/work/vadim/workspace/experiments/OSNOM-Lang/out"
    )
    parser.add_argument("--exp", type=str, default="")
    # if we use non-standard segmentation outputs
    parser.add_argument("--dir_segments", type=str)
    parser.add_argument(
        "--gt_type",
        choices=["visor_segmaps", "visor_DEVA100_segmaps"],
        default="visor_segmaps",
        type=str,
    )
    parser.add_argument(
        "--segment_type", required=False, type=str
    )
    parser.add_argument("--vid", required=True, type=str)
    parser.add_argument(
        "--class_name", type=str, help="Class to evaluate.", default=None
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vid = args.vid
    pid = vid.split("_")[0]
    root = Path(args.root)
    dir_gt = root / Path(f"{pid}/{vid}/{args.gt_type}")
    print(f"Running eval for {vid}.")

    if args.dir_segments is not None:
        dir_segments = args.dir_segments
    else:
        dir_segments = root / Path(f"{pid}/{vid}/segmaps/{args.segment_type}")

    ### NOTE: for efficiency, no need to run if you can read the hota.json file without errors
    hota_json_path = os.path.join(dir_segments, "hota-"+args.exp+".json" if args.exp else "hota.json")
    # try:
    #     with open(hota_json_path, "r") as f:
    #         hota_res = json.load(f)
    #     print("HOTA results already computed.")
    #     exit()
    # except:
    #     print("continue with evaluation")
    #     pass

    with open(os.path.join(dir_segments, "pred.json"), "r") as f:
        deva_preds = json.load(f)
    deva_ins_seg, gt_ins_seg = read_deva_and_gt_ins_seg(dir_segments, dir_gt)
    print(f"Loaded DEVA and GT segmentation maps for {vid}.")

    if args.class_name is not None:
        data = postprocess_deva(args.vid, args.class_name, deva_preds, deva_ins_seg, gt_ins_seg, root)
        hota = HOTA()
        res = hota.eval_sequence(data)
        print(res)
        print("HOTA Mean: ", np.mean(res["HOTA"]))
        # # Count ID metric
        # id_metric = count_ID_metric(data)
        # print("ID metric: ", id_metric)
    else:
        class_names = deva_preds["category_name2id"].keys()
        class_names = sorted(set(class_names).difference(IGNORE_CAT))
        all_res, all_res_id, all_res_mot = {}, {}, {}
        for class_name in class_names:
            data = postprocess_deva(args.vid, class_name, deva_preds, deva_ins_seg, gt_ins_seg, root)
            hota = HOTA()
            identity_metric = Identity()
            mot = CLEAR()
            res = hota.eval_sequence(data)
            all_res[class_name] = res
            all_res_id[class_name] = identity_metric.eval_sequence(data)
            all_res_mot[class_name] = mot.eval_sequence(data)
            print(f"Class: {class_name}")
            print(
                "HOTA: ", np.mean(res["HOTA"]),
                "HOTA(0): ", res["HOTA(0)"],
                "LocA(0): ", res["LocA(0)"],
            )
            # Count ID metric
            # id_metric = count_ID_metric(data)
            # print("ID metric: ", id_metric)
            print("=====================================")
        # res = hota.combine_classes_class_averaged(all_res, ignore_empty_classes=True)
        res = hota.combine_classes_det_averaged(all_res)
        res_id = identity_metric.combine_classes_det_averaged(all_res_id)
        res_mot = mot.combine_classes_det_averaged(all_res_mot)
        for class_name in all_res:
            res[class_name] = {
                "HOTA": np.mean(all_res[class_name]["HOTA"]),
                "HOTA(0)": all_res[class_name]["HOTA(0)"],
                "LocA(0)": all_res[class_name]["LocA(0)"],
            }
        res["ID metrics"] = {k: int(v) if isinstance(v, np.int64) else v for k, v in res_id.items()}
        res["CLEAR MOT"] = {k: int(v) if isinstance(v, np.int64) else v for k, v in res_mot.items()}
        res["HOTA_mean"] = np.mean(res["HOTA"])
        print("Class-averaged results:")
        print(res)

        # save res as json
        for k, v in res.items():
            if isinstance(v, np.ndarray):
                res[k] = v.tolist()
        with open(hota_json_path, "w") as f:
            json.dump(res, f, indent=4)

        print(
            "HOTA: ", np.mean(res["HOTA"]),
            "HOTA(0): ", res["HOTA(0)"],
            "DetA: ", np.mean(res["DetA"]),
            "AssA: ", np.mean(res["AssA"]),
        )
