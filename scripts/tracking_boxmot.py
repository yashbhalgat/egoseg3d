import argparse
import cv2, os, json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from boxmot import DeepOCSORT, BoTSORT, BYTETracker, OCSORT, HybridSORT


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_bbox_from_mask(mask):
    '''
    Input: binary mask of shape (H, W)
    Output: bbox as list [x_min, y_min, x_max, y_max]
    '''
    y, x = np.where(mask)
    bbox = [x.min(), y.min(), x.max(), y.max()]
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return bbox, area


def make_pred_json(filenames_all, ids_all, clss_all, scores_all, category_name2id):
    data = {"annotations": []}

    for i in range(len(filenames_all)):
        frame_dict = {
            "file_name": filenames_all[i],
            "segments_info": [
                # category, instance, location
                {
                    "category_id": clss_all[i][j],
                    "id": ids_all[i][j],
                    "score": scores_all[i][j],
                }
                for j in range(len(ids_all[i]))
            ],
        }
        data["annotations"].append(frame_dict)

    data["category_name2id"] = category_name2id

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", type=str, default="DeepOCSORT")
    parser.add_argument("--vid", type=str, default="P01_104")
    parser.add_argument("--save_suffix", type=str, default="s5")
    args = parser.parse_args()

    print(f"Tracking for {args.vid} using {args.tracker}")

    pid, vid = args.vid.split("_")[0], args.vid
    det_dir = f"./out/{pid}/{vid}/segmaps/deva_OWLv2_s5/"
    img_dir = f"./out/ek100/{vid}"
    save_dir = f"./out/{pid}/{vid}/segmaps/{args.tracker}_{args.save_suffix}"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{det_dir}/pred.json", "r") as f:
        preds = json.load(f)

    if args.tracker == "DeepOCSORT":
        tracker = DeepOCSORT(
            model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
            device='cuda:0',
            fp16=True,
        )
    elif args.tracker == "BoTSORT":
        tracker = BoTSORT(
            model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
            device='cuda:0',
            fp16=True,
        )
    elif args.tracker == "BYTETracker":
        tracker = BYTETracker()
    elif args.tracker == "OCSORT":
        tracker = OCSORT()
    elif args.tracker == "HybridSORT":
        tracker = HybridSORT(
            reid_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
            device='cuda:0',
            half=True,
            det_thresh=0.1, # not needed as 2D detection model already thresholds detections
        )
    else:
        raise ValueError(f"Tracker {args.tracker} not implemented")
            

    filenames_all, ids_all, clss_all, scores_all = [], [], [], []
    for anno in tqdm(preds["annotations"]):
        filenames_all.append(anno["file_name"])
        segmap = np.load(f"{det_dir}/Annotations/Raw/{anno['file_name'].split('.')[0]}.npy")
        masks, dets = [], []
        for seg_info in anno["segments_info"]:
            mask = segmap == seg_info["id"]
            bbox, area = get_bbox_from_mask(mask)
            if area > 100:
                masks.append(mask)
                dets.append(bbox + [seg_info["score"], seg_info["category_id"]]) # x1, y1, x2, y2, score, class

        masks= np.array(masks) if len(masks) > 0 else np.zeros((0, segmap.shape[0], segmap.shape[1]))
        dets = np.array(dets) if len(dets) > 0 else np.zeros((0, 6))
        im = cv2.imread(f"{img_dir}/{anno['file_name'].split('.')[0]}.jpg")
        # resize to segmap size
        im = cv2.resize(im, (segmap.shape[1], segmap.shape[0]), interpolation=cv2.INTER_LINEAR)

        # try:
        tracks = tracker.update(dets, im) # --> M x (x, y, x, y, id, conf, cls, ind)
        # except:
        #     breakpoint()

        if len(tracks) > 0:
            xyxys = tracks[:, 0:4].astype('int') # float64 to int
            ids = tracks[:, 4].astype('int') # float64 to int
            confs = tracks[:, 5]
            clss = tracks[:, 6].astype('int') # float64 to int
            inds = tracks[:, 7].astype('int') # float64 to int
            masks = masks[inds]
        else:
            ids = []
            clss = []
            confs = []
            masks = []

        # save segmentation map
        os.makedirs(f"{save_dir}/Annotations/Raw", exist_ok=True)
        new_segmap = np.zeros_like(segmap)
        for id_, mask in zip(ids, masks):
            new_segmap[mask] = id_
        np.save(f"{save_dir}/Annotations/Raw/{anno['file_name'].split('.')[0]}.npy", new_segmap)

        ids_all.append(ids)
        clss_all.append(clss)
        scores_all.append(confs)

    # save pred.json
    new_preds = make_pred_json(filenames_all, ids_all, clss_all, scores_all, preds["category_name2id"])
    with open(f"{save_dir}/pred.json", "w") as f:
        json.dump(new_preds, f, cls=NpEncoder)

    print(f"Saved to {save_dir}")
    
