import os, json, argparse
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm


def postprocess_deva_gt(deva_pred, segs, original_gtsegs, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    ins2cat_id = {}
    for anno in deva_pred["annotations"]:
        for seginfo in anno["segments_info"]:
            # ins_ids[seginfo["category_id"]].append(seginfo["id"])
            ins2cat_id[seginfo["id"]] = seginfo["category_id"]
    
    out_segs = {}
    for fname, seg in segs.items():
        uniq_ids = np.unique(seg)
        uniq_ids = uniq_ids[uniq_ids != 0]
        out_seg = np.zeros_like(seg)
        for ins_id in uniq_ids:
            out_seg[seg == ins_id] = ins2cat_id[ins_id]
        out_segs[fname] = out_seg

    # Original GT segs are good, where they exist. Take union of original GT segs and deva segs
    for fname, orig_seg in original_gtsegs.items():
        orig_seg = np.array(Image.fromarray(orig_seg).resize((out_segs[fname].shape[1], out_segs[fname].shape[0]), Image.NEAREST))
        out_segs[fname] = np.maximum(out_segs[fname], orig_seg)

    for fname, seg in tqdm(out_segs.items(), desc="Saving segs"):
        np.save(os.path.join(save_dir, fname.replace('.jpg', '.npy')), seg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid", type=str, required=True, help="Video name")
    args = parser.parse_args()
    vid = args.vid
    pid = vid.split("_")[0]

    save_dir = f"/work/yashsb/OSNOM-Lang/scripts/out/{pid}/{vid}/visor_DEVA100_segmaps/"
    if os.path.exists(save_dir):
        print(f"Wait... Check {save_dir} before proceeding")
        exit() # exit if the directory already exists

    pred_dir = f"/datasets/EPIC-KITCHENS/{vid}/visor_DEVA100_segmaps"
    with open(os.path.join(pred_dir, "pred.json"), "r") as f:
        deva_pred = json.load(f)

    segs = {}
    for fname in tqdm(os.listdir(os.path.join(pred_dir, "Annotations", "Raw")), desc="Loading segs"):
        if not fname.endswith('.npy'):
            continue
        segs[fname] = np.load(os.path.join(pred_dir, "Annotations", "Raw", fname))

    orig_gtdir = f"/work/yashsb/OSNOM-Lang/scripts/out/{pid}/{vid}/visor_segmaps/"
    original_gtsegs = {}
    for fname in tqdm(os.listdir(orig_gtdir), desc="Loading original GT segs"):
        if not fname.endswith('.npy'):
            continue
        if int(fname.split('_')[1].split('.')[0]) > 20000:
            continue
        original_gtsegs[fname] = np.load(os.path.join(orig_gtdir, fname))

    
    print(f"Saving to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    postprocess_deva_gt(deva_pred, segs, original_gtsegs, save_dir)