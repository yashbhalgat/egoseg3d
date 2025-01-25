import argparse, json, os, torch
from collections import defaultdict
import numpy as np
from PIL import Image

from eval_deva import init_visor_class_name2id


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./out/visor_new')
    parser.add_argument('--vid', type=str, default='P01_104')
    parser.add_argument('--save_root', type=str, default='./out/')
    args = parser.parse_args()
    vid, pid = args.vid, args.vid.split('_')[0]

    vid_classes = set(torch.load(f"./out/visor/{vid}_classes.pt")["class_names_20k_filtered"])
    print(vid_classes)
    vid_dirs2classes = {}
    for vid_dir in os.listdir(f"{args.input_dir}/results"):
        if vid_dir.startswith(vid):
            vid_dirs2classes[vid_dir] = set(vid_dir[len(vid)+1:].split('_'))
    vid_dirs_classes = []
    for vid_dir, classes in vid_dirs2classes.items():
        classes_ = vid_classes.intersection(classes)
        if len(classes_) > 0:
            vid_dirs_classes.append((vid_dir, classes_.pop()))

    fnames = defaultdict(list)
    for vid_dir, _ in vid_dirs_classes:
        fnames[vid_dir] = [f for f in os.listdir(f"{args.input_dir}/results/{vid_dir}") \
                            if f.endswith('.png') and not f.startswith('#')]
    # take intersection of all fnames
    common_fnames = set(fnames[vid_dirs_classes[0][0]])
    for vid_dir, _ in vid_dirs_classes:
        common_fnames = common_fnames.intersection(fnames[vid_dir])

    visor_class_name2id = init_visor_class_name2id(vid)
    segmaps = {}
    for fname in common_fnames:
        for vid_dir, c in vid_dirs_classes:
            mask = np.array(Image.open(f"{args.input_dir}/results/{vid_dir}/{fname}"))
            if fname not in segmaps:
                segmaps[fname] = np.zeros_like(mask)
            segmaps[fname][mask==1] = visor_class_name2id[c]

    # mapping
    with open(f"{args.input_dir}/frame_mapping.json", 'r') as f:
        frame_mapping = json.load(f)[vid]
    segmaps_mapped = {}
    for fname in segmaps:
        segmaps_mapped[frame_mapping[fname.replace('.png', '.jpg')]] = segmaps[fname]

    save_dir = f"{args.save_root}/{pid}/{vid}/visor_new_segmaps"
    os.makedirs(save_dir, exist_ok=True)
    for fname in segmaps_mapped:
        # Image.fromarray(segmaps_mapped[fname]).save(f"{save_dir}/{fname}")
        np.save(f"{save_dir}/{fname.split('.')[0]}.npy", segmaps_mapped[fname])
    print(f"Saved {len(segmaps_mapped)} segmaps to {save_dir}")
