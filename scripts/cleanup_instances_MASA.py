import argparse, json, os
import numpy as np
from tqdm import tqdm
from copy import deepcopy


def hash_function(c, i, max_i):
    # map (c, i) to a unique integer
    return c * max_i + i 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cleanup instances')
    parser.add_argument('--vid', type=str, help='VID of the instance to cleanup')
    args = parser.parse_args()

    vid, pid = args.vid, args.vid.split('_')[0]
    masa_dir = os.path.join(f"./out/{pid}/{vid}/segmaps/masa_s5_backup")
    save_dir = os.path.join(f"./out/{pid}/{vid}/segmaps/masa_s5")

    dino_dir = os.path.join(f"./out/{pid}/{vid}/features/masa_dino_s5_backup")
    dino_save_dir = os.path.join(f"./out/{pid}/{vid}/features/masa_dino_s5")

    with open(f"{masa_dir}/pred.json", 'r') as f:
        pred = json.load(f)

    max_ins = 0
    for anno in pred["annotations"]:
        for seg_info in anno["segments_info"]:
            max_ins = max(max_ins, seg_info["id"])

    H, W = np.load(f"{masa_dir}/Annotations/Raw/{pred['annotations'][0]['file_name']}".replace('.jpg', '.npy')).shape

    new_pred = deepcopy(pred)
    for i, anno in tqdm(enumerate(pred["annotations"]), total=len(pred["annotations"])):
        new_seg = np.zeros((H, W), dtype=np.int32)
        new_dino_feat = {}
        for j, seg_info in enumerate(pred["annotations"][i]["segments_info"]):
            new_i = hash_function(seg_info["category_id"], seg_info["id"], max_ins)
            # modify pred json
            new_pred["annotations"][i]["segments_info"][j]["id"] = new_i
            # modify raw segmap
            segmap = np.load(f"{masa_dir}/Annotations/Raw/{anno['file_name']}".replace('.jpg', '.npy'))
            new_seg[segmap == seg_info["id"]] = new_i
            # modify dino features
            dino_feat = np.load(f"{dino_dir}/{anno['file_name']}".replace('.jpg', '_feat.npy'), allow_pickle=True).item()
            if seg_info["id"] in dino_feat:
                new_dino_feat[new_i] = dino_feat[seg_info["id"]]

        np.save(f"{save_dir}/Annotations/Raw/{anno['file_name']}".replace('.jpg', '.npy'), new_seg)
        np.save(f"{dino_save_dir}/{anno['file_name']}".replace('.jpg', '_feat.npy'), new_dino_feat)

    with open(f"{save_dir}/pred.json", 'w') as f:
        json.dump(new_pred, f)

    print(f"Cleanup instances for {vid} done!")        

