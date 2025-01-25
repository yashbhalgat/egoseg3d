import argparse, os, json, re, cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse
from pathlib import Path

def extract_resolution(input_string):
    pattern = r"(\d+)x(\d+)"
    match = re.search(pattern, input_string)
    if match:
        width, height = map(int, match.groups())  # Convert the extracted values to integers
        return width, height
    else:
        return None, None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid", type=str)
    parser.add_argument(
        "--root", type=str, default="/work/vadim/workspace/experiments/OSNOM-Lang/out"
    )

    return parser.parse_args()

def convert_segarry_to_mask(seg_arrs, H, W):
    # input: seg_arr : [(N1, 2), (N2, 2), ...]. array of pixels on the mask boundary
    # output: mask : (H, W). binary mask
    mask = np.zeros((H, W), dtype=np.uint8)
    for seg_arr in seg_arrs:
        seg_arr = np.array(seg_arr)
        mask = cv2.fillPoly(mask, [seg_arr.astype(np.int32)], color=1)
    return mask

def load_annotations(json_path):
    print('Loading annotations...')
    with open(json_path, 'r') as f:
        data = json.load(f)
        annotations = data["video_annotations"]
        width, height = extract_resolution(data["info"]["details"])
        if width is None or height is None:
            print("Could not extract resolution from the JSON file. Let's debug.")
            breakpoint()
        del data # free up memory
    return annotations, width, height

def visor_JSON_to_segmaps(annotations, width, height, output_dir):

    all_annotations = defaultdict(dict)
    for anno in tqdm(annotations):

        image_name = "_".join(anno["image"]["name"].split("_")[-2:]) # P01_01_frame_0000000937.png --> frame_0000000937.png
        image_name = image_name.split(".")[0]

        for obj_anno in anno["annotations"]:
            obj_name = obj_anno["name"]
            obj_cls_id = obj_anno["class_id"]
            obj_tuple = (obj_name, obj_cls_id)
            if obj_tuple not in all_annotations[image_name]:
                try:
                    all_annotations[image_name][obj_tuple] = obj_anno["segments"]
                except:
                    print(f"Error in {image_name} for {obj_tuple}")
                    breakpoint()

    for image_name, obj_annos in tqdm(all_annotations.items()):
        segmap = -1 * np.ones((height, width), dtype=np.int32)
        for obj_tuple, segments in obj_annos.items():
            _, obj_cls_id = obj_tuple
            mask = convert_segarry_to_mask(segments, height, width)
            segmap[mask == 1] = obj_cls_id

        np.save(os.path.join(output_dir, image_name + ".npy"), segmap)


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root)
    vid = args.vid
    pid = vid.split('_')[0]
    dir_out = root / Path(f'{pid}/{vid}/visor_segmaps')
    os.makedirs(dir_out, exist_ok=True)
    path_json = root / Path(f'visor/{vid}_interpolations.json')

    annotations, width, height = load_annotations(path_json)
    visor_JSON_to_segmaps(annotations, width, height, dir_out)
