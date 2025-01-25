import os
from visor_JSON_to_segmaps import *
import torch
from tqdm.notebook import tqdm


def image2id(x):
    return int(x["image"]["name"].split("_")[-1].split(".")[0])


def reduce_annotations(annotations):
    annotations_reduced = []
    for ann in annotations:
        ann_reduced = {"image": ann["image"], "annotations": [], "class2id": {}}
        for x in ann["annotations"]:
            x_reduced = {k: v for k, v in x.items() if "segments" not in k}
            ann_reduced["annotations"].append(x_reduced)
            if (x_reduced["class_id"] != 300) and (x_reduced["class_id"] != 301):
                ann_reduced["class2id"][x_reduced["name"]] = x_reduced["class_id"]
        annotations_reduced.append(ann_reduced)

    return annotations_reduced


def main():
    videos = [x for x in os.listdir(f"{VISOR_PATH}/") if ".json" in x]
    video_ids = [
        x.split("_inter")[0] for x in os.listdir(f"{VISOR_PATH}/") if ".json" in x
    ]

    for vid in tqdm(video_ids):

        annotations, _, _ = load_annotations(f'{VISOR_PATH}/{vid}_interpolations.json')
        annotations_reduced = {"visor": reduce_annotations(annotations)}

        class_names = set()
        class_names_20k_frames = set()

        n_frames_20k = 0
        for x in annotations_reduced["visor"]:
            class_names.update(x["class2id"])
            if image2id(x) < 20000:
                class_names_20k_frames.update(x["class2id"])
                n_frames_20k += 1

        class_names_20k_frames_filtered = []
        # parse only classes that are clearly defined (can be used as input for detection model)
        # meaning, exclude class names with `/`
        for x in class_names_20k_frames:
            if "/" in x:
                continue
            else:
                class_names_20k_frames_filtered += [x]

        annotations_reduced["n_frames_20k"] = n_frames_20k
        annotations_reduced["class_names"] = class_names
        annotations_reduced["class_names_20k"] = class_names_20k_frames
        annotations_reduced["class_names_20k_filtered"] = (
            class_names_20k_frames_filtered
        )

        print(f'VISOR frames: {n_frames_20k}')
        print(f'VISOR classes: {class_names_20k_frames_filtered}')
        print(f'VISOR nb classes: {len(class_names_20k_frames_filtered)}')

        torch.save(annotations_reduced, f"{VISOR_PATH}/{vid}_classes.pt")


if __name__ == "__main__":

    # extracts class names and IDs from VISOR interpolations
    # example input: ./out/visor/{vid}_interpolations.json'
    # example output: f"./out/visor/{vid}_classes.pt"
    # with "{vid}" being a video ID
    # reads and extracts both from and to VISOR_PATH
    VISOR_PATH = './out/visor'
    main()

