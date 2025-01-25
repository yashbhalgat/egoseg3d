"""
Before running this script, JSON data from VISOR is converted into segmentation maps
using `scripts/visor_JSON_to_segmaps.py`

This script extracts features from the segmentation maps and saves them as numpy arrays
Supported features:
- DINOv2
- CLIP
"""

import argparse, os, torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from feature_model import FEATURE_EXTRACTORS


def extract_features(seg_dir, images_dir, output_dir, feature_extractor):
    for filename in tqdm(os.listdir(seg_dir)):
        if filename.endswith(".npy"):
            image = np.array(Image.open(os.path.join(images_dir, filename.replace(".npy", ".jpg"))))
            seg_map = np.load(os.path.join(seg_dir, filename))
            # resize to image size
            seg_map = np.array(Image.fromarray(seg_map).resize((image.shape[1], image.shape[0]), resample=Image.NEAREST))
            uniq_labels = np.unique(seg_map) # Get unique labels in the segmentation map
            uniq_labels = uniq_labels[uniq_labels != -1]
            features = {}
            for label in uniq_labels:
                mask = seg_map==label
                if mask.sum() > 1:
                    features[label] = feature_extractor.extract_features(image, mask).cpu().numpy()

            np.save(os.path.join(output_dir, filename.split(".")[0]+"_feat.npy"), features)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visor_dir", type=str, required=True, help="Directory containing segmentation maps")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted features")
    parser.add_argument("--feature_type", type=str, required=True, choices=["dinov2", "clip"], help="Feature to extract")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = FEATURE_EXTRACTORS[args.feature_type](device=device)

    extract_features(args.visor_dir, args.images_dir, args.output_dir, feature_extractor)
