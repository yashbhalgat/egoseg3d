"""
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

from global_vars import SEG_VALID_COUNTS


def extract_features(seg_dir, images_dir, output_dir, feature_extractor):
    for filename in tqdm(os.listdir(seg_dir)):
        if filename.endswith(".npy"):
            image = np.array(
                Image.open(os.path.join(images_dir, filename.replace(".npy", ".jpg")))
            )
            seg_map = np.load(os.path.join(seg_dir, filename))
            # resize to image size
            seg_map = np.array(
                Image.fromarray(seg_map).resize(
                    (image.shape[1], image.shape[0]), resample=Image.NEAREST
                )
            )

            # OLD: resulted in errors with Deva labels, because of small segments
            # uniq_labels = np.unique(seg_map) # Get unique labels in the segmentation map
            # uniq_labels = uniq_labels[uniq_labels != 0]
            # features = {}

            # NEW, with filtering of small segments
            # Get unique labels in the segmentation map
            uniq_labels, counts = np.unique(seg_map, return_counts=True)
            counts = counts[uniq_labels != 0]
            uniq_labels = uniq_labels[uniq_labels != 0]
            features = {}

            for label in uniq_labels:
                mask = seg_map == label
                if mask.sum() > SEG_VALID_COUNTS:
                    features[label] = (
                        feature_extractor.extract_features(image, mask).cpu().numpy()
                    )

            np.save(
                os.path.join(output_dir, filename.split(".")[0] + "_feat.npy"), features
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deva_seg_dir",
        type=str,
        required=True,
        help="Directory containing segmentation maps",
    )
    parser.add_argument(
        "--images_dir", type=str, required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save extracted features",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        required=True,
        choices=["dinov2", "clip"],
        help="Feature to extract",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = FEATURE_EXTRACTORS[args.feature_type](device=device)

    extract_features(
        args.deva_seg_dir, args.images_dir, args.output_dir, feature_extractor
    )
