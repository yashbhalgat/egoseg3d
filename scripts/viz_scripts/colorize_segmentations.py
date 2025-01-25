import os, argparse
import numpy as np
from PIL import Image


def read_color_map(colormap_path):
    with open(colormap_path, 'r') as f:
        lines = f.readlines()
    color_map = []
    for line in lines:
        color_map.append([int(x) for x in line.strip().split(",")])
    return np.array(color_map)[1:] # exclude first color, i.e. white. Now, first color is black.

def colorize_visor_seg(input_seg_dir, output_save_dir, bg=-1):
    colormap = read_color_map("colormaps/cmap_glasbey50.txt")
    for fname in os.listdir(input_seg_dir):
        if not fname.endswith('.npy'):
            continue
        seg = np.load(os.path.join(input_seg_dir, fname))
        seg_color = colormap[(seg-bg)%49] # background is -1, so +1 to get the black color
        seg_color = Image.fromarray(seg_color.astype(np.uint8))
        seg_color.save(os.path.join(output_save_dir, fname.replace('.npy', '.png')))
        print(f"Saved {fname.replace('.npy', '.png')} to {output_save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colorize Visor segmentation outputs")
    parser.add_argument("--input_seg_dir", type=str, required=True, help="Directory containing Visor segmentation outputs")
    parser.add_argument("--output_save_dir", type=str, required=True, help="Directory to save colorized segmentation outputs")
    parser.add_argument("--bg_id", type=int, default=-1, help="Background ID in the segmentation outputs")
    args = parser.parse_args()
    if not os.path.exists(args.output_save_dir):
        os.makedirs(args.output_save_dir)
    colorize_visor_seg(args.input_seg_dir, args.output_save_dir, args.bg_id)
