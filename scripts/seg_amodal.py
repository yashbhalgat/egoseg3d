
import tracking
import utils
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import skimage
from utils import tqdm
import argparse
import os

COLORS, NUM_COL = utils.make_colors()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid", type=str)
    parser.add_argument(
        "--root", type=str, default="/work/vadim/workspace/experiments/OSNOM-Lang/out"
    )
    parser.add_argument(
        "-f", type=str
    )

    return parser.parse_args()

def init(args):
    vid = args.vid
    pid = vid.split('_')[0]
    segtype = 'tracked-shortlist-refined'
    loader = tracking.DevaLoader(args.root + f'/{pid}/{vid}/segmaps/{segtype}/')
    dir_images = Path(f"/work/vadim/workspace/experiments/OSNOM-Lang/out/mesh/{vid}/images")

    return loader, dir_images

def visualise_frame(loader, dir_images, frame, key=None):
    image = tracking.load_im(frame, dir_images)
    instances = loader.load_instances(frame).copy()
    print(f'Instances in image: {np.unique(instances)}')
    if key is None:
        im = skimage.color.label2rgb(instances, colors=COLORS, image=image, alpha=0.7)
    else:
        mask = instances == key
        im = skimage.color.label2rgb(mask, colors=COLORS[1:], image=image, alpha=0.7)

    plt.imshow(im)
    plt.show()


def visualise_instances(loader, dir_images, frames, return_images=False, each_nth=1, selected_instance=None):

    colors, num_col = utils.make_colors()

    ims = []

    for frame in tqdm(frames[::each_nth]):
        image = tracking.load_im(frame, dir_images)
        instances = loader.load_instances(frame).copy()
        instances_img = (colors[instances % num_col] * 0.7 + (image / 255) * 0.3)

        if return_images:
            # use this to select an object in a video
            im = instances_img
            im = utils.draw_text(im, frame)
            ims += [im]
        else:
            # use this to select frames in jupyter notebook
            if selected_instance is not None:
                mask = instances == selected_instance
                im = utils.blend_mask(image, mask, alpha=0.7)
            else:
                im = instances_img

            im = utils.draw_text(im, frame)
            f = plt.imshow(im)
            plt.show()

    return ims

def save_masks(loader, dir_images, frames, selected_instance, dir_out):

    for frame in tqdm(frames):
        instances = loader.load_instances(frame).copy()
        mask = ((instances == selected_instance)).astype(np.uint8) * 255
        skimage.io.imsave(dir_out / (frame + '.png'), mask)


if __name__ == '__main__':

    args = parse_args()
    loader, dir_images = init(args)
    frames = sorted(loader.frame2id)[::]

    selected_instance = None

    # we save images only if we don't select a specific instance
    # a specific instance is used when we try to find frames for the amodal seg
    return_images = True
    if return_images:
        assert selected_instance is None

    ims = visualise_instances(
        loader,
        dir_images,
        frames,
        each_nth=1,
        return_images=return_images,
        selected_instance=selected_instance
    )

    root = Path(args.root)

    dir_amodal = root / 'vis' / 'seg_amodal' / args.vid
    os.makedirs(dir_amodal, exist_ok=True)

    utils.write_mp4(str(dir_amodal / 'summary'), ims)