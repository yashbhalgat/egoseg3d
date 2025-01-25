import io
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import imageio
import os
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw, ImageFont
import torch


class SimpleVideoReader(Dataset):
    def __init__(self, image_dir, subsample_factor=1):
        """
        image_dir - points to a directory of jpg images
        """
        self.image_dir = image_dir
        self.subsample_factor = subsample_factor
        frames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg') or f.endswith('.png')])
        self.frames = frames[::self.subsample_factor]
        self.height, self.width = self.get_height_width()
        self.fps = 30 // self.subsample_factor

    def get_height_width(self):
        im_path = os.path.join(self.image_dir, self.frames[0])
        img = np.array(Image.open(im_path).convert('RGB'))
        return img.shape[0], img.shape[1]

    def __getitem__(self, idx):
        frame = self.frames[idx]

        im_path = os.path.join(self.image_dir, frame)
        img = Image.open(im_path).convert('RGB')
        img = np.array(img)

        return img, im_path

    def __len__(self):
        return len(self.frames)


def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def save_as_json(path, dict):
    with open(path, "w") as fp:
        json.dump(dict, fp)


def split_into_chunks(lst, n):
    """
    # Example usage
    my_list = ["apple", "banana", "cherry", "date", "elderberry",
    "fig", "grape"]
    n = 3
    chunks = list(split_into_chunks(my_list, n))
    print(chunks)
    """
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def sample_linearly(x, nb_samples):
    indices = np.linspace(0, len(x) - 1, nb_samples).round().astype(int)
    return [x[i] for i in indices]


def figsize(im, scale=10):
    return [x / max(list(im.shape)) * scale for x in im.shape[:2]][::-1]


def fig2im(f, show=False, with_alpha=False, is_notebook=False):

    # f: figure from previous plot (generated with plt.figure())
    buf = io.BytesIO()
    buf.seek(0)
    if is_notebook:
        plt.savefig(
            buf, format="png", bbox_inches="tight", transparent=True, pad_inches=0
        )
    else:
        plt.savefig(buf, format="jpg", bbox_inches="tight")
    if not show:
        plt.close(f)
    im = Image.open(buf)
    # return without alpha channel (contains only 255 values)
    return np.array(im)[..., : 3 + with_alpha]


def write_mp4(name, src, fps=10):

    if type(src) == str:

        src = os.path.normpath(src)
        if src[-1] != '*':
            src = src + '/*'
        src = sorted(glob(src))

    if type(src[0]) == str:
        src = [plt.imread(fpath) for fpath in src]

    imageio.mimwrite(name + ".mp4", src, "mp4", fps=fps)


def tqdm(x):
    if 'JPY_PARENT_PID' in os.environ:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    return tqdm(x)


def blend_mask(im, mask, colour=[1, 0, 0], alpha=0.5, show_im=False):
    if len(im.shape) == 2:
        im = np.tile(im[:, :, None], [1, 1, 3])
    """Blend an image with a mask (colourised via `colour` and `alpha`)."""
    if type(im) == torch.Tensor:
        im = im.numpy()
    im = im.copy()
    if im.max() > 1:
        im = im.astype(float) / 255
    for ch, rgb_v in zip([0, 1, 2], colour):
        im[:, :, ch][mask == 1] = im[:, :, ch][mask == 1] * (1 - alpha) + rgb_v * alpha
    if show_im:
        plt.imshow(im)
        plt.axis("off")
        plt.show()
    return im


# Function to read the RGB values from the file and create a colormap
def create_colormap_from_file(num_col=60):
    """
    Example:

        # File path
        file_path = './scripts/cmap_glasbey60.txt'

        # Create the colormap
        cmap = create_colormap_from_file(file_path)

        # To visualize the colormap
        fig, ax = plt.subplots(figsize=(6, 1), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
        # fig.subplots_adjust(top=0.5, bottom=0.5, left=0.2, right=0.8)
        cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=ax, orientation='horizontal')
        plt.show()
    """

    file_path = f"/work/vadim/workspace/experiments/OSNOM-Lang/scripts/colormaps/cmap_glasbey{num_col}.txt"

    with open(file_path, "r") as file:
        lines = file.readlines()

    # Parse the RGB values
    colors = []
    for line in lines:
        r, g, b = map(int, line.strip().split(","))
        colors.append((r / 255, g / 255, b / 255))  # Normalize the values to [0, 1]

    # Create a colormap
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    return cmap


def make_colors(num_col=500):
    cmap = create_colormap_from_file(num_col=num_col)
    colors = np.array(cmap.colors)[1:]
    num_col -= 1
    return colors, num_col


def draw_text(im, text):
    height, width = im.shape[:2]

    # Convert to PIL image for adding titles
    pil_img = Image.fromarray((im * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)

    # You can specify a font file if you have one, otherwise the default will be used
    font = ImageFont.truetype("arial.ttf", 20)
    (width + height) * 0.2

    # Add titles
    draw.text((width * 0.1, height * 0.1), text, fill="white", font=font)
    return np.array(pil_img)