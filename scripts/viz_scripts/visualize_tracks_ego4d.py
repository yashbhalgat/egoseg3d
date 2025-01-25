import os
import numpy as np
import matplotlib.pyplot as plt
from tracking import *
from tqdm.notebook import tqdm
import skimage.color
import tracking
import utils
from tracking import DevaLoader
from utils import tqdm, create_colormap_from_file

from PIL import Image, ImageDraw, ImageFont

def stack_images(image, visor, instances, instances_3d, instances_masa, categories, colors, num_col):
    height, width, _ = image.shape
    spacing = 20  # space between images
    title_height = 40  # height for titles
    total_width = width * 5 + spacing * 4
    total_height = height + title_height

    # Create a blank canvas
    stacked_images = np.ones((total_height, total_width, 3), dtype=float)

    visor_img = (colors[visor // 3 % num_col] * 0.3 + (image / 255) * 0.7)
    instances_img = (colors[instances % num_col] * 0.7 + (image / 255) * 0.3)
    instances_3d_img = (colors[instances_3d % num_col] * 0.7 + (image / 255) * 0.3)
    instances_masa_img = (colors[instances_masa % num_col] * 0.7 + (image / 255) * 0.3)
    categories_img = (colors[categories % num_col] * 0.7 + (image / 255) * 0.3)

    # Place images on the canvas
    stacked_images[title_height:title_height + height, 0:width] = visor_img
    stacked_images[title_height:title_height + height, width + spacing:2 * width + spacing] = instances_img
    stacked_images[title_height:title_height + height, 2 * width + 2 * spacing:3 * width + 2 * spacing] = instances_3d_img
    stacked_images[title_height:title_height + height, 3 * width + 3 * spacing:4 * width + 3 * spacing] = instances_masa_img
    stacked_images[title_height:title_height + height, 4 * width + 4 * spacing:5 * width + 4 * spacing] = categories_img

    # Convert to PIL image for adding titles
    pil_img = Image.fromarray((stacked_images * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)

    # You can specify a font file if you have one, otherwise the default will be used
    font = ImageFont.truetype("arial.ttf", 30)
    # font = ImageFont.load_default()

    # Add titles
    draw.text((width // 2 - 30, 10), "VISOR", fill="black", font=font)
    draw.text((width + spacing + width // 2 - 60, 10), "2D baseline", fill="black", font=font)
    draw.text((2 * width + 2 * spacing + width // 2 - 30, 10), "Ours", fill="black", font=font)
    draw.text((3 * width + 3 * spacing + width // 2 - 30, 10), "Masa", fill="black", font=font)
    draw.text((4 * width + 4 * spacing + width // 2 - 30, 10), "2D/Ours - Categories", fill="black", font=font)

    return np.array(pil_img)


vid = 'bff3d583-ca3b-44b8-9740-3b34c5a8d7a9'
segtype_2d = 'deva_OWLv2_s5'
segtype_3d = 'bl100'
segtype_masa = 'deva_OWLv2_s5'

loader_2d = DevaLoader(f'out/{vid}/segmaps/{segtype_2d}/')
loader_3d = DevaLoader(f'out/{vid}/segmaps/{segtype_3d}/')
loader_masa = DevaLoader(f'out/{vid}/segmaps/{segtype_masa}/')

scene = Scene(vid)

observations = Observations(scene, loader_3d)

dir_images = scene.dir_images
visor_frames = set([x.split('.')[0] for x in os.listdir(observations.dir_visor)]) if os.path.exists(observations.dir_visor) else set()

# class_name = None
class_name = "kettle"
if class_name:
    # from eval_deva import init_visor_class_name2id
    # visor_class_name2id = init_visor_class_name2id(vid)
    visor_class_name2id = {class_name: 7}

### colormap
num_col = 500
cmap = create_colormap_from_file(num_col=num_col)
colors = np.array(cmap.colors)[1:]
num_col -= 1

frames = sorted(loader_2d.frame2id)
print(f'Number of frames: {len(frames)}')



### Plotting
ims = []
visualise = True
save_fig = True
each_nth = 10

visor_zeros = np.zeros_like(loader_2d.load_instances(frames[0], visualise=False))
visor_labels_max = 0

for frame in tqdm(frames[::]):

    if frame not in visor_frames:
        visor = visor_zeros
    else:
        visor = tracking.load_visor(frame, observations.dir_visor)

        # # add colour consistency to VISOR labels for visualisation
        # visor_labels_max = np.max(visor)
        # for l in range(visor_labels_max):
        #     visor[0, l] = l

    image = tracking.load_im(frame, dir_images)
    instances = loader_2d.load_instances(frame, visualise=False)

    categories = loader_2d.load_categories(frame, instances=instances, visualise=visualise)
    instances_3d = loader_3d.load_instances(frame, visualise=False).copy()

    instances_masa = loader_masa.load_instances(frame, visualise=False)

    if class_name:
        visor = visor * (visor == visor_class_name2id[class_name])
        instances = instances * (categories == loader_2d.catname2id[class_name])
        instances_3d = instances_3d * (categories == loader_3d.catname2id[class_name])
        instances_masa = instances_masa * (categories == loader_masa.catname2id[class_name])

    if visualise:
        # # NOTE visualisation can be sped up a lot by directly stacking images with np.stack instead
        # # of visualising with plt and then transforming to an image
        # f, ax = plt.subplots(1, 4, figsize=(10, 20))
        # # ax[0].imshow(skimage.color.label2rgb(visor, colors=cmap.colors, image=image, alpha=0.3))
        # ax[0].imshow(colors[visor//3 % num_col] * 0.3 + (image/255) * 0.7)
        # ax[0].set_title('VISOR')
        # # ax[1].imshow(skimage.color.label2rgb(instances, colors=cmap.colors, image=image, alpha=0.7))
        # ax[1].imshow(colors[instances % num_col] * 0.7 + (image/255) * 0.3)
        # ax[1].set_title('2D baseline')
        # # ax[2].imshow(skimage.color.label2rgb(instances_3d, colors=cmap.colors, image=image, alpha=0.7))
        # ax[2].imshow(colors[instances_3d % num_col] * 0.7 + (image/255) * 0.3)
        # ax[2].set_title('Ours')
        # # ax[3].imshow(skimage.color.label2rgb(categories, colors=cmap.colors, image=image, alpha=0.7))
        # ax[3].imshow(colors[categories % num_col] * 0.7 + (image/255) * 0.3)
        # ax[3].set_title('2D/Ours - Categories')

        # [x.axis('off') for x in ax]

        if save_fig:
            # ims += [utils.fig2im(f)]
            ims += [stack_images(image, visor, instances, instances_3d, instances_masa, categories, colors, num_col)]

if save_fig:
    utils.write_mp4(vid+f'_tracks_{segtype_3d}'+(class_name if class_name else ''), ims, fps=5)
