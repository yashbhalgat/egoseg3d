"""
First clone Depth-Anything at this point: https://github.com/LiheYoung/Depth-Anything/tree/1e1c8d373ae6383ef6490a5c2eb5ef29fd085993
Then, place this script in `Depth-Anything/metric_depth` directory and run it.
`python3 single_image.py --input_dir <input dir> --output_dir <output dir>`
"""

import argparse, os, glob, torch, cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config

# Global settings
FL = 715.0873
FY = 256 * 0.6
FX = 256 * 0.6
NYU_DATA = False
DATASET = 'nyu'

def process_images(model, input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = glob.glob(os.path.join(input_dir, '*.png')) + glob.glob(os.path.join(input_dir, '*.jpg'))
    for image_path in tqdm(image_paths, desc="Processing Images"):
        try:
            color_image = Image.open(image_path).convert('RGB')
            original_width, original_height = color_image.size
            image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

            pred = model(image_tensor, dataset=DATASET)
            if isinstance(pred, dict):
                pred = pred.get('metric_depth', pred.get('out'))
            elif isinstance(pred, (list, tuple)):
                pred = pred[-1]
            pred = pred.squeeze().detach().cpu().numpy()

            # Resize color image and depth to final size
            resized_pred = np.array(Image.fromarray(pred).resize((original_width, original_height), Image.NEAREST))
            norm_resized_pred = (resized_pred - np.min(resized_pred)) / (np.max(resized_pred) - np.min(resized_pred) + 1e-6) * 255.0
            norm_resized_pred = norm_resized_pred.astype(np.uint8)

            # Save the resized depth to OUTPUT_DIR
            output_path = os.path.join(output_dir, os.path.basename(image_path).split('.')[0] + '_depth.npy')
            np.save(output_path, resized_pred)
            # save as grayscale
            cv2.imwrite(output_path.replace('.npy', '.png'), norm_resized_pred)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def main(model_name, pretrained_resource, input_dir, output_dir):
    config = get_config(model_name, "eval", DATASET)
    config.pretrained_resource = pretrained_resource
    model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    process_images(model, input_dir, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='zoedepth', help="Name of the model to test")
    parser.add_argument("-p", "--pretrained_resource", type=str, default='local::./checkpoints/depth_anything_metric_depth_indoor.pt', help="Pretrained resource to use for fetching weights.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save output depth maps")

    args = parser.parse_args()
    main(args.model, args.pretrained_resource, args.input_dir, args.output_dir)
