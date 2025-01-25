import os, cv2, torch, time
import numpy as np
from torchvision import transforms
from PIL import Image
import sys
sys.path.append('/work/yashsb/Depth-Anything')

from feature_model import DINOv2Extractor
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from transformers import Owlv2Processor, Owlv2ForObjectDetection



def run_dino(img, feature_extractor):
    return feature_extractor.model(**img)

def run_depth_anything(img, depth_anything):
    return depth_anything(img)

def run_owlv2(inputs, owl_model):
    return owl_model(**inputs)



if __name__ == "__main__":
    # load DINOv2
    dinov2 = DINOv2Extractor(device=torch.device("cuda"))
    # load DepthAnything
    os.chdir('/work/yashsb/Depth-Anything')
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format('vits')).to('cuda').eval()
    depth_transform = transforms.Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    os.chdir('/work/yashsb/OSNOM-Lang/scripts')
    # load OWLv2
    owl_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to('cuda').eval()

    # Load the image
    image_path = "/work/yashsb/datasets/EPIC-Fields/EPIC-KITCHENS/P01/P01_104/frame_0000008187.jpg"
    orig_image = cv2.imread(image_path)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    
    # Preprocess the image for DINOv2
    image_for_dino = dinov2.preprocess_image(orig_image, mask=np.ones_like(orig_image[..., 0] > 0))

    # Preprocess the image for DepthAnything
    image_for_depth = orig_image / 255.0
    h, w = image_for_depth.shape[:2]
    image_for_depth = depth_transform({'image': image_for_depth})['image']
    image_for_depth = torch.from_numpy(image_for_depth).unsqueeze(0).to('cuda')

    # Preprocess the image for OWLv2
    texts = [[f"a photo of table"]]
    image_for_owlv2 = owl_processor(text=texts, images=Image.fromarray(orig_image), return_tensors="pt").to('cuda')

    # Create CUDA streams
    stream_dino = torch.cuda.Stream()
    stream_depth = torch.cuda.Stream()
    stream_owl = torch.cuda.Stream()

    num_iterations = 100

    # Warm-up iterations (optional but recommended)
    for _ in range(10):
        with torch.no_grad():
            with torch.cuda.stream(stream_dino):
                run_dino(image_for_dino, dinov2)
            with torch.cuda.stream(stream_depth):
                run_depth_anything(image_for_depth, depth_anything)
            with torch.cuda.stream(stream_owl):
                run_owlv2(image_for_owlv2, owl_model)
        torch.cuda.synchronize()

    # Measure the time for num_iterations
    start_time = time.time()

    for i in range(num_iterations):
        with torch.no_grad():
            with torch.cuda.stream(stream_dino):
                run_dino(image_for_dino, dinov2)
            with torch.cuda.stream(stream_depth):
                run_depth_anything(image_for_depth, depth_anything)
            if i % 5 == 0:
                with torch.cuda.stream(stream_owl):
                    run_owlv2(image_for_owlv2, owl_model)

        # Synchronize streams to ensure all computations are done
        torch.cuda.synchronize()

    end_time = time.time()

    # Calculate FPS
    total_time = end_time - start_time
    fps = num_iterations / total_time

    print(f"FPS: {fps:.2f}")
