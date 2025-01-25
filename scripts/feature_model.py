import torch, torchvision
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image

import open_clip
from transformers import AutoImageProcessor, Dinov2Model


def get_seg_img(mask, image):
    image = image.copy()
    image[mask==0] = np.array([0, 0,  0], dtype=np.uint8)
    # get bounding box of the mask without using cv2
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    x,y,w,h = np.int32((x0, y0, x1-x0, y1-y0))
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad


class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, image, mask):
        pass


class CLIPExtractor(FeatureExtractor):
    def __init__(self, device):
        self.device = device
        self.load_pretrained_model()

    def load_pretrained_model(self):
        """Load and return the pretrained OpenCLIP model."""
        # Load your model here
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k", precision="fp16",
        )
        model.eval()
        self.model = model.to(self.device)
        
    def extract_features(self, image, mask):
        seg_img = get_seg_img(mask, image)
        pad_seg_img = pad_img(seg_img)
        masked_image = torch.from_numpy(pad_seg_img[None, ...]).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            processed_input = self.process(masked_image).half().to(self.device)
            feature = self.model.encode_image(processed_input)
        return feature


class DINOv2Extractor(FeatureExtractor):
    def __init__(self, device):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = Dinov2Model.from_pretrained('facebook/dinov2-base').to(self.device)

    def preprocess_image(self, image, mask):
        seg_img = get_seg_img(mask, image)
        pad_seg_img = pad_img(seg_img) # H x W x 3
        # masked_image = torch.from_numpy(pad_seg_img[None, ...]).permute(0, 3, 1, 2).float() # 1 x 3 x H x W
        masked_image = Image.fromarray(pad_seg_img)
        inputs = self.processor(masked_image, return_tensors="pt").to(self.device)
        return inputs

    def extract_features(self, image, mask):
        inputs = self.preprocess_image(image, mask)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.pooler_output # [1, 768]


# Dictionary to map 'feature_type' keys to feature extractor classes
FEATURE_EXTRACTORS = {
    "dinov2": DINOv2Extractor,
    "clip": CLIPExtractor
}
