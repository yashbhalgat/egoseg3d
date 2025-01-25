# Reference: https://github.com/IDEA-Research/Grounded-Segment-Anything

from typing import Dict, List
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
import supervision as sv

try:
    from groundingdino.util.inference import Model as GroundingDINOModel
except ImportError:
    # not sure why this happens sometimes
    from GroundingDINO.groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything import sam_model_registry, SamPredictor, sam_hq_model_registry
from deva.ext.MobileSAM.setup_mobile_sam import setup_model as setup_mobile_sam
from deva.ext.LightHQSAM.setup_light_hqsam import setup_model as setup_light_hqsam
import numpy as np
import torch

from deva.inference.object_info import ObjectInfo


def get_grounding_dino_model(config: Dict, device: str) -> (GroundingDINOModel, SamPredictor):
    GROUNDING_DINO_CONFIG_PATH = config['GROUNDING_DINO_CONFIG_PATH']
    GROUNDING_DINO_CHECKPOINT_PATH = config['GROUNDING_DINO_CHECKPOINT_PATH']

    gd_model = GroundingDINOModel(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                  model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
                                  device=device)

    # Building SAM Model and SAM Predictor
    variant = config['sam_variant'].lower()
    if variant == 'mobile':
        MOBILE_SAM_CHECKPOINT_PATH = config['MOBILE_SAM_CHECKPOINT_PATH']

        # Building Mobile SAM model
        checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
        mobile_sam = setup_mobile_sam()
        mobile_sam.load_state_dict(checkpoint, strict=True)
        mobile_sam.to(device=device)
        sam = SamPredictor(mobile_sam)
    elif variant == 'original':
        SAM_ENCODER_VERSION = config['SAM_ENCODER_VERSION']
        SAM_CHECKPOINT_PATH = config['SAM_CHECKPOINT_PATH']

        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
            device=device)
        sam = SamPredictor(sam)
    elif variant == 'sam_hq':
        # Building HQ-SAM model with better Mask quality
        SAM_ENCODER_VERSION = config['SAM_ENCODER_VERSION']
        HQ_SAM_CHECKPOINT_PATH = config['HQ_SAM_CHECKPOINT_PATH']
        sam_hq = sam_hq_model_registry[SAM_ENCODER_VERSION](checkpoint=HQ_SAM_CHECKPOINT_PATH).to(
            device=device)
        sam = SamPredictor(sam_hq)
    elif variant == 'sam_hq_light':
        LIGHT_HQ_SAM_CHECKPOINT_PATH = config['LIGHT_HQ_SAM_CHECKPOINT_PATH']

        # Building Light HQ-SAM model with good Mask quality and efficiency
        checkpoint = torch.load(LIGHT_HQ_SAM_CHECKPOINT_PATH)
        light_hq_sam = setup_light_hqsam()
        light_hq_sam.load_state_dict(checkpoint, strict=True)
        light_hq_sam.to(device=device)
        sam = SamPredictor(light_hq_sam)

    return gd_model, sam


def segment_with_text(config: Dict, gd_model, sam: SamPredictor,
                      image: np.ndarray, prompts: List[str],
                      min_side: int) -> (torch.Tensor, List[ObjectInfo]):
    """
    HACK:
    DEVA uses GroundingDINO, so gd_model is that by default
    We want to use OWLv2. So, gd_model in this case will be a tuple: (owl_processor, owl_model)
    """

    BOX_THRESHOLD = TEXT_THRESHOLD = config['DINO_THRESHOLD']
    NMS_THRESHOLD = config['DINO_NMS_THRESHOLD']

    sam.set_image(image, image_format='RGB')
    h, w = image.shape[:2]

    if gd_model[0] == "groundingdino":
        ############# GroundingDINO detector
        _, grounding_dino_model = gd_model
        detections = grounding_dino_model.predict_with_classes(image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                                classes=prompts,
                                                box_threshold=BOX_THRESHOLD,
                                                text_threshold=TEXT_THRESHOLD)
    elif gd_model[0] == "owlv2":
        ############# OWLv2 detector
        _, owl_processor, owl_model = gd_model
        texts = [[f"a photo of {prompt}" for prompt in prompts]]
        inputs = owl_processor(text=texts, images=Image.fromarray(image), return_tensors="pt")
        outputs = owl_model(**inputs)
        results = owl_processor.post_process_object_detection(outputs=outputs, target_sizes=torch.Tensor([[h,w]]), threshold=0.1)
        conf_mask = results[0]["scores"] > BOX_THRESHOLD
        detections = sv.Detections(xyxy=results[0]["boxes"][conf_mask].detach().numpy(),
                                   confidence=results[0]["scores"][conf_mask].detach().numpy(),
                                   class_id=results[0]["labels"][conf_mask].detach().numpy())
    elif gd_model[0] == "detic":
        ############# Detic detector
        _, detic_model = gd_model
        outputs = detic_model(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        conf_mask = outputs["instances"].scores > BOX_THRESHOLD
        detections = sv.Detections(xyxy=outputs["instances"].pred_boxes.tensor[conf_mask].cpu().numpy(),
                                confidence=outputs["instances"].scores[conf_mask].cpu().numpy(),
                                class_id=outputs["instances"].pred_classes[conf_mask].cpu().numpy())
    


    nms_idx = torchvision.ops.nms(torch.from_numpy(detections.xyxy),
                                  torch.from_numpy(detections.confidence),
                                  NMS_THRESHOLD).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    result_masks = []
    for box in detections.xyxy:
        masks, scores, _ = sam.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])

    detections.mask = np.array(result_masks)

    if min_side > 0:
        scale = min_side / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        new_h, new_w = h, w

    output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device=sam.device)
    curr_id = 1
    segments_info = []

    # sort by descending area to preserve the smallest object
    for i in np.flip(np.argsort(detections.area)):
        mask = detections.mask[i]
        confidence = detections.confidence[i]
        class_id = detections.class_id[i]
        mask = torch.from_numpy(mask.astype(np.float32))
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (new_h, new_w), mode='bilinear')[0, 0]
        mask = (mask > 0.5).float()

        if mask.sum() > 0:
            output_mask[mask > 0] = curr_id
            segments_info.append(ObjectInfo(id=curr_id, category_id=class_id, score=confidence))
            curr_id += 1

    return output_mask, segments_info
