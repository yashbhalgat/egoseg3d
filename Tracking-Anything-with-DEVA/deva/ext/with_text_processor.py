from os import path
from typing import Dict, List

import cv2
import torch
import torch.nn.functional as F
import numpy as np

from deva.inference.object_info import ObjectInfo
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.frame_utils import FrameInfo
from deva.inference.result_utils import ResultSaver
from deva.inference.demo_utils import get_input_frame_for_deva
from deva.ext.grounding_dino import segment_with_text
# try:
#     from groundingdino.util.inference import Model as GroundingDINOModel
# except ImportError:
#     # not sure why this happens sometimes
#     from GroundingDINO.groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything import SamPredictor


def make_segmentation_with_text(cfg: Dict, image_np: np.ndarray, gd_model,
                                sam_model: SamPredictor, prompts: List[str],
                                min_side: int) -> (torch.Tensor, List[ObjectInfo]):
    mask, segments_info = segment_with_text(cfg, gd_model, sam_model, image_np, prompts, min_side)
    return mask, segments_info


@torch.inference_mode()
def process_frame_with_text(deva: DEVAInferenceCore,
                            gd_model,
                            sam_model: SamPredictor,
                            frame_path: str,
                            result_saver: ResultSaver,
                            ti: int,
                            image_np: np.ndarray = None) -> None:
    # image_np, if given, should be in RGB
    if image_np is None:
        image_np = cv2.imread(frame_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cfg = deva.config
    raw_prompt = cfg['prompt']
    prompts = raw_prompt.split('.')

    h, w = image_np.shape[:2]
    new_min_side = cfg['size']
    need_resize = new_min_side > 0
    image = get_input_frame_for_deva(image_np, new_min_side)

    frame_name = path.basename(frame_path)
    frame_info = FrameInfo(image, None, None, ti, {
        'frame': [frame_name],
        'shape': [h, w],
    })

    if cfg['temporal_setting'] == 'semionline':
        if ti + cfg['num_voting_frames'] > deva.next_voting_frame:
            mask, segments_info = make_segmentation_with_text(cfg, image_np, gd_model, sam_model,
                                                              prompts, new_min_side)
            frame_info.mask = mask
            frame_info.segments_info = segments_info
            frame_info.image_np = image_np  # for visualization only
            # wait for more frames before proceeding
            deva.add_to_temporary_buffer(frame_info)

            if ti == deva.next_voting_frame:
                # process this clip
                this_image = deva.frame_buffer[0].image
                this_frame_name = deva.frame_buffer[0].name
                this_image_np = deva.frame_buffer[0].image_np

                _, mask, new_segments_info = deva.vote_in_temporary_buffer(
                    keyframe_selection='first')
                prob = deva.incorporate_detection(this_image, mask, new_segments_info)
                deva.next_voting_frame += cfg['detection_every']

                result_saver.save_mask(prob,
                                       this_frame_name,
                                       need_resize=need_resize,
                                       shape=(h, w),
                                       image_np=this_image_np,
                                       prompts=prompts)

                for frame_info in deva.frame_buffer[1:]:
                    this_image = frame_info.image
                    this_frame_name = frame_info.name
                    this_image_np = frame_info.image_np
                    prob = deva.step(this_image, None, None)
                    result_saver.save_mask(prob,
                                           this_frame_name,
                                           need_resize,
                                           shape=(h, w),
                                           image_np=this_image_np,
                                           prompts=prompts)

                deva.clear_buffer()
        else:
            # standard propagation
            prob = deva.step(image, None, None)
            result_saver.save_mask(prob,
                                   frame_name,
                                   need_resize=need_resize,
                                   shape=(h, w),
                                   image_np=image_np,
                                   prompts=prompts)

    elif cfg['temporal_setting'] == 'online':
        if ti % cfg['detection_every'] == 0:
            # incorporate new detections
            mask, segments_info = make_segmentation_with_text(cfg, image_np, gd_model, sam_model,
                                                              prompts, new_min_side)
            frame_info.segments_info = segments_info
            prob = deva.incorporate_detection(image, mask, segments_info)
        else:
            # Run the model on this frame
            prob = deva.step(image, None, None)
        result_saver.save_mask(prob,
                               frame_name,
                               need_resize=need_resize,
                               shape=(h, w),
                               image_np=image_np,
                               prompts=prompts)
    

def get_segmentation_from_groundtruth(gtdir_path, frame_path, image_shape, device):
    # get filename from frame_path
    gt_seg = np.load(path.join(gtdir_path, path.basename(frame_path).split('.')[0] + '.npy'))
    gt_seg = torch.tensor(gt_seg).unsqueeze(0).float()
    
    segments_info = []
    new_h, new_w = image_shape
    output_mask = torch.zeros((new_h, new_w), dtype=torch.int64, device=device)
    for unqid in torch.unique(gt_seg):
        if unqid == -1:
            continue
        mask = (gt_seg == unqid).float()
        mask = F.interpolate(mask.unsqueeze(0), (new_h, new_w), mode='bilinear')[0, 0]
        mask = (mask > 0.5).float()
        output_mask[mask > 0] = unqid.long()
        segments_info.append(ObjectInfo(id=unqid.item(), category_id=unqid.item(), score=1.0))
    return output_mask, segments_info


@torch.inference_mode()
def process_frame_groundtruth(deva: DEVAInferenceCore,
                            frame_path: str,
                            gtdir_path: str,
                            result_saver: ResultSaver,
                            ti: int,
                            image_np: np.ndarray = None) -> None:
    # image_np, if given, should be in RGB
    if image_np is None:
        image_np = cv2.imread(frame_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    cfg = deva.config
    
    h, w = image_np.shape[:2]
    new_min_side = cfg['size']
    need_resize = new_min_side > 0
    image = get_input_frame_for_deva(image_np, new_min_side)

    frame_name = path.basename(frame_path)
    frame_info = FrameInfo(image, None, None, ti, {
        'frame': [frame_name],
        'shape': [h, w],
    })

    if path.exists(path.join(gtdir_path, path.basename(frame_path).split('.')[0] + '.npy')):
        # incorporate new detections
        mask, segments_info = get_segmentation_from_groundtruth(gtdir_path, frame_path, image.shape[-2:], deva.network.pixel_encoder.conv1.weight.device)
        frame_info.segments_info = segments_info
        prob = deva.incorporate_detection(image, mask, segments_info)
    else:
        # Run the model on this frame
        prob = deva.step(image, None, None)
    result_saver.save_mask(prob,
                            frame_name,
                            need_resize=need_resize,
                            shape=(h, w),
                            image_np=image_np,
                            prompts=[None for _ in range(100000)])
