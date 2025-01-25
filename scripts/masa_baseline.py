import os, sys, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc, resource, argparse, tqdm, torch, json
import numpy as np
from torch.multiprocessing import set_start_method

from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmcv.ops.nms import batched_nms

from masa_utils import filter_and_update_tracks
from utils import SimpleVideoReader
from tracking import Scene

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'masa'))
sys.path.insert(0, project_root)
import masa
from masa.apis import inference_masa, init_masa, inference_detector, build_test_pipeline
from masa.models.sam import SamPredictor, sam_model_registry

import warnings
warnings.filterwarnings('ignore')

# Ensure the right start method for multiprocessing
try:
    set_start_method('spawn')
except RuntimeError:
    pass

### NOT used right now
def set_file_descriptor_limit(limit):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))

# Set the file descriptor limit to 65536
# set_file_descriptor_limit(65536)

def visualize_frame(args, visualizer, frame, track_result, frame_idx, fps=None):
    visualizer.add_datasample(name='video_' + str(frame_idx), image=frame[:, :, ::-1],
        data_sample=track_result[0], draw_gt=False, show=False,
        out_file=None, pred_score_thr=args.score_thr, fps=fps,)
    frame = visualizer.get_image()
    gc.collect()
    return frame


def convert_masa_outputs_to_DEVA_format(instances_list, frame_paths):
    masa_segs = {}
    masa_preds_json = {}
    masa_preds_json["annotations"] = []
    for frame_path, instances in tqdm.tqdm(zip(frame_paths, instances_list), desc='Converting to DEVA format'):
        frame_name = os.path.basename(frame_path)
        masa_preds_json["annotations"].append({
                "file_name": frame_name,
                "segments_info": []
            })
        
        seg = np.zeros(instances[0].ori_shape, dtype=np.int32)
        if 'masks' in instances[0].pred_track_instances:
            for instance in instances[0].pred_track_instances:
                seg[instance.masks[0] > 0] = instance.instances_id.item() + 1 # +1 to make the background 0
                masa_preds_json["annotations"][-1]["segments_info"].append({
                    "id": int(instance.instances_id.item()) + 1, # +1 to make the background 0
                    "category_id": int(instance.labels.item()),
                    "score": instance.scores.item(),
                    "area": int((instance.masks[0] > 0).sum()),
                })
        masa_segs[frame_name] = seg
    return masa_segs, masa_preds_json


def parse_args():
    parser = argparse.ArgumentParser(description='MASA video demo')
    parser.add_argument('--img_dir', help='dir path for images')
    parser.add_argument('--classes', type=str, help='path to classes pt file', default=None)
    parser.add_argument('--save_dir', type=str, help='Dir path to save results')
    parser.add_argument('--subsample_factor', type=int, default=1, help='Subsample factor for video reader')
    
    # Default MASA arguments
    parser.add_argument('--det_config', help='Detector Config file')
    parser.add_argument('--masa_config', help='Masa Config file')
    parser.add_argument('--det_checkpoint', help='Detector Checkpoint file')
    parser.add_argument('--masa_checkpoint', help='Masa Checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thr', type=float, default=0.2, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--texts', help='text prompt')
    parser.add_argument('--line_width', type=int, default=5, help='Line width')
    parser.add_argument('--unified', action='store_true', help='Use unified model, which means the masa adapter is built upon the detector model.')
    parser.add_argument('--detector_type', type=str, default='mmdet', help='Choose detector type')
    parser.add_argument('--fp16', action='store_true', help='Activation fp16 mode')
    parser.add_argument('--no-post', action='store_true', help='Do not post-process the results ')
    parser.add_argument('--show_fps', action='store_true', help='Visualize the fps')
    parser.add_argument('--sam_mask', action='store_true', help='Use SAM to generate mask for segmentation tracking')
    parser.add_argument('--sam_path',  type=str, default='saved_models/pretrain_weights/sam_vit_h_4b8939.pth', help='Default path for SAM models')
    parser.add_argument('--sam_type', type=str, default='vit_h', help='Default type for SAM models')
    parser.add_argument('--wait-time', type=float, default=1, help='The interval of show (s), 0 is block')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.classes is not None:
        categories = torch.load(args.classes)
        # Override "texts"
        args.texts = ' . '.join(categories['class_names_20k_filtered'])

    # build the model from a config file and a checkpoint file
    if args.unified:
        masa_model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)
    else:
        det_model = init_detector(args.det_config, args.det_checkpoint, palette='random', device=args.device)
        masa_model = init_masa(args.masa_config, args.masa_checkpoint, device=args.device)
        # build test pipeline
        det_model.cfg.test_dataloader.dataset.pipeline[
            0].type = 'mmdet.LoadImageFromNDArray'
        test_pipeline = Compose(det_model.cfg.test_dataloader.dataset.pipeline)

    if args.sam_mask:
        print('Loading SAM model...')
        device = args.device
        sam_model = sam_model_registry[args.sam_type](args.sam_path)
        sam_predictor = SamPredictor(sam_model.to(device))

    video_reader = SimpleVideoReader(args.img_dir, subsample_factor=args.subsample_factor)
    
    #### parsing the text input
    texts = args.texts
    if texts is not None:
        masa_test_pipeline = build_test_pipeline(masa_model.cfg, with_text=True)
    else:
        masa_test_pipeline = build_test_pipeline(masa_model.cfg)

    if texts is not None:
        masa_model.cfg.visualizer['texts'] = texts
    else:
        masa_model.cfg.visualizer['texts'] = det_model.dataset_meta['classes']

    # init visualizer
    masa_model.cfg.visualizer['save_dir'] = args.save_dir
    masa_model.cfg.visualizer['line_width'] = args.line_width
    if args.sam_mask:
        masa_model.cfg.visualizer['alpha'] = 0.5
    visualizer = VISUALIZERS.build(masa_model.cfg.visualizer)

    frame_idx = 0
    instances_list = []
    frames, frame_paths = [], []
    fps_list = []
    for frame, frame_path in tqdm.tqdm(video_reader, total=len(video_reader)):
        frame = frame[:, :, ::-1] # need this with SimpleVideoReader

        # unified models mean that masa build upon and reuse the foundation model's backbone features for tracking
        if args.unified:
            track_result = inference_masa(masa_model, frame,
                                          frame_id=frame_idx,
                                          video_len=len(video_reader),
                                          test_pipeline=masa_test_pipeline,
                                          text_prompt=texts,
                                          fp16=args.fp16,
                                          detector_type=args.detector_type,
                                          show_fps=args.show_fps)
            if args.show_fps:
                track_result, fps = track_result
        else:

            if args.detector_type == 'mmdet':
                result = inference_detector(det_model, frame,
                                            text_prompt=texts,
                                            test_pipeline=test_pipeline,
                                            fp16=args.fp16)

            # Perfom inter-class NMS to remove nosiy detections
            det_bboxes, keep_idx = batched_nms(boxes=result.pred_instances.bboxes,
                                               scores=result.pred_instances.scores,
                                               idxs=result.pred_instances.labels,
                                               class_agnostic=True,
                                               nms_cfg=dict(type='nms',
                                                             iou_threshold=0.5,
                                                             class_agnostic=True,
                                                             split_thr=100000))

            det_bboxes = torch.cat([det_bboxes,
                                            result.pred_instances.scores[keep_idx].unsqueeze(1)],
                                               dim=1)
            det_labels = result.pred_instances.labels[keep_idx]

            track_result = inference_masa(masa_model, frame, frame_id=frame_idx,
                                          video_len=len(video_reader),
                                          test_pipeline=masa_test_pipeline,
                                          det_bboxes=det_bboxes,
                                          det_labels=det_labels,
                                          fp16=args.fp16,
                                          show_fps=args.show_fps)
            if args.show_fps:
                track_result, fps = track_result

        frame_idx += 1
        if 'masks' in track_result[0].pred_track_instances:
            if len(track_result[0].pred_track_instances.masks) >0:
                track_result[0].pred_track_instances.masks = torch.stack(track_result[0].pred_track_instances.masks, dim=0)
                track_result[0].pred_track_instances.masks = track_result[0].pred_track_instances.masks.cpu().numpy()

        track_result[0].pred_track_instances.bboxes = track_result[0].pred_track_instances.bboxes.to(torch.float32)
        instances_list.append(track_result.to('cpu'))
        frames.append(frame)
        frame_paths.append(frame_path)
        if args.show_fps:
            fps_list.append(fps)

    if not args.no_post:
        instances_list = filter_and_update_tracks(instances_list, (frame.shape[1], frame.shape[0]))

    if args.sam_mask:
        print('Start to generate mask using SAM!')
        for idx, (frame, track_result) in tqdm.tqdm(enumerate(zip(frames, instances_list))):
            track_result = track_result.to(device)
            track_result[0].pred_track_instances.instances_id = track_result[0].pred_track_instances.instances_id.to(device)
            # if (track_result[0].pred_track_instances.scores.float() > args.score_thr).to(device).sum() == 0:
            #     breakpoint()
            track_result[0].pred_track_instances = track_result[0].pred_track_instances[(track_result[0].pred_track_instances.scores.float() > args.score_thr).to(device)]
            input_boxes = track_result[0].pred_track_instances.bboxes
            if len(input_boxes) == 0:
                instances_list[idx] = track_result
            else:
                sam_predictor.set_image(frame)
                transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, frame.shape[:2])
                masks, _, _ = sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                track_result[0].pred_track_instances.masks = masks.squeeze(1).cpu().numpy()
                instances_list[idx] = track_result

            if idx % 10 == 0:
                gc.collect() # release memory
                torch.cuda.empty_cache()

    masa_segs, masa_preds_json = convert_masa_outputs_to_DEVA_format(instances_list, frame_paths)
    categories = [c.strip() for c in texts.split('.')]
    masa_preds_json["category_name2id"] = {category: i for i, category in enumerate(categories)}
    
    # HACK: remove frams from JSON for which we do not have camera poses esimated from COLMAP
    vid = "_".join(args.classes.split('/')[-1].split('.')[0].split('_')[:-1])
    scene = Scene(vid)
    masa_preds_json["annotations"] = [anno for anno in masa_preds_json["annotations"] \
                                                if anno["file_name"].split(".")[0] in scene.frames]
    
    save_dir = args.save_dir
    print(f'Saving results to {save_dir}')
    os.makedirs(save_dir, exist_ok=True)
    anno_save_dir = os.path.join(save_dir, 'Annotations', 'Raw')
    os.makedirs(anno_save_dir, exist_ok=True)
    for frame, seg in masa_segs.items():
        np.save(os.path.join(anno_save_dir, frame.split('.')[0] + '.npy'), seg)
    with open(os.path.join(save_dir, 'pred.json'), 'w') as f:
        json.dump(masa_preds_json, f)


if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'masa')))
    main()
