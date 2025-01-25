import os, sys, json, torch
from tqdm import tqdm
from os import path
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image

from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.data.simple_video_reader import no_collate
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from deva.ext.grounding_dino import get_grounding_dino_model
# from deva.ext.with_text_processor import process_frame_with_text as process_frame
from deva.ext.with_text_processor import process_frame_groundtruth

from transformers import Owlv2Processor, Owlv2ForObjectDetection

### Import for DETIC
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
sys.path.insert(0, '../Detic/')
sys.path.insert(0, '../Detic/third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder


class SimpleVideoReader(Dataset):
    def __init__(self, image_dir, subsample_factor=1):
        """
        image_dir - points to a directory of jpg images
        """
        self.image_dir = image_dir
        self.subsample_factor = subsample_factor
        frames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg') or f.endswith('.png')])

        self.frames = frames[:20000][::self.subsample_factor]

    def __getitem__(self, idx):
        frame = self.frames[idx]

        im_path = path.join(self.image_dir, frame)
        img = Image.open(im_path).convert('RGB')
        img = np.array(img)

        return img, im_path

    def __len__(self):
        return len(self.frames)

def load_Detic_model():
    ### Change dir to "../Detic", change back at the end.
    orig_dir = os.getcwd(); print("Current working directory: ", orig_dir)
    os.chdir("../Detic"); print("Changed working directory to: ", os.getcwd())

    cfg = get_cfg(); add_centernet_config(cfg); add_detic_config(cfg)
    cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
    predictor = DefaultPredictor(cfg)

    os.chdir(orig_dir); print("Changed working directory back to: ", os.getcwd())
    return predictor

def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

def add_vocab_to_Detic(predictor, vocab_list=['headphone','webcam','paper','coffee']):
    metadata = MetadataCatalog.get("__unused")
    metadata.thing_classes = vocab_list
    classifier = get_clip_embeddings(metadata.thing_classes)
    reset_cls_test(predictor.model, classifier, len(metadata.thing_classes))
    # Reset visualization threshold
    for cascade_stages in range(len(predictor.model.roi_heads.box_predictor)):
        predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = 0.3

def postprocess_deva_gt(deva_pred, segs, original_gtsegs, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    ins2cat_id = {}
    for anno in deva_pred["annotations"]:
        for seginfo in anno["segments_info"]:
            # ins_ids[seginfo["category_id"]].append(seginfo["id"])
            ins2cat_id[seginfo["id"]] = seginfo["category_id"]
    
    out_segs = {}
    for fname, seg in segs.items():
        uniq_ids = np.unique(seg)
        uniq_ids = uniq_ids[uniq_ids != 0]
        out_seg = np.zeros_like(seg)
        for ins_id in uniq_ids:
            out_seg[seg == ins_id] = ins2cat_id[ins_id]
        out_segs[fname] = out_seg

    # Original GT segs are good, where they exist. Take union of original GT segs and deva segs
    for fname, orig_seg in original_gtsegs.items():
        orig_seg = np.array(Image.fromarray(orig_seg).resize((out_segs[fname].shape[1], out_segs[fname].shape[0]), Image.NEAREST))
        out_segs[fname] = np.maximum(out_segs[fname], orig_seg)

    for fname, seg in out_segs.items():
        np.save(os.path.join(save_dir, fname.replace('.jpg', '.npy')), seg)


if __name__ == '__main__':
    os.chdir('/work/yashsb/OSNOM-Lang/Tracking-Anything-with-DEVA')

    torch.autograd.set_grad_enabled(False)

    # for id2rgb
    np.random.seed(42)
    """
    Arguments loading
    """
    parser = ArgumentParser()
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to the groundtruth segmentations')

    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_text_default_args(parser)
    deva_model, cfg, args = get_model_and_config(parser)
    gd_model, sam_model = get_grounding_dino_model(cfg, 'cuda')
    gd_model = ("groundingdino", gd_model)

    if 'classes' in args:
        print('--- OVERWRITING PROMPT WITH VIDEO CLASSES ---')
        classes_cache_path = cfg['classes']
        classes = torch.load(classes_cache_path)
        cfg['prompt'] = '.'.join(classes['class_names_20k_filtered'])
        print('Prompt:')
        print(cfg['prompt'])

    if args.detector_type == 'owlv2':
        # HACK: using OWLv2
        del gd_model
        owl_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        owl_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        gd_model = ("owlv2", owl_processor, owl_model)
    elif args.detector_type == 'detic':
        # HACK: using DETIC
        del gd_model
        gd_model = load_Detic_model()
        add_vocab_to_Detic(gd_model, vocab_list=cfg["prompt"].split('.'))
        gd_model = ("detic", gd_model)

    cfg['temporal_setting'] = args.temporal_setting.lower()
    assert cfg['temporal_setting'] in ['semionline', 'online']

    # get data
    video_reader = SimpleVideoReader(cfg['img_path'], subsample_factor=args.subsample_factor)
    loader = DataLoader(video_reader, batch_size=None, collate_fn=no_collate, num_workers=8)
    out_path = cfg['output']

    # Start eval
    vid_length = len(loader)
    # no need to count usage for LT if the video is not that long anyway
    cfg['enable_long_term_count_usage'] = (
        cfg['enable_long_term']
        and (vid_length / (cfg['max_mid_term_frames'] - cfg['min_mid_term_frames']) *
             cfg['num_prototypes']) >= cfg['max_long_term_elements'])

    print('Configuration:', cfg)

    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = cfg['num_voting_frames'] - 1
    deva.enabled_long_id()
    result_saver = ResultSaver(out_path, None, dataset='demo', object_manager=deva.object_manager)

    with torch.cuda.amp.autocast(enabled=cfg['amp']):
        for ti, (frame, im_path) in enumerate(tqdm(loader)):
            process_frame_groundtruth(deva, im_path, args.gt_dir, result_saver, ti, image_np=frame)
        flush_buffer(deva, result_saver)
    result_saver.end()

    categories = cfg["prompt"].split('.')
    category_name2id = {category: i for i, category in enumerate(categories)}
    # save this as a video-level json
    with open(path.join(out_path, 'pred.json'), 'w') as f:
        result_saver.video_json["category_name2id"] = category_name2id
        json.dump(result_saver.video_json, f, indent=4)  # prettier json

    ### POSTPROCESSING DEVA GT -- replace instance_id with category_id in Annotations/Raw
    segs = {}
    for fname in os.listdir(os.path.join(result_saver.output_root, "Annotations", "Raw")):
        if not fname.endswith('.npy'):
            continue
        segs[fname] = np.load(os.path.join(result_saver.output_root, "Annotations", "Raw", fname))
    original_gtsegs = {}
    for fname in os.listdir(args.gt_dir):
        if not fname.endswith('.npy'):
            continue
        original_gtsegs[fname] = np.load(os.path.join(args.gt_dir, fname))
    postprocess_deva_gt(result_saver.video_json, segs, original_gtsegs, os.path.join(result_saver.output_root, "Annotations", "Postprocessed"))

