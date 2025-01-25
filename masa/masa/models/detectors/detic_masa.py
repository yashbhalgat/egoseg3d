"""
Author: Siyuan Li
Licensed: Apache-2.0 License
"""
from typing import List, Union

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmengine.logging import print_log
from torch import Tensor

from projects.Detic_new.detic import Detic


def encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code.

    Args:
        mask_results (list): bitmap mask results.

    Returns:
        list | tuple: RLE encoded mask.
    """
    encoded_mask_results = []
    for mask in mask_results:
        encoded_mask_results.append(
            mask_util.encode(
                np.array(mask[:, :, np.newaxis], order="F", dtype="uint8")
            )[0]
        )  # encoded with RLE
    return encoded_mask_results


class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name="ViT-B/32"):
        super().__init__()
        import clip
        from clip.simple_tokenizer import SimpleTokenizer

        self.tokenizer = SimpleTokenizer()
        pretrained_model, _ = clip.load(model_name, device="cpu")
        self.clip = pretrained_model

    @property
    def device(self):
        return self.clip.device

    @property
    def dtype(self):
        return self.clip.dtype

    def tokenize(
        self, texts: Union[str, List[str]], context_length: int = 77
    ) -> torch.LongTensor:
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [
            [sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts
        ]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                st = torch.randint(len(tokens) - context_length + 1, (1,))[0].item()
                tokens = tokens[st : st + context_length]
            result[i, : len(tokens)] = torch.tensor(tokens)

        return result

    def forward(self, text):
        text = self.tokenize(text)
        text_features = self.clip.encode_text(text)
        return text_features


def get_class_weight(original_caption, prompt_prefix="a "):
    if isinstance(original_caption, str):
        if original_caption == "coco":
            from mmdet.datasets import CocoDataset

            class_names = CocoDataset.METAINFO["classes"]
        elif original_caption == "cityscapes":
            from mmdet.datasets import CityscapesDataset

            class_names = CityscapesDataset.METAINFO["classes"]
        elif original_caption == "voc":
            from mmdet.datasets import VOCDataset

            class_names = VOCDataset.METAINFO["classes"]
        elif original_caption == "openimages":
            from mmdet.datasets import OpenImagesDataset

            class_names = OpenImagesDataset.METAINFO["classes"]
        elif original_caption == "lvis":
            from mmdet.datasets import LVISV1Dataset

            class_names = LVISV1Dataset.METAINFO["classes"]
        else:
            if not original_caption.endswith("."):
                original_caption = original_caption + " . "
            original_caption = original_caption.split(" . ")
            class_names = list(filter(lambda x: len(x) > 0, original_caption))

    # for test.py
    else:
        class_names = list(original_caption)

    text_encoder = CLIPTextEncoder()
    text_encoder.eval()
    texts = [prompt_prefix + x for x in class_names]
    print_log(f"Computing text embeddings for {len(class_names)} classes.")
    embeddings = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return class_names, embeddings


def reset_cls_layer_weight(roi_head, weight):
    if type(weight) == str:
        print_log(f"Resetting cls_layer_weight from file: {weight}")
        zs_weight = (
            torch.tensor(np.load(weight), dtype=torch.float32)
            .permute(1, 0)
            .contiguous()
        )  # D x C
    else:
        zs_weight = weight
    zs_weight = torch.cat(
        [zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1))], dim=1
    )  # D x (C + 1)
    zs_weight = F.normalize(zs_weight, p=2, dim=0)
    zs_weight = zs_weight.to("cuda")
    num_classes = zs_weight.shape[-1]

    for bbox_head in roi_head.bbox_head:
        bbox_head.num_classes = num_classes
        del bbox_head.fc_cls.zs_weight
        bbox_head.fc_cls.zs_weight = zs_weight


@MODELS.register_module()
class DeticMasa(Detic):
    def predict(
        self,
        batch_inputs: Tensor,
        detection_features: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True,
    ) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # For single image inference
        if "custom_entities" in batch_data_samples[0]:
            text_prompts = batch_data_samples[0].text
            if text_prompts != self._text_prompts:
                self._text_prompts = text_prompts
                class_names, zs_weight = get_class_weight(text_prompts)
                self._entities = class_names
                reset_cls_layer_weight(self.roi_head, zs_weight)

        assert self.with_bbox, "Bbox head must be implemented."

        # x = self.extract_feat(batch_inputs)
        x = detection_features

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get("proposals", None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False
            )
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale
        )

        for data_sample, pred_instances in zip(batch_data_samples, results_list):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    label_names.append(self._entities[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances

        return batch_data_samples
