"""
Code adapted from timm https://github.com/huggingface/pytorch-image-models

Modifications and additions for mivolo by / Copyright 2023, Irina Tolstykh, Maxim Kuprashevich
"""

import os
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
from facexlib.utils.image_dto import ImageDTO
from facexlib.utils.misc import box_iou

# because of ultralytics bug it is important to unset CUBLAS_WORKSPACE_CONFIG after the module importing
os.unsetenv("CUBLAS_WORKSPACE_CONFIG")

import timm

# register new models
from facexlib.genderage.mivolo_model import *  # noqa: F403, F401
from timm.layers import set_layer_config
from timm.models._factory import parse_model_name
from timm.models._helpers import load_state_dict, remap_state_dict
from timm.models._hub import load_model_config_from_hf
from timm.models._pretrained import PretrainedCfg
from timm.models._registry import is_model, model_entrypoint, split_model_name_tag

def load_checkpoint(
    model, checkpoint_path, use_ema=True, strict=True, remap=False, filter_keys=None, state_dict_map=None
):
    if os.path.splitext(checkpoint_path)[-1].lower() in (".npz", ".npy"):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, "load_pretrained"):
            timm.models._model_builder.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError("Model cannot load numpy checkpoint")
        return
    state_dict = load_state_dict(checkpoint_path, use_ema)
    if remap:
        state_dict = remap_state_dict(model, state_dict)
    if filter_keys:
        for sd_key in list(state_dict.keys()):
            for filter_key in filter_keys:
                if filter_key in sd_key:
                    if sd_key in state_dict:
                        del state_dict[sd_key]

    rep = []
    if state_dict_map is not None:
        # 'patch_embed.conv1.' : 'patch_embed.conv.'
        for state_k in list(state_dict.keys()):
            for target_k, target_v in state_dict_map.items():
                if target_v in state_k:
                    target_name = state_k.replace(target_v, target_k)
                    state_dict[target_name] = state_dict[state_k]
                    rep.append(state_k)
        for r in rep:
            if r in state_dict:
                del state_dict[r]

    incompatible_keys = model.load_state_dict(state_dict, strict=strict if filter_keys is None else False)
    return incompatible_keys


def create_model(
    model_name: str,
    pretrained: bool = False,
    pretrained_cfg: Optional[Union[str, Dict[str, Any], PretrainedCfg]] = None,
    pretrained_cfg_overlay: Optional[Dict[str, Any]] = None,
    checkpoint_path: str = "",
    scriptable: Optional[bool] = None,
    exportable: Optional[bool] = None,
    no_jit: Optional[bool] = None,
    filter_keys=None,
    state_dict_map=None,
    **kwargs,
):
    """Create a model
    Lookup model's entrypoint function and pass relevant args to create a new model.
    """
    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_name = parse_model_name(model_name)
    if model_source == "hf-hub":
        assert not pretrained_cfg, "pretrained_cfg should not be set when sourcing model from Hugging Face Hub."
        # For model names specified in the form `hf-hub:path/architecture_name@revision`,
        # load model weights + pretrained_cfg from Hugging Face hub.
        pretrained_cfg, model_name = load_model_config_from_hf(model_name)
    else:
        model_name, pretrained_tag = split_model_name_tag(model_name)
        if not pretrained_cfg:
            # a valid pretrained_cfg argument takes priority over tag in model name
            pretrained_cfg = pretrained_tag

    if not is_model(model_name):
        raise RuntimeError("Unknown model (%s)" % model_name)

    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        model = create_fn(
            pretrained=pretrained,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=pretrained_cfg_overlay,
            **kwargs,
        )

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, filter_keys=filter_keys, state_dict_map=state_dict_map)

    return model


def crop_object(
    bboxes: np.ndarray, full_image: ImageDTO, ind: int, is_person: bool = False
) -> Tuple[ImageDTO]:
    '''
    Crop object from image based on bboxes.

    Args:
        bboxes: Numpy array of shape (N, 10) containing N detected objects of format (face_bbox, person_bbox)
        full_image: ImageDTO object containing full image
        ind: Index of object to crop
        is_person: If True, mask out faces in case of intersection.
    '''

    zero_img = ImageDTO(np.zeros((1, 1, 3), dtype=np.uint8))

    IOU_THRESH = 0.000001
    MIN_PERSON_CROP_AFTERCUT_RATIO = 0.4
    CROP_ROUND_RATE = 0.3
    MIN_PERSON_SIZE = 50

    if is_person:
        obj_bbox = bboxes[ind, 5:9]
    else:
        obj_bbox = bboxes[ind, :4]

    if np.isnan(obj_bbox[0]):
        return zero_img
    
    x1, y1, x2, y2 = obj_bbox.astype(int)
    # get crop of face or person
    if isinstance(full_image.image, torch.Tensor):
        obj_image = full_image.image[:, y1:y2, x1:x2].clone()
        crop_h, crop_w = obj_image.shape[1:]
    else:
        obj_image = full_image.image[y1:y2, x1:x2].copy()
        crop_h, crop_w = obj_image.shape[:2]

    if is_person and (crop_h < MIN_PERSON_SIZE or crop_w < MIN_PERSON_SIZE):
        return zero_img

    if not is_person:
        return ImageDTO(obj_image)

    # calc iou between obj_bbox and other bboxes
    iou_matrix = box_iou(obj_bbox, bboxes[:, :4])

    # cut out other faces in case of intersection
    for other_ind, (det, iou) in enumerate(zip(bboxes[:, :4], iou_matrix)):
        if iou < IOU_THRESH or np.isnan(det[0]):
            continue
        o_x1, o_y1, o_x2, o_y2 = det.astype(int)

        # remap current_person_bbox to reference_person_bbox coordinates
        o_x1 = max(o_x1 - x1, 0)
        o_y1 = max(o_y1 - y1, 0)
        o_x2 = min(o_x2 - x1, crop_w)
        o_y2 = min(o_y2 - y1, crop_h)

        if isinstance(full_image.image, torch.Tensor):
            obj_image[:, o_y1:o_y2, o_x1:o_x2] = 0
        else:
            obj_image[o_y1:o_y2, o_x1:o_x2] = 0

    iou_matrix = box_iou(obj_bbox, bboxes[:, 5:9])

    # cut out other persons in case of intersection
    for other_ind, (det, iou) in enumerate(zip(bboxes[:, 5:9], iou_matrix)):
        if ind == other_ind or iou < IOU_THRESH:
            continue
        o_x1, o_y1, o_x2, o_y2 = det.astype(int)

        # remap current_person_bbox to reference_person_bbox coordinates
        o_x1 = max(o_x1 - x1, 0)
        o_y1 = max(o_y1 - y1, 0)
        o_x2 = min(o_x2 - x1, crop_w)
        o_y2 = min(o_y2 - y1, crop_h)

        if (o_y1 / crop_h) < CROP_ROUND_RATE:
            o_y1 = 0
        if ((crop_h - o_y2) / crop_h) < CROP_ROUND_RATE:
            o_y2 = crop_h
        if (o_x1 / crop_w) < CROP_ROUND_RATE:
            o_x1 = 0
        if ((crop_w - o_x2) / crop_w) < CROP_ROUND_RATE:
            o_x2 = crop_w
        
        if isinstance(full_image.image, torch.Tensor):
            obj_image[:, o_y1:o_y2, o_x1:o_x2] = 0
        else:
            obj_image[o_y1:o_y2, o_x1:o_x2] = 0

    count_nonzero = np.count_nonzero if isinstance(obj_image, np.ndarray) else torch.count_nonzero
    remain_ratio = count_nonzero(obj_image) / (obj_image.shape[0] * obj_image.shape[1] * obj_image.shape[2])
    if remain_ratio < MIN_PERSON_CROP_AFTERCUT_RATIO:
        return zero_img

    return ImageDTO(obj_image)