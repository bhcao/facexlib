# Copied from Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import torch
import torch.nn as nn

from facexlib.utils.misc import get_root_logger


def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def make_divisible(x, divisor):
    """
    Returns the nearest number that is divisible by the given divisor.

    Args:
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): The divisor.

    Returns:
        (int): The nearest number divisible by the divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    """
    Verify image size is a multiple of the given stride in each dimension. If the image size is not a multiple of the
    stride, update it to the nearest multiple of the stride that is greater than or equal to the given floor value.

    Args:
        imgsz (int | cList[int]): Image size.
        stride (int): Stride value.
        min_dim (int): Minimum number of dimensions.
        max_dim (int): Maximum number of dimensions.
        floor (int): Minimum allowed value for image size.

    Returns:
        (List[int] | int): Updated image size.
    """
    # Convert stride to integer if it is a tensor
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    # Convert image size to list if it is an integer
    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str):  # i.e. '640' or '[640,640]'
        imgsz = [int(imgsz)] if imgsz.isnumeric() else eval(imgsz)
    else:
        raise TypeError(
            f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
            f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'"
        )

    # Apply max_dim
    if len(imgsz) > max_dim:
        msg = (
            "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list "
            "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        )
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} is not a valid image size. {msg}")
        get_root_logger().warning(f"WARNING ⚠️ updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]
    # Make image size a multiple of the stride
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    # Print warning message if image size was updated
    if sz != imgsz:
        get_root_logger().warning(f"WARNING ⚠️ imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}")

    # Add missing dimensions if necessary
    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz

def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


DEFAULT_CFG_DICT = {
    "task": "detect", # (str) YOLO task, i.e. detect, segment, classify, pose, obb
    "mode": "predict", # (str) YOLO mode, i.e. train, val, predict, export, track, benchmark

    # Val/Test settings ----------------------------------------------------------------------------------------------------
    "val": True, # (bool) validate/test during training
    "split": "val", # (str) dataset split to use for validation, i.e. 'val', 'test' or 'train'
    "save_json": False, # (bool) save results to JSON file
    "save_hybrid": False, # (bool) save hybrid version of labels (labels + additional predictions)
    "conf": None, # (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
    "iou": 0.7, # (float) intersection over union (IoU) threshold for NMS
    "max_det": 300, # (int) maximum number of detections per image
    "half": False, # (bool) use half precision (FP16)
    "dnn": False, # (bool) use OpenCV DNN for ONNX inference
    "plots": True, # (bool) save plots and images during train/val
    
    # Predict settings -----------------------------------------------------------------------------------------------------
    "source": None, # (str, optional) source directory for images or videos
    "vid_stride": 1, # (int) video frame-rate stride
    "stream_buffer": False, # (bool) buffer all streaming frames (True) or return the most recent frame (False)
    "agnostic_nms": False, # (bool) class-agnostic NMS
    "classes": None, # (int | list[int], optional) filter results by class, i.e. classes=0, or classes=[0,2,3]
    "retina_masks": False, # (bool) use high-resolution segmentation masks
    
    # Hyperparameters ------------------------------------------------------------------------------------------------------
    "lr0": 0.01, # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    "lrf": 0.01, # (float) final learning rate (lr0 * lrf)
    "momentum": 0.937, # (float) SGD momentum/Adam beta1
    "weight_decay": 0.0005, # (float) optimizer weight decay 5e-4
    "warmup_epochs": 3.0, # (float) warmup epochs (fractions ok)
    "warmup_momentum": 0.8, # (float) warmup initial momentum
    "warmup_bias_lr": 0.1, # (float) warmup initial bias lr
    "box": 7.5, # (float) box loss gain
    "cls": 0.5, # (float) cls loss gain (scale with pixels)
    "dfl": 1.5, # (float) dfl loss gain
    "pose": 12.0, # (float) pose loss gain
    "kobj": 1.0, # (float) keypoint obj loss gain
    "nbs": 64, # (int) nominal batch size
    "hsv_h": 0.015, # (float) image HSV-Hue augmentation (fraction)
    "hsv_s": 0.7, # (float) image HSV-Saturation augmentation (fraction)
    "hsv_v": 0.4, # (float) image HSV-Value augmentation (fraction)
    "degrees": 0.0, # (float) image rotation (+/- deg)
    "translate": 0.1, # (float) image translation (+/- fraction)
    "scale": 0.5, # (float) image scale (+/- gain)
    "shear": 0.0, # (float) image shear (+/- deg)
    "perspective": 0.0, # (float) image perspective (+/- fraction), range 0-0.001
    "flipud": 0.0, # (float) image flip up-down (probability)
    "fliplr": 0.5, # (float) image flip left-right (probability)
    "bgr": 0.0, # (float) image channel BGR (probability)
    "mosaic": 1.0, # (float) image mosaic (probability)
    "mixup": 0.0, # (float) image mixup (probability)
    "copy_paste": 0.0, # (float) segment copy-paste (probability)
    "copy_paste_mode": "flip", # (str) the method to do copy_paste augmentation (flip, mixup)
    "auto_augment": "randaugment", # (str) auto augmentation policy for classification (randaugment, autoaugment, augmix)
    "erasing": 0.4, # (float) probability of random erasing during classification training (0-0.9), 0 means no erasing, must be less than 1.0.
    "crop_fraction": 1.0, # (float) image crop fraction for classification (0.1-1), 1.0 means no crop, must be greater than 0.
}