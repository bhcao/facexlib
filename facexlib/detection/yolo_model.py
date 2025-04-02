# Modified from Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from typing import Any, List, Union
import numpy as np
from PIL import Image
import threading
from pathlib import Path
import torch
import contextlib
from copy import deepcopy

from facexlib.genderage.structures import PersonAndFaceResult
from facexlib.utils.image_dto import ImageDTO
from facexlib.utils.misc import get_root_logger

from .yolo_blocks import (
    C2PSA,
    SPPF,
    C2f,
    C3k2,
    Concat,
    Conv,
    Detect,
)

from . import yolo_ops as ops
from .yolo_results import Results

from .yolo_utils import (
    fuse_conv_and_bn,
    make_divisible,
    check_imgsz,
    initialize_weights,
    DEFAULT_CFG_DICT
)

# because of ultralytics bug it is important to unset CUBLAS_WORKSPACE_CONFIG after the module importing
os.unsetenv("CUBLAS_WORKSPACE_CONFIG")

YOLO_V8X_CFG = {
    "nc": 2,
    "depth_multiple": 1.0,
    "width_multiple": 1.25,
    "backbone": [
        [-1, 1, 'Conv', [64, 3, 2]], 
        [-1, 1, 'Conv', [128, 3, 2]], 
        [-1, 3, 'C2f', [128, True]], 
        [-1, 1, 'Conv', [256, 3, 2]], 
        [-1, 6, 'C2f', [256, True]], 
        [-1, 1, 'Conv', [512, 3, 2]], 
        [-1, 6, 'C2f', [512, True]], 
        [-1, 1, 'Conv', [512, 3, 2]], 
        [-1, 3, 'C2f', [512, True]], 
        [-1, 1, 'SPPF', [512, 5]]
    ],
    "head": [
        [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']], 
        [[-1, 6], 1, 'Concat', [1]], 
        [-1, 3, 'C2f', [512]], 
        [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']], 
        [[-1, 4], 1, 'Concat', [1]], 
        [-1, 3, 'C2f', [256]], 
        [-1, 1, 'Conv', [256, 3, 2]], 
        [[-1, 12], 1, 'Concat', [1]], 
        [-1, 3, 'C2f', [512]], 
        [-1, 1, 'Conv', [512, 3, 2]], 
        [[-1, 9], 1, 'Concat', [1]], 
        [-1, 3, 'C2f', [512]], 
        [[15, 18, 21], 1, 'Detect', ['nc']]
    ],
    "ch": 3,
    'scale': 'x',
}

# TODO: train on yolo11x (SOTA)
YOLO_V11X_CFG = {
    'nc': 80,
    "depth_multiple": 1.0,
    "width_multiple": 1.5,
    "max_channels": 512,
    'backbone': [
        [-1, 1, 'Conv', [64, 3, 2]],
        [-1, 1, 'Conv', [128, 3, 2]],
        [-1, 2, 'C3k2', [256, False, 0.25]],
        [-1, 1, 'Conv', [256, 3, 2]],
        [-1, 2, 'C3k2', [512, False, 0.25]],
        [-1, 1, 'Conv', [512, 3, 2]],
        [-1, 2, 'C3k2', [512, True]],
        [-1, 1, 'Conv', [1024, 3, 2]],
        [-1, 2, 'C3k2', [1024, True]],
        [-1, 1, 'SPPF', [1024, 5]],
        [-1, 2, 'C2PSA', [1024]]
    ],
    'head': [
        [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
        [[-1, 6], 1, 'Concat', [1]],
        [-1, 2, 'C3k2', [512, False]],
        [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
        [[-1, 4], 1, 'Concat', [1]],
        [-1, 2, 'C3k2', [256, False]],
        [-1, 1, 'Conv', [256, 3, 2]],
        [[-1, 13], 1, 'Concat', [1]],
        [-1, 2, 'C3k2', [512, False]],
        [-1, 1, 'Conv', [512, 3, 2]],
        [[-1, 10], 1, 'Concat', [1]],
        [-1, 2, 'C3k2', [1024, True]],
        [[16, 19, 22], 1, 'Detect', ['nc']]
    ],
    'ch': 3,
    'scale': 'x',
}

def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = d.get("max_channels", float("inf"))
    nc, act, scale = (d.get(x) for x in ("nc", "activation", "scale"))
    depth, width = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple"))

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            get_root_logger().info(f"activation: {act}")  # print

    if verbose:
        get_root_logger().info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    base_modules = frozenset({Conv, SPPF, C2PSA, C2f, C3k2, torch.nn.ConvTranspose2d})
    # modules with 'repeat' arguments
    repeat_modules = frozenset({C2f, C3k2, C2PSA})
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m == Detect:
            args.append([ch[x] for x in f])
            m.legacy = legacy
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            get_root_logger().info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)


class YOLODetectionModel(torch.nn.Module):
    """YOLO detection model."""

    def __init__(
        self,
        device: str = "cuda",
        half: bool = True,
        verbose: bool = False,
        conf_thresh: float = 0.4,
        iou_thresh: float = 0.7,
        yaml_cfg = YOLO_V8X_CFG,
        cfg = DEFAULT_CFG_DICT,
        names = None,
        ch = 3,
        nc = None,
    ) -> None:
        super().__init__()
        
        self.overrides = {}  # overrides for trainer object

        # Load or create new YOLO model
        __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # to avoid deterministic warnings
        
        self.yaml = yaml_cfg  # cfg dict
        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            get_root_logger().info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])} if names is None else names  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Performs a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR
        
        self.stride = max(int(self.stride.max()), 32)  # model stride
        self.device = torch.device(device)

        # Init weights, biases
        initialize_weights(self)
    
        self.training = False  # set model to eval mode
    
        # Module updates
        for m in self.modules():
            if hasattr(m, "inplace"):
                m.inplace = True
            elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility

        # only remember these arguments when loading a PyTorch model
        self.overrides = {"imgsz": 640, "single_cls": False, "conf": 0.25, "batch": 1, "save": False, "mode": "predict"}
        self.overrides.update({"conf": conf_thresh, "iou": iou_thresh, "half": half, "verbose": verbose})

        self.fp16 = half  # FP16
        if half and next(self.parameters()).type != "cpu":
            self.model = self.model.half()
        
        self.args = cfg
        self.done_warmup = False

        # Usable if setup is done
        self.imgsz = None
        self.plotted_img = None
        self.txt_path = None
        self._lock = threading.Lock()  # for automatic thread-safe inference


    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x


    @torch.no_grad()
    def fuse(self, thresh=10):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (torch.nn.Module): The fused model is returned.
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        is_fused = sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model
    
        if not is_fused:
            for m in self.model.modules():
                if isinstance(m, (Conv)) and hasattr(m, "bn"):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward

        return self

    @torch.no_grad()
    def predict(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        **kwargs: Any,
    ) -> List[Results]:

        self.args = {**self.args, **self.overrides, **kwargs}

        """Streams real-time inference on camera feed and saves results to file."""
        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.imgsz = check_imgsz(self.args["imgsz"], stride=self.stride, min_dim=2)  # check image size
            if not isinstance(source, (list, tuple)):
                source = [source]

            # Warmup model
            if not self.done_warmup:
                if self.device.type != "cpu":
                    im = torch.empty(1, 3, *self.imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
                    self.forward(im)  # warmup
                self.done_warmup = True

            # Preprocess
            im0s = [ImageDTO(i) for i in source]
            same_shape = len({x.image.shape for x in im0s}) == 1
            im_re = [x.resize(self.imgsz, keep_ratio=True).pad(
                None if same_shape else self.imgsz, stride=32 if same_shape else None
            ) for x in im0s]
            im = torch.cat([i.to_tensor().to(device=self.device, dtype=torch.half if self.fp16 else torch.float) for i in im_re])

            # Inference
            preds = self.forward(im, **kwargs)
            # Postprocess
            preds = ops.non_max_suppression(
                preds,
                self.args["conf"],
                self.args["iou"],
                self.args["classes"],
                self.args["agnostic_nms"],
                max_det=self.args["max_det"],
                nc=len(self.names),
                end2end=getattr(self, "end2end", False),
            )

            results = []
            for pred, orig_img, resized_img in zip(preds, im0s, im_re):
                pred[:, :4] = resized_img.restore_keypoints(pred[:, :4])
                results.append(Results(orig_img.image, names=self.names, boxes=pred[:, :6]))

        return results

    # change name to keep the same with RetinaFace
    def detect_faces(self, image: Union[np.ndarray, str, "Image.Image"], **kwargs) -> PersonAndFaceResult:
        results: Results = self.predict(image, **kwargs)[0]
        return PersonAndFaceResult(results)