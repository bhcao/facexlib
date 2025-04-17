

from copy import deepcopy
import torch
import torch.nn as nn

from typing import Tuple, Callable, Literal

import numpy as np
from tqdm import tqdm

from facexlib.resolution.swinir_arch import SwinIR
from facexlib.utils.image_dto import ImageDTO


def sliding_windows(
    h: int, w: int, tile_size: int, tile_stride: int
) -> Tuple[int, int, int, int]:
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)

    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)

    coords = []
    for hi in hi_list:
        for wi in wi_list:
            coords.append((hi, hi + tile_size, wi, wi + tile_size))
    return coords


# https://github.com/csslc/CCSR/blob/main/model/q_sampler.py#L503
def gaussian_weights(tile_width: int, tile_height: int) -> np.ndarray:
    """Generates a gaussian mask of weights for tile contributions"""
    latent_width = tile_width
    latent_height = tile_height
    var = 0.01
    midpoint = (
        latent_width - 1
    ) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [
        np.exp(
            -(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)
        )
        / np.sqrt(2 * np.pi * var)
        for x in range(latent_width)
    ]
    midpoint = latent_height / 2
    y_probs = [
        np.exp(
            -(y - midpoint)
            * (y - midpoint)
            / (latent_height * latent_height)
            / (2 * var)
        )
        / np.sqrt(2 * np.pi * var)
        for y in range(latent_height)
    ]
    weights = np.outer(y_probs, x_probs)
    return weights


def make_tiled_fn(
    fn: Callable[[torch.Tensor], torch.Tensor],
    size: int,
    stride: int,
    scale_type: Literal["up", "down"] = "up",
    scale: int = 1,
    channel: int | None = None,
    weight: Literal["uniform", "gaussian"] = "gaussian",
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    # callback: Callable[[int, int, int, int], None] | None = None,
    progress: bool = True,
) -> Callable[[torch.Tensor], torch.Tensor]:
    # Only split the first input of function.
    def tiled_fn(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if scale_type == "up":
            scale_fn = lambda n: int(n * scale)
        else:
            scale_fn = lambda n: int(n // scale)

        b, c, h, w = x.size()
        out_dtype = dtype or x.dtype
        out_device = device or x.device
        out_channel = channel or c
        out = torch.zeros(
            (b, out_channel, scale_fn(h), scale_fn(w)),
            dtype=out_dtype,
            device=out_device,
        )
        count = torch.zeros_like(out, dtype=torch.float32)
        weight_size = scale_fn(size)
        weights = (
            gaussian_weights(weight_size, weight_size)[None, None]
            if weight == "gaussian"
            else np.ones((1, 1, weight_size, weight_size))
        )
        weights = torch.tensor(
            weights,
            dtype=out_dtype,
            device=out_device,
        )

        indices = sliding_windows(h, w, size, stride)
        pbar = tqdm(
            indices, desc=f"Tiled Processing", disable=not progress, leave=False
        )
        for hi, hi_end, wi, wi_end in pbar:
            x_tile = x[..., hi:hi_end, wi:wi_end]
            out_hi, out_hi_end, out_wi, out_wi_end = map(
                scale_fn, (hi, hi_end, wi, wi_end)
            )
            if len(args) or len(kwargs):
                kwargs.update(dict(hi=hi, hi_end=hi_end, wi=wi, wi_end=wi_end))
            out[..., out_hi:out_hi_end, out_wi:out_wi_end] += (
                fn(x_tile, *args, **kwargs) * weights
            )
            count[..., out_hi:out_hi_end, out_wi:out_wi_end] += weights
        out = out / count
        return out

    return tiled_fn


NETWORK_G_CONFIG = {
    "SwinIR": {
        "img_size": 64,
        "patch_size": 1,
        "in_chans": 3,
        "embed_dim": 180,
        "depths": [6, 6, 6, 6, 6, 6],
        "num_heads": [6, 6, 6, 6, 6, 6],
        "window_size": 8,
        "mlp_ratio": 2,
        "img_range": 1.0,
        "upsampler": "nearest+conv",
        "resi_connection": "1conv",
        "unshuffle": False,
        "unshuffle_scale": 8
    }
}


class SwinIRModel(nn.Module):
    def __init__(self, model_type="SwinIR", upscale=2, device="cuda"):
        super().__init__()
        net_work_config = deepcopy(NETWORK_G_CONFIG[model_type])
        self.model = SwinIR(sf=upscale, **net_work_config)

        self.upscale = upscale
        self.device = device
    
    def inference(self, lq, tile_size=None, tile_pad=None):
        lq = ImageDTO(lq)
        w0, h0 = lq.size
        tiled = tile_size is not None and tile_pad is not None

        if tiled and (w0 < tile_size or h0 < tile_size):
            print("[SwinIR]: the input size is tiny and unnecessary to tile.")
            tiled = False
        if tiled:
            if tile_size % 64 != 0:
                raise ValueError("SwinIR (cleaner) tile size must be a multiple of 64")

        if not tiled:
            # For backward capability, put the resize operation before forward
            lq = lq.pad(center=False, fill=0, stride=64).to_tensor().to(self.device)
            output = self.model(lq)[..., :h0*self.upscale, :w0*self.upscale]
        else:
            tile_stride = tile_size - tile_pad
            tiled_model = make_tiled_fn(
                self.model,
                size=tile_size,
                stride=tile_stride,
                scale=self.upscale,
            )
            output = tiled_model(lq.to_tensor().to(self.device))
        return ImageDTO(output, min_max=(0, 1))