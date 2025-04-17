from copy import deepcopy
import torch
from torch.nn import functional as F
import numpy as np
import math
from PIL import Image

from facexlib.resolution.hat_arch import HAT
from facexlib.resolution.sr_model import SRModel
from facexlib.utils.image_dto import ImageDTO

NETWORK_G_CONFIG = {
    'HAT-L': {
        "in_chans": 3,
        "img_size": 64,
        "window_size": 16,
        "compress_ratio": 3,
        "squeeze_factor": 30,
        "conv_scale": 0.01,
        "overlap_ratio": 0.5,
        "img_range": 1.,
        "depths": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        "mlp_ratio": 2,
        "upsampler": 'pixelshuffle',
        "resi_connection": '1conv'
    },
    'HAT': {
        "in_chans": 3,
        "img_size": 64,
        "window_size": 16,
        "compress_ratio": 3,
        "squeeze_factor": 30,
        "conv_scale": 0.01,
        "overlap_ratio": 0.5,
        "img_range": 1.,
        "depths": [6, 6, 6, 6, 6, 6],
        "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6],
        "mlp_ratio": 2,
        "upsampler": 'pixelshuffle',
        "resi_connection": '1conv'
    },
}

class HATModel(SRModel):
    def __init__(self, model_type, scale=1, device='cuda', half=False, **kwargs):
        net_work_config = deepcopy(NETWORK_G_CONFIG[model_type])
        net_g = HAT(upscale=scale, **net_work_config)
        super(HATModel, self).__init__(net_g, device=device, half=half, **kwargs)

        # store for later use
        self.window_size = net_work_config['window_size']
        self.scale = scale

    def pre_process(self):
        # pad to multiplication of window_size
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % self.window_size != 0:
            self.mod_pad_h = self.window_size - h % self.window_size
        if w % self.window_size != 0:
            self.mod_pad_w = self.window_size - w % self.window_size
        self.img = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        # model inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.img)
            # self.net_g.train()

    def tile_process(self, tile_size, tile_pad):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    if hasattr(self, 'net_g_ema'):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            output_tile = self.net_g_ema(input_tile)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            output_tile = self.net_g(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]

    def inference(self, image: np.ndarray | Image.Image, tile_size=None, tile_pad=None):
        self.lq = ImageDTO(image).to_tensor().to(device=self.device, dtype=self.dtype)

        self.pre_process()
        if tile_size is not None and tile_pad is not None:
            self.tile_process(tile_size, tile_pad)
        else:
            self.process()
        self.post_process()

        visuals = self.get_current_visuals()

        # tentative for out of GPU memory
        del self.lq
        del self.output
        torch.cuda.empty_cache()

        return ImageDTO(visuals['result'], min_max=(0, 1))