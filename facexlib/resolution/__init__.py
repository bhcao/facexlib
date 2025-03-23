import torch
from .hat_model import HATModel

from facexlib.utils import load_file_from_url

def init_resolution_model(model_name, scale, tile_size=None, tile_pad=None, half=False, device=None, model_rootpath=None):
    '''Use tile_size = 512 and tile_pad = 32 when OOM.'''
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name not in ['HAT-S', 'HAT', 'HAT-L']:
        raise ValueError(f'{model_name} is not implemented.')

    if scale not in [2, 4]:
        raise ValueError(f'Unsupported scale {scale}.')

    model_path = load_file_from_url(f'{model_name}_x{scale}', save_dir=model_rootpath)

    return HATModel(model_name, scale = scale, tile_size = tile_size, tile_pad = tile_pad,
                    load_path = model_path,
                    strict_load = True,
                    param_key = 'params_ema',
                    device = device, half = half)