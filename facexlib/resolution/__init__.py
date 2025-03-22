from .hat_model import HATModel

from facexlib.utils import load_file_from_url

def init_resolution_model(model_name, scale, tile_size=None, tile_pad=None, half=False, device='cuda', model_rootpath=None):
    '''Use tile_size = 512 and tile_pad = 32 when OOM.'''
    if scale not in [2, 4]:
        raise ValueError(f'Unsupported scale {scale}.')
    
    if model_name == 'HAT-S':
        if scale == 2:
            model_url = 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/HAT_SRx2.pth'
        else:
            model_url = 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/HAT_SRx4.pth'
    elif model_name == 'HAT':
        if scale == 2:
            model_url = 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/HAT_SRx2_ImageNet-pretrain.pth'
        else:
            model_url = 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/HAT_SRx4_ImageNet-pretrain.pth'
    elif model_name == 'HAT-L':
        if scale == 2:
            model_url = 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/HAT-L_SRx2_ImageNet-pretrain.pth'
        else:
            model_url = 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/HAT-L_SRx4_ImageNet-pretrain.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)

    return HATModel(model_name, scale = scale, tile_size = tile_size, tile_pad = tile_pad,
                    load_path = model_path,
                    strict_load = True,
                    param_key = 'params_ema',
                    device = device, half = half)