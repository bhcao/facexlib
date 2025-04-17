import warnings

from facexlib.utils import build_model
from .hopenet_arch import HopeNet


def init_headpose_model(model_name, half=False, device=None, model_rootpath=None) -> HopeNet:
    warnings.warn('init_headpose_model is deprecated, use build_model instead.', FutureWarning)
    assert model_name in ['hopenet'], f'Please use build_model to initialize other models.'
    return build_model(model_name, half=half, device=device, save_dir=model_rootpath, singleton=False, auto_download=True)
