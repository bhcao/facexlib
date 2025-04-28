from typing_extensions import deprecated

from facexlib.utils import build_model
from .modnet import MODNet


@deprecated('`init_matting_model` is deprecated. Please use `build_model` instead.', category=FutureWarning)
def init_matting_model(model_name='modnet', half=False, device=None, model_rootpath=None) -> MODNet:
    assert model_name in ['modnet'], f'Please use build_model to initialize other models.'
    return build_model(model_name, half=half, device=device, save_dir=model_rootpath, singleton=False, auto_download=True)
