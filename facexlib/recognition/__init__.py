from typing_extensions import deprecated

from facexlib.utils import build_model
from .arcface_arch import Backbone
from .utils import calculate_sim


@deprecated('`init_recognition_model` is deprecated. Please use `build_model` instead.', category=FutureWarning)
def init_recognition_model(model_name, half=False, device=None, model_rootpath=None) -> Backbone:
    assert model_name in ['arcface'], f'Please use build_model to initialize other models.'
    return build_model(model_name, half=half, device=device, save_dir=model_rootpath, singleton=False, auto_download=True)
