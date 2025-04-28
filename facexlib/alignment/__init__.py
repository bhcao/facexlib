from typing_extensions import deprecated

from facexlib.utils import build_model
from .awing_arch import FAN
from .convert_98_to_68_landmarks import landmark_98_to_68

__all__ = ['FAN', 'landmark_98_to_68']


@deprecated("`init_alignment_model` is deprecated. Please use `build_model` instead.", category=FutureWarning)
def init_alignment_model(model_name, half=False, device=None, model_rootpath=None) -> FAN:
    assert model_name in ['awing_fan'], f'Please use build_model to initialize other models.'
    return build_model(model_name, half=half, device=device, save_dir=model_rootpath, singleton=False, auto_download=True)
