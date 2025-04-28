from typing import Union
from typing_extensions import deprecated

from facexlib.utils import build_model
from .bisenet import BiSeNet
from .parsenet import ParseNet


@deprecated('`init_parsing_model` is deprecated. Please use `build_model` instead.', category=FutureWarning)
def init_parsing_model(model_name='bisenet', half=False, device=None, model_rootpath=None) -> Union[BiSeNet, ParseNet]:
    assert model_name in ['bisenet', 'parsenet'], f'Please use build_model to initialize other models.'
    return build_model(model_name, half=half, device=device, save_dir=model_rootpath, singleton=False, auto_download=True)
