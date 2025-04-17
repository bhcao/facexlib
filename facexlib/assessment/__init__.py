import warnings
from facexlib.utils import build_model
from .hyperiqa_net import HyperIQA


def init_assessment_model(model_name, half=False, device=None, model_rootpath=None) -> HyperIQA:
    warnings.warn('init_assessment_model is deprecated, use build_model instead.', FutureWarning)
    assert model_name in ['hypernet'], f'Please use build_model to initialize other models.'
    return build_model(model_name, half=half, device=device, save_dir=model_rootpath, singleton=False, auto_download=True)
