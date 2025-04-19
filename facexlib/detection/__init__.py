import warnings

from facexlib.utils import build_model
from .retinaface import RetinaFace


def init_detection_model(model_name, half=False, device=None, model_rootpath=None) -> RetinaFace:
    warnings.warn('init_assessment_model is deprecated, use build_model instead.', FutureWarning)
    assert model_name in ['retinaface_resnet50', 'retinaface_mobile0.25'], f'Please use build_model to initialize other models.'
    return build_model(model_name, half=half, device=device, save_dir=model_rootpath, singleton=False, auto_download=True)
