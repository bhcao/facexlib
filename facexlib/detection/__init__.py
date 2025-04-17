from typing import Union
import warnings

from facexlib.utils import build_model
from .retinaface import RetinaFace
from .yolo_model import YOLODetectionModel
from .insight_retina import InsightRetina


def init_detection_model(model_name, half=False, device=None, model_rootpath=None) -> Union[RetinaFace, YOLODetectionModel, InsightRetina]:
    warnings.warn('init_assessment_model is deprecated, use build_model instead.', FutureWarning)
    assert model_name in ['retinaface_resnet50', 'retinaface_mobile0.25', 'yolov8x_person_face', 'insight_retina'], f'Please use build_model to initialize other models.'
    return build_model(model_name, half=half, device=device, save_dir=model_rootpath, singleton=False, auto_download=True)
