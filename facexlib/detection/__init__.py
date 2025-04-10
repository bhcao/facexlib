import torch
from copy import deepcopy

from facexlib.utils import load_file_from_url
from .retinaface import RetinaFace
from .yolo_model import YOLODetectionModel
from .insight_retina import InsightRetina


def init_detection_model(model_name, half=False, device=None, model_rootpath=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name == 'retinaface_resnet50':
        model = RetinaFace(network_name='resnet50', half=half, device=device)
    elif model_name == 'retinaface_mobile0.25':
        model = RetinaFace(network_name='mobile0.25', half=half, device=device)
    elif model_name == 'insight_retina':
        model = InsightRetina()
    elif model_name == 'yolov8x_person_face':
        model = YOLODetectionModel(half=half, device=device, names={'person': 0, 'face': 1})
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(model_name, save_dir=model_rootpath)

    # TODO: clean pretrained model
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)

    # Fuse model for faster inference
    if model_name == 'yolov8x_person_face':
        model.fuse()
    return model
