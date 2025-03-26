import torch

from facexlib.utils import load_file_from_url
from .arcface_arch import Backbone
from .arcface_official import ArcFace
from .utils import norm_crop, calculate_sim


def init_recognition_model(model_name, half=False, device=None, model_rootpath=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name == 'arcface':
        model = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se')
    elif model_name == 'antelopev2':
        model = ArcFace(layer_num=[3, 13, 30, 3])
    elif model_name == 'buffalo_l':
        model = ArcFace(layer_num=[3, 4, 14, 3])
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(model_name, save_dir=model_rootpath)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    return model
