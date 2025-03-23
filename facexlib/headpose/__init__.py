import torch

from facexlib.utils import load_file_from_url
from .hopenet_arch import HopeNet


def init_headpose_model(model_name, half=False, device=None, model_rootpath=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_name == 'hopenet':
        model = HopeNet('resnet', [3, 4, 6, 3], 66)
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(model_name, save_dir=model_rootpath)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)['params']
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model
