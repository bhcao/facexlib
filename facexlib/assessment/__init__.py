import torch

from facexlib.utils import load_file_from_url
from .hyperiqa_net import HyperIQA


def init_assessment_model(model_name, half=False, device=None, model_rootpath=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name == 'hypernet':
        model = HyperIQA(16, 112, 224, 112, 56, 28, 14, 7)
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    # load the pre-trained hypernet model
    hypernet_model_path = load_file_from_url(model_name, save_dir=model_rootpath)
    model.hypernet.load_state_dict((torch.load(hypernet_model_path, map_location=lambda storage, loc: storage)))
    model = model.eval()
    model = model.to(device)
    return model
