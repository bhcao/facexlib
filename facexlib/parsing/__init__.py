import torch

from facexlib.utils import load_file_from_url
from .bisenet import BiSeNet
from .parsenet import ParseNet


def init_parsing_model(model_name='bisenet', half=False, device=None, model_rootpath=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name == 'bisenet':
        model = BiSeNet(num_class=19)
    elif model_name == 'parsenet':
        model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(model_name, save_dir=model_rootpath)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model
