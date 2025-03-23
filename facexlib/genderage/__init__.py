import torch
from facexlib.utils import load_file_from_url
from .mi_volo import MiVOLO

def init_genderage_model(model_name, half=False, device=None, model_rootpath=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name == 'volo_d1':
        use_person = False
    elif model_name == 'mivolo_d1':
        use_person = True
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(model_name, save_dir=model_rootpath)

    return MiVOLO(model_path, half=half, device=device, use_persons=use_person)