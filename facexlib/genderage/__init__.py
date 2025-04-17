import torch
from facexlib.utils import load_file_from_url
from .mi_volo import MiVOLO

# TODO: remove this function and use build_model instead
def init_genderage_model(model_name, half=False, device=None, model_rootpath=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name == 'volo_d1':
        use_person = False
        url = 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/model_utk_age_gender_4.23_97.69.pth.tar'
    elif model_name == 'mivolo_d1':
        use_person = True
        url = 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/model_imdb_cross_person_4.22_99.46.pth.tar'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(url, model_dir='facexlib/weights', save_dir=model_rootpath)

    return MiVOLO(model_path, half=half, device=device, use_persons=use_person)