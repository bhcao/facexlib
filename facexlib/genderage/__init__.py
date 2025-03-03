from facexlib.utils import load_file_from_url
from .mi_volo import MiVOLO

def init_genderage_model(model_name, half=False, device='cuda', model_rootpath=None):
    if model_name == 'volo_d1':
        model_url = 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/model_utk_age_gender_4.23_97.69.pth.tar'
        use_person = False
    elif model_name == 'mivolo_d1':
        model_url = 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/model_imdb_cross_person_4.22_99.46.pth.tar'
        use_person = True
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(
        url=model_url, model_dir='facexlib/weights', progress=True, file_name=None, save_dir=model_rootpath)

    return MiVOLO(model_path, half=half, device=device, use_persons=use_person)