import os
from torch.hub import download_url_to_file
from urllib.parse import urlparse

DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights')

MODEL_ZOO = {
    'awing_fan': 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth',
    'hypernet': 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/assessment_hyperIQA.pth',
    'retinaface_resnet50': 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth',
    'retinaface_mobile0.25': 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_mobilenet0.25_Final.pth',
    'yolov8x_person_face': 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/yolov8x_person_face.pt',
    'volo_d1': 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/model_utk_age_gender_4.23_97.69.pth.tar',
    'mivolo_d1': 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/model_imdb_cross_person_4.22_99.46.pth.tar',
    'hopenet': 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/headpose_hopenet.pth',
    'modnet': 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/matting_modnet_portrait.pth',
    'bisenet': 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth',
    'parsenet': 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth',
    'arcface': 'https://github.com/xinntao/facexlib/releases/download/v0.1.0/recognition_arcface_ir_se50.pth',
    'antelopev2': 'https://github.com/bhcao/facexlib/releases/download/v0.3.2/antelopev2.pth',
    'buffalo_l': 'https://github.com/bhcao/facexlib/releases/download/v0.3.2/buffalo_l.pth',
    'HAT-S_x2': 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/HAT-S_SRx2.pth',
    'HAT-S_x4': 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/HAT-S_SRx4.pth',
    'HAT_x2': 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/HAT_SRx2_ImageNet-pretrain.pth',
    'HAT_x4': 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/HAT_SRx4_ImageNet-pretrain.pth',
    'HAT-L_x2': 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/HAT-L_SRx2_ImageNet-pretrain.pth',
    'HAT-L_x4': 'https://github.com/bhcao/facexlib/releases/download/v0.3.1/HAT-L_SRx4_ImageNet-pretrain.pth',
}

def load_file_from_url(model_name_or_url, progress=True, file_name=None, save_dir=None):
    """Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Set os.environ['BLOCK_DOWNLOADING'] == 'false' to cancel block downloading.
    """

    if model_name_or_url.startswith('http://') or model_name_or_url.startswith('https://'):
        url = model_name_or_url
    else:
        url = MODEL_ZOO[model_name_or_url]

    if save_dir is None:
        save_dir = DEFAULT_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(save_dir, filename))
    if not os.path.exists(cached_file):
        block_downloading = os.environ.get('BLOCK_DOWNLOADING', 'true')
        if block_downloading.lower() == 'true':
            raise RuntimeError(f'{cached_file} does not exist. Please download it manually from {url}')
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file