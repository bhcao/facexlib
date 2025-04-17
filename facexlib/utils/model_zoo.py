import importlib
import inspect
import os
from pathlib import Path
import torch
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import yaml

DEFAULT_SAVE_DIR = Path(__file__).absolute().parent.parent / 'weights'
MODEL_ZOO = yaml.load(open(Path(__file__).absolute().parent / 'model_zoo.yaml', 'r'), Loader=yaml.FullLoader)

__cached_models = {}

def build_model(model_name, progress=True, file_name=None, save_dir=None, half=False, device=None, singleton=True, auto_download=False):
    """Build a model from a given url.

    Args:
        model_name (str): The name of the model in the model zoo.
        progress (bool, optional): Whether to show the progress of downloading. Defaults to True.
        file_name (str, optional): The name of the downloaded file. Defaults to the same as the url.
        save_dir (str, optional): The directory to save the downloaded file. Defaults to `facexlib/weights`.
        half (bool, optional): Whether to use half precision. Only works for models support half initialization.
        device (str, optional): The device to load the model. Defaults to 'cuda' if available, otherwise 'cpu'.
        singleton (bool, optional): Whether to return a singleton model or a list of models. Defaults to True.
        auto_download (bool, optional): Whether to automatically download the model if it does not exist.

    Returns:
        The built model.
    """
    if singleton and hasattr(__cached_models, model_name):
        return __cached_models[model_name]

    if model_name not in MODEL_ZOO:
        raise NotImplementedError(f'{model_name} is not implemented.')

    # get the model information from the model zoo
    model_dict = MODEL_ZOO[model_name]
    target = model_dict['target']
    url = model_dict['url']
    param_key = model_dict.get('param_key', None)
    param_prefix = model_dict.get('param_prefix', None)
    load_module = model_dict.get('load_module', None)
    args = model_dict.get('args', [])
    kwargs = model_dict.get('kwargs', {})

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the model class
    package_name, class_name = target.rsplit('.', 1)
    target_class = getattr(importlib.import_module(package_name), class_name)

    # create the save directory if not exists
    save_dir = Path(DEFAULT_SAVE_DIR if save_dir is None else save_dir)
    save_dir.mkdir(exist_ok=True)

    # download the model weights
    filename = os.path.basename(urlparse(url).path) if file_name is None else file_name
    cached_file = save_dir.absolute() / filename
    if not cached_file.exists():
        if not auto_download:
            raise RuntimeError(f'{cached_file} does not exist. Please download it manually from {url}')
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)

    # load the model parameters
    if param_key is not None:
        state_dict = torch.load(cached_file, weights_only=True)[param_key]
    else:
        state_dict = torch.load(cached_file, weights_only=True)
    if param_prefix is not None:
        state_dict = {k[len(param_prefix):]: v for k, v in state_dict.items() if k.startswith(param_prefix)}
    
    # create the model instance
    init_params = inspect.signature(target_class.__init__).parameters.keys()
    if 'device' in init_params:
        kwargs['device'] = device
    if 'half' in init_params:
        kwargs['half'] = half
    model = target_class(*args, **kwargs)
    if load_module is not None:
        getattr(model, load_module).load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    # fuse conv and bn
    if hasattr(model, 'fuse'):
        model.fuse()

    if singleton:
        __cached_models[model_name] = model
    return model
