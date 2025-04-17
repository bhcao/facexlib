import cv2
import logging
import os
import numpy as np
import torch
import warnings
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from facexlib.utils.image_dto import ImageDTO

initialized_logger = {}

# modified from basicsr.utils
def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.setLevel(log_level)
    if log_file is not None:
        # add file handler
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    warnings.warn("imwrite is deprecated, use ImageDTO.save instead", FutureWarning)

    return ImageDTO(img).save(file_path, params, auto_mkdir)


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """
    warnings.warn("img2tensor is deprecated, use ImageDTO.to_tensor instead", FutureWarning)

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def load_file_from_url(url, model_dir=None, progress=True, file_name=None, save_dir=None):
    """Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
    """
    warnings.warn("load_file_from_url is deprecated, use build_model instead", FutureWarning)

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    if save_dir is None:
        save_dir = os.path.join(ROOT_DIR, model_dir)
    os.makedirs(save_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(save_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def box_iou(box1, box2, over_second=False):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    If over_second == True, return mean(intersection-over-union, (inter / area2))

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format. The last element will
    be seen as score and will be ignored.

    Args:
        box1 (Tensor[N | None, 4] | list)
        box2 (Tensor[M | None, 4] | list)

    Returns:
        iou (Tensor[N | None, M | None]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # in case they are lists
    box1 = np.array(box1)
    box2 = np.array(box2)

    box1_ = box1[None, :] if len(box1.shape) == 1 else box1
    box2_ = box2[None, :] if len(box2.shape) == 1 else box2

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1_.T)
    area2 = box_area(box2_.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2) # ignore score
    box_sizes = np.minimum(box1_[:, None, 2:4], box2_[:, 2:4]) - np.maximum(box1_[:, None, :2], box2_[:, :2])
    inter = np.prod(np.clip(box_sizes, a_min=0, a_max=None), axis=2)

    iou = inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
    if over_second:
        result = (inter / area2 + iou) / 2  # mean(inter / area2, iou)
    else:
        result = iou
    
    if len(box2.shape) == 1:
        result = result.squeeze(1)
    if len(box1.shape) == 1:
        result = result.squeeze(0)

    return result