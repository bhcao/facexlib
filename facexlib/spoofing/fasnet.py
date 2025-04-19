# Ref: github.com/serengil/deepface/blob/master/deepface/models/spoofing/FasNetBackbone.py

from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from facexlib.utils.image_dto import ImageDTO
from facexlib.utils.misc import get_root_logger
from .backbone import MiniFASNet, MiniFASNetSE

logger = get_root_logger()

keep_dict = {
    "1.8M": [
        32, 32, 103, 103, 64, 13, 13, 64, 26, 26, 64, 13, 13, 64, 52, 52,
        64, 231, 231, 128, 154, 154, 128, 52, 52, 128, 26, 26, 128, 52, 52, 128,
        26, 26, 128, 26, 26, 128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512,
        512,
    ],
    "1.8M_": [
        32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13, 13, 64, 13, 13,
        64, 231, 231, 128, 231, 231, 128, 52, 52, 128, 26, 26, 128, 77, 77, 128,
        26, 26, 128, 26, 26, 128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 
        512,
    ],
}

class Fasnet(nn.Module):
    """
    Mini Face Anti Spoofing Net Library from repo: github.com/minivision-ai/Silent-Face-Anti-Spoofing
    """

    def __init__(self, embedding_size, conv6_kernel=(7, 7), drop_p=0.2, drop_p_se=0.75, num_classes=3, img_channel=3, device='cpu'):
        super(Fasnet, self).__init__()

        self.device = device

        # Fasnet will use 2 distinct models to predict, then it will find the sum of predictions
        # to make a final prediction

        self.mini_fasnet = MiniFASNet(
            keep_dict["1.8M_"], embedding_size, conv6_kernel, drop_p, num_classes, img_channel
        )

        self.mini_fasnet_se = MiniFASNetSE(
            keep_dict["1.8M"], embedding_size, conv6_kernel, drop_p_se, num_classes, img_channel
        )


    @torch.no_grad()
    def analyze(self, img: np.ndarray, bbox: Union[list, np.ndarray]):
        """
        Analyze a given image spoofed or not

        Args:
            img (np.ndarray): pre loaded image
            bbox (list or np.ndarray): bounding box (x1, y1, x2, y2) of the face in the image

        Returns:
            result (tuple): a result tuple consisting of is_real and score
        """

        first_img = crop(img, np.array(bbox)[:4], 2.7, 80, 80)
        second_img = crop(img, np.array(bbox)[:4], 4, 80, 80)

        first_img = ImageDTO(first_img).to_tensor(rgb2bgr=True, mean=0.0, std=1.0).to(self.device)
        second_img = ImageDTO(second_img).to_tensor(rgb2bgr=True, mean=0.0, std=1.0).to(self.device)

        first_result = self.mini_fasnet.forward(first_img)
        first_result = F.softmax(first_result).cpu().numpy()

        second_result = self.mini_fasnet_se.forward(second_img)
        second_result = F.softmax(second_result).cpu().numpy()

        prediction = np.zeros((1, 3))
        prediction += first_result
        prediction += second_result

        label = np.argmax(prediction)
        is_real = label == 1
        score = prediction[0][label] / 2

        return is_real, score


# subsdiary classes and functions

def _get_new_box(src_w, src_h, bbox, scale):
    x = bbox[0]
    y = bbox[1]
    box_w = bbox[2] - x
    box_h = bbox[3] - y
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y
    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2
    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0
    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0
    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1
    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1
    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)


def crop(org_img, bbox, scale, out_w, out_h):
    src_h, src_w, _ = np.shape(org_img)
    left_top_x, left_top_y, right_bottom_x, right_bottom_y = _get_new_box(src_w, src_h, bbox, scale)
    img = org_img[left_top_y : right_bottom_y + 1, left_top_x : right_bottom_x + 1]
    dst_img = cv2.resize(img, (out_w, out_h))
    return dst_img