from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from scipy.optimize import linear_sum_assignment
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from facexlib.utils.misc import box_iou

CROP_ROUND_RATE = 0.1
MIN_PERSON_CROP_NONZERO = 0.5


def aggregate_votes_winsorized(ages, max_age_dist=6):
    # Replace any annotation that is more than a max_age_dist away from the median
    # with the median + max_age_dist if higher or max_age_dist - max_age_dist if below
    median = np.median(ages)
    ages = np.clip(ages, median - max_age_dist, median + max_age_dist)
    return np.mean(ages)

def assign_faces(
    persons_bboxes: List[torch.tensor], faces_bboxes: List[torch.tensor], iou_thresh: float = 0.0001
) -> Tuple[List[Optional[int]], List[int]]:
    """
    Assign person to each face if it is possible.
    Return:
        - assigned_faces List[Optional[int]]: mapping of face_ind to person_ind
                                            ( assigned_faces[face_ind] = person_ind ). person_ind can be None
        - unassigned_persons_inds List[int]: persons indexes without any assigned face
    """

    assigned_faces: List[Optional[int]] = [None for _ in range(len(faces_bboxes))]
    unassigned_persons_inds: List[int] = [p_ind for p_ind in range(len(persons_bboxes))]

    if len(persons_bboxes) == 0 or len(faces_bboxes) == 0:
        return assigned_faces, unassigned_persons_inds

    cost_matrix = box_iou(torch.stack(persons_bboxes).cpu().numpy(), torch.stack(faces_bboxes).cpu().numpy(), over_second=True)
    persons_indexes, face_indexes = [], []

    if len(cost_matrix) > 0:
        persons_indexes, face_indexes = linear_sum_assignment(cost_matrix, maximize=True)

    matched_persons = set()
    for person_idx, face_idx in zip(persons_indexes, face_indexes):
        ciou = cost_matrix[person_idx][face_idx]
        if ciou > iou_thresh:
            if person_idx in matched_persons:
                # Person can not be assigned twice, in reality this should not happen
                continue
            assigned_faces[face_idx] = person_idx
            matched_persons.add(person_idx)

    unassigned_persons_inds = [p_ind for p_ind in range(len(persons_bboxes)) if p_ind not in matched_persons]

    return assigned_faces, unassigned_persons_inds


def class_letterbox(im, new_shape=(640, 640), color=(0, 0, 0), scaleup=True):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    if im.shape[0] == new_shape[0] and im.shape[1] == new_shape[1]:
        return im

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    # ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im


def prepare_classification_images(
    img_list: List[Optional[np.ndarray]],
    target_size: int = 224,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    device=None,
) -> torch.tensor:

    prepared_images: List[torch.tensor] = []

    for img in img_list:
        if img is None:
            img = torch.zeros((3, target_size, target_size), dtype=torch.float32)
            img = F.normalize(img, mean=mean, std=std)
            img = img.unsqueeze(0)
            prepared_images.append(img)
            continue
        img = class_letterbox(img, new_shape=(target_size, target_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img / 255.0
        img = (img - mean) / std
        img = img.astype(dtype=np.float32)

        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)

        prepared_images.append(img)

    if len(prepared_images) == 0:
        return None

    prepared_input = torch.concat(prepared_images)

    if device:
        prepared_input = prepared_input.to(device)

    return prepared_input
