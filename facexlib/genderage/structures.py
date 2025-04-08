import os
from typing import List, Optional, Tuple

import torch
import numpy as np
from facexlib.utils.misc import box_iou

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from facexlib.utils.image_dto import ImageDTO

# because of ultralytics bug it is important to unset CUBLAS_WORKSPACE_CONFIG after the module importing
os.unsetenv("CUBLAS_WORKSPACE_CONFIG")

AGE_GENDER_TYPE = Tuple[float, str]

class PersonAndFaceResult:
    def __init__(self, results: np.ndarray):
        if len(results) > 0:
            assert results.shape[1] == 10, "Results should have both persons and faces"

        self.yolo_results = results

        n_objects = len(self.yolo_results)
        self.ages: List[Optional[float]] = [None for _ in range(n_objects)]
        self.genders: List[Optional[str]] = [None for _ in range(n_objects)]
        self.gender_scores: List[Optional[float]] = [None for _ in range(n_objects)]

    @property
    def n_objects(self) -> int:
        return self.n_faces + self.n_persons

    @property
    def n_faces(self) -> int:
        return len(self.yolo_results) - np.sum(np.isnan(self.yolo_results[:, 0]))

    @property
    def n_persons(self) -> int:
        return len(self.yolo_results) - np.sum(np.isnan(self.yolo_results[:, 5]))

    def set_age(self, ind: Optional[int], age: float):
        if ind is not None:
            self.ages[ind] = age

    def set_gender(self, ind: Optional[int], gender: str, gender_score: float):
        if ind is not None:
            self.genders[ind] = gender
            self.gender_scores[ind] = gender_score

    def crop_object(
        self, full_image: np.ndarray, ind: int, obj_bbox: np.ndarray, is_person: bool = False
    ) -> Optional[np.ndarray]:

        IOU_THRESH = 0.000001
        MIN_PERSON_CROP_AFTERCUT_RATIO = 0.4
        CROP_ROUND_RATE = 0.3
        MIN_PERSON_SIZE = 50

        x1, y1, x2, y2 = obj_bbox.astype(int)
        # get crop of face or person
        obj_image = full_image[y1:y2, x1:x2].copy()
        crop_h, crop_w = obj_image.shape[:2]

        if is_person and (crop_h < MIN_PERSON_SIZE or crop_w < MIN_PERSON_SIZE):
            return None

        if not is_person:
            return obj_image

        # calc iou between obj_bbox and other bboxes
        iou_matrix = box_iou(obj_bbox, self.yolo_results[:, :4])

        # cut out other faces in case of intersection
        for other_ind, (det, iou) in enumerate(zip(self.yolo_results[:, :4], iou_matrix)):
            if iou < IOU_THRESH or np.isnan(det[0]):
                continue
            o_x1, o_y1, o_x2, o_y2 = det.astype(int)

            # remap current_person_bbox to reference_person_bbox coordinates
            o_x1 = max(o_x1 - x1, 0)
            o_y1 = max(o_y1 - y1, 0)
            o_x2 = min(o_x2 - x1, crop_w)
            o_y2 = min(o_y2 - y1, crop_h)

            obj_image[o_y1:o_y2, o_x1:o_x2] = 0


        iou_matrix = box_iou(obj_bbox, self.yolo_results[:, 5:9])

        # cut out other persons in case of intersection
        for other_ind, (det, iou) in enumerate(zip(self.yolo_results[:, 5:9], iou_matrix)):
            if ind == other_ind or iou < IOU_THRESH:
                continue
            o_x1, o_y1, o_x2, o_y2 = det.astype(int)

            # remap current_person_bbox to reference_person_bbox coordinates
            o_x1 = max(o_x1 - x1, 0)
            o_y1 = max(o_y1 - y1, 0)
            o_x2 = min(o_x2 - x1, crop_w)
            o_y2 = min(o_y2 - y1, crop_h)

            if (o_y1 / crop_h) < CROP_ROUND_RATE:
                o_y1 = 0
            if ((crop_h - o_y2) / crop_h) < CROP_ROUND_RATE:
                o_y2 = crop_h
            if (o_x1 / crop_w) < CROP_ROUND_RATE:
                o_x1 = 0
            if ((crop_w - o_x2) / crop_w) < CROP_ROUND_RATE:
                o_x2 = crop_w

            obj_image[o_y1:o_y2, o_x1:o_x2] = 0

        remain_ratio = np.count_nonzero(obj_image) / (obj_image.shape[0] * obj_image.shape[1] * obj_image.shape[2])
        if remain_ratio < MIN_PERSON_CROP_AFTERCUT_RATIO:
            return None

        return obj_image

    def collect_crops(self, image, device, use_persons: bool = True, use_faces: bool = True,
            target_size: int = 224,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD):
        """
        Return
            faces: faces_with_bodies, faces_without_bodies, [None] * len(crops_persons_wo_face)
            persons: bodies_with_faces, [None] * len(faces_without_bodies), bodies_without_faces
        """

        zero_img = np.zeros((1, 1, 3), dtype=np.uint8)
        assert use_persons or use_faces, "Must specify at least one of use_persons and use_faces"

        faces_crops = []
        bodies_crops = []
        bodies_inds = []
        faces_inds = []

        for ind, bbox_2 in enumerate(self.yolo_results):
            face_ind = None
            person_ind = None
            face_image = zero_img
            person_image = zero_img

            if not np.isnan(bbox_2[0]) and use_faces:
                face_image = self.crop_object(image, ind, bbox_2[:4])
                face_ind = ind
            if not np.isnan(bbox_2[5]) and use_persons:
                person_image = self.crop_object(image, ind, bbox_2[5:9], is_person=True)
                person_ind = ind

            if face_image is not None or person_image is not None:
                faces_crops.append(ImageDTO(face_image).resize(target_size, keep_ratio=True).pad(target_size, fill=0)
                                   .to_tensor(mean=mean, std=std, timm_form=True))
                bodies_crops.append(ImageDTO(person_image).resize(target_size, keep_ratio=True).pad(target_size, fill=0)
                                    .to_tensor(mean=mean, std=std, timm_form=True))
                bodies_inds.append(person_ind)
                faces_inds.append(face_ind)

        return (bodies_inds, torch.concat(bodies_crops).to(device)), (faces_inds, torch.concat(faces_crops).to(device))
