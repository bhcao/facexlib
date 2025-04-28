"""
Facial emotions recognition implementation
"""

from typing import List, Tuple

import numpy as np

import torch
import torch.nn as nn
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from facexlib.utils import ImageDTO

class EmotiEffLibRecognizer(nn.Module):
    """
    Abstract class for emotion recognizer classes
    """

    def __init__(self, network_name: str, img_size: int, device = "cpu") -> None:
        super().__init__()
        self.idx_to_emotion_class = {
            0: "Anger",
            1: "Contempt",
            2: "Disgust",
            3: "Fear",
            4: "Happiness",
            5: "Neutral",
            6: "Sadness",
            7: "Surprise",
        }

        self.mean = IMAGENET_DEFAULT_MEAN
        self.std = IMAGENET_DEFAULT_STD
        self.img_size = img_size
        self.device = device
        self.model = timm.create_model(network_name, num_classes=8) # features_only=True


    def build_model_post_hook(self):
        self.classifier_weights = self.model.classifier.weight.cpu().data.numpy()
        self.classifier_bias = self.model.classifier.bias.cpu().data.numpy()
        self.model.classifier = torch.nn.Identity()


    def _get_probab(self, features: np.ndarray) -> np.ndarray:
        """
        Compute the final classification scores for the given feature representations.

        Args:
            features (np.ndarray): The extracted feature vectors.

        Returns:
            np.ndarray: The raw classification scores (logits) before applying any activation
                        function.
        """
        x = np.dot(features, np.transpose(self.classifier_weights)) + self.classifier_bias
        return x


    def classify_emotions(
        self, features: np.ndarray, logits: bool = True
    ) -> Tuple[List[str], np.ndarray]:
        """
        Classify emotions based on extracted feature representations.

        Args:
            features (np.ndarray): The extracted feature vectors.
            logits (bool, optional):
                If True, returns raw model scores (logits). If False, applies softmax normalization
                to obtain probability distributions. Defaults to True.

        Returns:
            Tuple[List[str], np.ndarray]:
                - A list of predicted emotion labels.
                - The corresponding model output scores (logits or probabilities), as a NumPy array.
        """
        x = self._get_probab(features)
        preds = np.argmax(x, axis=1)

        if not logits:
            e_x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
            e_x = e_x / e_x.sum(axis=1)[:, None]
            scores = e_x

        return [self.idx_to_emotion_class[pred] for pred in preds], scores


    def predict_emotions(self, face_img: List[ImageDTO], logits: bool = False) -> Tuple[List[str], np.ndarray]:
        """
        Predict the emotions presented on a given facial image or a list of facial images.

        Args:
            face_img (List[ImageDTO]): A list of face images.
            logits (bool, optional):
                If True, returns raw model scores (logits). If False, applies softmax normalization
                to obtain probability distributions. Defaults to False.

        Returns:
            Tuple[Union[str, List[str]], np.ndarray]:
                - The predicted emotion label(s) as a list of strings (for single image only with
                  one element).
                - The corresponding model output scores (logits or probabilities), as a NumPy array.
        """

        img_tensor = torch.concat([ImageDTO(img).resize(self.img_size).to_tensor(
            rgb2bgr=True, mean=self.mean, std=self.std, timm_form=True
        ) for img in face_img]).to(self.device)
        features = self.model(img_tensor)
        features = features.data.cpu().numpy()

        return self.classify_emotions(features, logits)


    def predict(self, face_img, bboxes: List[np.ndarray], logits: bool = False):
        face_img = ImageDTO(face_img)

        face_imgs = []
        for bbox in bboxes:
            face_img_cropped = face_img.crop_align(bbox)
            face_imgs.append(face_img_cropped)

        return self.predict_emotions(face_imgs, logits)
