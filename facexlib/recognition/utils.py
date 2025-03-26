'''Ref: https://github.com/deepinsight/insightface'''

import cv2
import numpy as np
from skimage import transform as trans

def calculate_sim(feat1, feat2):
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    return sim

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def norm_crop(img, landmark):
    assert landmark.shape == (5, 2) or landmark.shape == (10,)
    if landmark.shape == (10,):
        landmark = landmark.reshape((5, 2))
    
    tform = trans.SimilarityTransform()
    tform.estimate(landmark, arcface_dst)
    M = tform.params[0:2, :]
    warped = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    return warped