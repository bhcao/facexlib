'''Ref: https://github.com/deepinsight/insightface'''

import numpy as np

def calculate_sim(feat1, feat2):
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    return sim

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

mtlface_dst = np.array(
    [[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
     [33.5493, 92.3655], [62.7299, 92.2041]],
    dtype=np.float32)
