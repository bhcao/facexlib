# Warning: We havn't tested the code on original arcface model

import argparse
import math
import cv2
import numpy as np
import torch

from facexlib.utils import build_model
from facexlib.recognition import calculate_sim


def main(args):
    det_net = build_model(args.det_model_name)
    recog_net = build_model(args.recog_model_name)

    img1 = cv2.imread(args.img_path1)
    img2 = cv2.imread(args.img_path2)

    with torch.no_grad():
        bbox1 = det_net.detect_faces(img1, 0.97)[0]
        bbox2 = det_net.detect_faces(img2, 0.97)[0]

        output = recog_net.get([img1, img2], [bbox1[5:], bbox2[5:]])

    output = output.data.cpu().numpy()

    dist = calculate_sim(output[0], output[1])
    dist = np.arccos(dist) / math.pi * 180

    if dist < 10:
        print(f'Theses two images are almost identical (distance: {dist:.2f} degrees).')
    else:
        print(f'Theses two images are not identical (distance: {dist:.2f} degrees).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path1', type=str, default='assets/test.jpg')
    parser.add_argument('--img_path2', type=str, default='assets/test2.jpg')
    parser.add_argument(
        '--det_model_name', type=str, default='retinaface_resnet50', help='retinaface_resnet50 | retinaface_mobile0.25')
    parser.add_argument('--recog_model_name', type=str, default='antelopev2', help='arcface | antelopev2 | buffalo_l')

    args = parser.parse_args()
    main(args)      