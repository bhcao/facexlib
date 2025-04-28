import argparse
import cv2
import torch

from facexlib.utils import build_model

def main(args):
    det_net = build_model(args.det_model_name)
    exp_net = build_model(args.exp_model_name)

    img = cv2.imread(args.img_path)

    with torch.no_grad():
        bbox = det_net.detect_faces(img, 0.97)
        emotions = exp_net.predict(img, bbox)
        print(f'Emotions: {emotions}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='assets/test2.jpg')
    parser.add_argument(
        '--det_model_name', type=str, default='retinaface_resnet50', help='retinaface_resnet50 | retinaface_mobile0.25')
    parser.add_argument('--exp_model_name', type=str, default='emotiefflib_enet_b2', help='emotiefflib_enet_b2')

    args = parser.parse_args()
    main(args)