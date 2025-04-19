import argparse
import cv2
import torch

from facexlib.utils import build_model

def main(args):
    # initialize model
    det_net = build_model(args.det_model)
    spoof_net = build_model(args.spoof_model)

    img_fake = cv2.imread(args.img_path_fake)
    img_real = cv2.imread(args.img_path_real)
    with torch.no_grad():
        bbox_fake, bbox_real = det_net.detect_faces([img_fake, img_real], pad_to_same_size=True)
        is_real, score = spoof_net.analyze(img_fake, bbox_fake[0])
        print(f'Fake image is real: {is_real}, Score: {score:.4f}')
        is_real, score = spoof_net.analyze(img_real, bbox_real[0])
        print(f'Real image is real: {is_real}, Score: {score:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path_fake', type=str, default='assets/fake_face.jpg')
    parser.add_argument('--img_path_real', type=str, default='assets/test2.jpg')
    parser.add_argument('--det_model', type=str, default='retinaface_resnet50')
    parser.add_argument('--spoof_model', type=str, default='mini_fasnet')
    args = parser.parse_args()

    main(args)
