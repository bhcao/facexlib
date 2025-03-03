import argparse
import cv2
import torch

from facexlib.genderage import init_genderage_model
from facexlib.detection import init_detection_model

def main(args):    
    det_net = init_detection_model(args.detection_model_name, "cuda", half=args.half)
    genderage_net = init_genderage_model(args.genderage_model_name, "cuda", half=args.half)

    img = cv2.imread(args.img_path)
    with torch.no_grad():
        detected_objects = det_net.detect_faces(img)
        genderage_net.predict(img, detected_objects)
        print(detected_objects.ages, detected_objects.genders)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='assets/test.jpg')
    parser.add_argument('--detection_model_name', type=str, default='yolov8x_person_face')
    parser.add_argument('--genderage_model_name', type=str, default='mivolo_d1', help='mivolo_d1 | volo_d1')
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()

    main(args)