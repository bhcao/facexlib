import argparse
import cv2
import torch

from facexlib.utils import build_model

def main(args):    
    det_net = build_model(args.detection_model_name, half=args.half)
    genderage_net = build_model(args.genderage_model_name, half=args.half)

    img = cv2.imread(args.img_path)
    with torch.no_grad():
        bboxes = det_net.detect_faces(img)
        ages, genders, genders_scores, age_embeds = genderage_net.predict(img, bboxes, logits=True)
        print(age_embeds.shape, ages, genders, genders_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='assets/test2.jpg')
    parser.add_argument('--detection_model_name', type=str, default='yolov8x_person_face')
    parser.add_argument('--genderage_model_name', type=str, default='mivolo_d1', help='mivolo_d1 | volo_d1')
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()

    main(args)