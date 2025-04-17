import argparse
import cv2
import numpy as np
import os
import torch

from facexlib.utils import build_model

# Colors for all parts
part_colors_parsenet = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255],
               [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204],
               [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
#     0: 'background' 1: 'skin'   2: 'nose'
#     3: 'eye_g'  4: 'l_eye'  5: 'r_eye'
#     6: 'l_brow' 7: 'r_brow' 8: 'l_ear'
#     9: 'r_ear'  10: 'mouth' 11: 'u_lip'
#     12: 'l_lip' 13: 'hair'  14: 'hat'
#     15: 'ear_r' 16: 'neck_l'    17: 'neck'
#     18: 'cloth'

# Colors for all 20 parts
part_colors_bisenet = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0],
               [170, 255, 0], [0, 255, 85], [0, 255, 170], [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255],
               [0, 170, 255], [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255],
               [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]
# 0: 'background'
# attributions = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye',
#                 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose',
#                 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l',
#                 16 'cloth', 17 'hair', 18 'hat']


def vis_parsing_maps(img, parsing_anno, stride, save_anno_path=None, save_vis_path=None, model_name='bisenet'):

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    if save_anno_path is not None:
        cv2.imwrite(save_anno_path, vis_parsing_anno)

    if save_vis_path is not None:
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255
        num_of_class = np.max(vis_parsing_anno)
        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            if model_name == 'bisenet':
                vis_parsing_anno_color[index[0], index[1], :] = part_colors_bisenet[pi]
            else:
                vis_parsing_anno_color[index[0], index[1], :] = part_colors_parsenet[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        vis_im = cv2.addWeighted(img, 0.4, vis_parsing_anno_color, 0.6, 0)

        cv2.imwrite(save_vis_path, vis_im)


def main(args):
    net = build_model(args.model_name)

    img_name = os.path.basename(args.input)
    img_basename = os.path.splitext(img_name)[0]

    img_input = cv2.imread(args.input)
    img_input = cv2.resize(img_input, (512, 512), interpolation=cv2.INTER_LINEAR)

    with torch.no_grad():
        out = net.parse(args.input)[0]
    out = out.squeeze(0).cpu().numpy().argmax(0)

    vis_parsing_maps(
        img_input,
        out,
        stride=1,
        save_anno_path=os.path.join(args.output, f'{img_basename}.png'),
        save_vis_path=os.path.join(args.output, f'{img_basename}_vis.png'),
        model_name=args.model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='bisenet', help='bisenet | parsenet')
    parser.add_argument('--input', type=str, default='assets/test.jpg')
    parser.add_argument('--output', type=str, default='results', help='output folder')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    main(args)
