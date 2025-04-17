import torch
import argparse
from PIL import Image

from facexlib.utils import build_model

def main(args):

    # create model, tile_size: max patch size for the tile mode, suitable with your GPU memory
    model = build_model(args.model_name, half = args.half)
    print(f'Model [{model.__class__.__name__}] is created.')

    with torch.no_grad():
        sr_img = model.inference(Image.open(args.img_path), tile_size = 256, tile_pad = 32)

    sr_img.save(args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='assets/test2.jpg')
    parser.add_argument('--model_name', type=str, default='SwinIR_x2', help='SwinIR_x2 | HAT_x2 | HAT_x4 | HAT-L_x2 | HAT-L_x4')
    parser.add_argument('--output_path', type=str, default='./test_super_resolution.png')
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()

    main(args)