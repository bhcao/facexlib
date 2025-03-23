import torch
import argparse
from PIL import Image

from facexlib.resolution import init_resolution_model
from facexlib.utils.misc import imwrite

def main(args):

    # create model, tile_size: max patch size for the tile mode, suitable with your GPU memory
    model = init_resolution_model(args.model_name, args.scale, tile_size = 256, tile_pad = 32, half = args.half)
    print(f'Model [{model.__class__.__name__}] is created.')

    with torch.no_grad():
        sr_img = model.inference(Image.open(args.img_path))

    imwrite(sr_img, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='../assets/test.jpg')
    parser.add_argument('--model_name', type=str, default='HAT', help='HAT-S | HAT | HAT-L')
    parser.add_argument('--scale', type=int, default=4, help='2 | 4')
    parser.add_argument('--output_path', type=str, default='./test.png')
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()

    main(args)