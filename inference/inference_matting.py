import argparse
import cv2
import numpy as np
import torch.nn.functional as F

from facexlib.utils import build_model, ImageDTO


def main(args):
    modnet = build_model('modnet')

    img = ImageDTO(args.img_path)
    img_t = img.to_tensor(mean=127.5, std=127.5).to(device=next(modnet.parameters()).device)

    # resize image for input
    _, _, im_h, im_w = img_t.shape
    ref_size = 512
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    img_t = F.interpolate(img_t, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(img_t, True)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    cv2.imwrite(args.save_path, (matte * 255).astype('uint8'))

    # get foreground
    matte = matte[:, :, None]
    foreground = img.image * matte + np.full(img.image.shape, 1) * (1 - matte)
    cv2.imwrite(args.save_path.replace('.png', '_fg.png'), foreground * 255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='assets/test.jpg')
    parser.add_argument('--save_path', type=str, default='test_matting.png')
    args = parser.parse_args()

    main(args)
