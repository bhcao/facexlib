from typing import Union
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from timm.layers import to_2tuple, to_3tuple

class ImageDTO:
    def __init__(self, image: Union[Image.Image, np.ndarray, str, Path, torch.Tensor, "ImageDTO"],
                 bgr2rgb=True, min_max=(-1, 1), keep_tensor=False):
        """
        A data transfer object for image data.

        Args:
            image (Union[Image.Image, np.ndarray, str, Path, torch.Tensor]): The path to the image file
                or the image data.
            bgr2rgb (bool, optional): Whether to change bgr to rgb. When image is a tensor, you need to
                determine whether it is in RGB format already.
            min_max (tuple, optional): The min and max values for clamping. Only valid when image is
                torch.Tensor.
            keep_tensor (bool, optional): Whether to keep the input tensor for gradient calculation.
        """
        self.tensor = None
        self.last_scale = (1, 1)

        if isinstance(image, Image.Image):
            self.image = np.array(image.convert('RGB'))
        
        elif isinstance(image, str) or isinstance(image, Path):
            self.image = np.array(Image.open(image).convert('RGB'))
        
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # opencv uses BGR format in default
            if bgr2rgb:
                self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                self.image = image
        elif isinstance(image, torch.Tensor):
            tensor = image.squeeze(0).float().clamp(*min_max)
            # normalize to [-1, 1] if min_max is not (-1, 1)
            if min_max != (-1, 1):
                tensor = (2 * tensor - (min_max[1] + min_max[0])) / (min_max[1] - min_max[0])
    
            assert tensor.dim() == 3, f'Only support 3D tensor. But received with dimension: {tensor.dim()}'

            # the kept tensor is in BGR format and in range [-1, 1]
            if keep_tensor:
                self.tensor = tensor[::-1, :, :] if bgr2rgb else tensor

            # CHW -> HWC
            image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
            if bgr2rgb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Change type to uint8
            self.image = ((image + 1.0) * 127.5).round().astype(np.uint8)
        elif isinstance(image, ImageDTO):
            self.image = image.image
            self.last_scale = image.last_scale
            self.tensor = image.tensor
        else:
            raise TypeError(f'Unsupported image type: {type(image)}')


    def to_tensor(self, size=None, keep_ratio=False, center=True, fill=114, stride=None, rgb2bgr=False, mean=None,
                  std=None, to_01=True, device=None, dtype=None) -> torch.Tensor:
        """
        Convert the image data to a tensor.
        
        Resizing and padding images to a specified shape then normalize the tensor.

        Args:
            size (tuple): Target shape (height, width) for resizing.
            keep_ratio (bool): Whether to keep the aspect ratio of the image.
            center (bool): Whether to center the image or align to top-left.
            fill (int): The value to fill the padding.
            stride (int): When specified, padding will only to make the image shape divisible by the stride.
            rgb2bgr (bool): Whether to change RGB to BGR. Set to True if your model is trained with BGR format.
            mean (list, optional): The mean values for normalization.
            std (list, optional): The std values for normalization.
            to_01 (bool): Whether to normalize the tensor to [0, 1].
            device (str, optional): The device to put the tensor.
            dtype (torch.dtype, optional): The data type of the tensor.

        Returns:
            The image data in tensor format and the resize ratio if size is not None.
        """
        # convert image to tensor or use the kept tensor
        if self.tensor is not None:
            tensor = self.tensor.unsqueeze(0)
            if to_01:
                tensor = (tensor + 1.0) / 2.0
            else:
                tensor = ((tensor + 1.0) * 127.5).round()
        else:
            tensor = torch.from_numpy(self.image).permute(2, 0, 1).float().unsqueeze(0)
            if to_01:
                tensor = tensor / 255.0

        if to_01:
            fill = fill / 255.0

        if rgb2bgr:
            tensor = tensor[:, ::-1, :, :]

        if size is not None:
            size = to_2tuple(size)
            # determine size and padding
            if keep_ratio:
                im_ratio = self.image.shape[0] / self.image.shape[1]
                model_ratio = size[1] / size[0]
                if im_ratio > model_ratio:
                    new_h = size[1]
                    new_w = int(round(new_h / im_ratio))
                else:
                    new_w = size[0]
                    new_h = int(round(new_w * im_ratio))
                
                dw, dh = size[0] - new_w, size[1] - new_h
            else:
                dw, dh = 0, 0
                new_w, new_h = size
            
            if stride is not None:
                dw, dh = dw % stride, dh % stride
            
            self.last_scale = (new_w / self.image.shape[1], new_h / self.image.shape[0])
    
            # resize
            if new_h != self.image.shape[0] or new_w != self.image.shape[1]:
                tensor = F.interpolate(tensor, size=(new_h, new_w), mode='bilinear')
    
            # padding
            if dw > 0 or dh > 0:
                if center:
                    top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
                    left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
                else:
                    top, bottom = 0, dh
                    left, right = 0, dw
                
                tensor = F.pad(tensor, (left, right, top, bottom), mode='constant', value=fill)

        if mean is not None or std is not None:
            mean = mean if mean is not None else 0.0
            std = std if std is not None else 1.0
            TF.normalize(tensor, mean=mean, std=std, inplace=True)
        
        tensor = tensor.to(device=device, dtype=dtype)
        return tensor
    
    