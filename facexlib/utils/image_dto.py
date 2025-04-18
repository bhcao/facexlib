from typing import Tuple, Union
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F_v
import torchvision.transforms._functional_tensor as F_t

from timm.layers import to_2tuple, to_3tuple

class ImageDTO:
    def __init__(self, image: Union[Image.Image, np.ndarray, str, Path, torch.Tensor, "ImageDTO"],
                 bgr2rgb=None, min_max=(-1, 1), keep_tensor=True):
        """
        A data transfer object for image data.

        Args:
            image (Union[Image.Image, np.ndarray, str, Path, torch.Tensor]): The path to the image file
                or the image data.
            bgr2rgb (bool, optional): Whether to change bgr to rgb. When image is a tensor, you need to
                determine whether it is in RGB format already. Defaults to True for `np.ndarray` (read by 
                cv2) and False for `torch.Tensor` (generated by torch models).
            min_max (tuple, optional): The min and max values for clamping. Only valid when image is
                torch.Tensor.
            keep_tensor (bool, optional): Whether to keep the input tensor for gradient calculation.
        """
        # scale first, padding later
        self.scale = (1.0, 1.0)
        self.left_top = (0.0, 0.0)

        if image is None:
            return

        if isinstance(image, Image.Image):
            self.image = np.array(image.convert('RGB'))
        
        elif isinstance(image, str) or isinstance(image, Path):
            self.image = np.array(Image.open(image).convert('RGB'))
        
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # opencv uses BGR format in default
            if bgr2rgb is None or bgr2rgb:
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
                self.image = tensor[[2, 1, 0], :, :] if bgr2rgb else tensor
            else:
                # CHW -> HWC
                image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
                if bgr2rgb:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Change type to uint8
                self.image = ((image + 1.0) * 127.5).round().astype(np.uint8)

        elif isinstance(image, ImageDTO):
            self.scale = image.scale
            self.left_top = image.left_top
            self.orig_shape = image.orig_shape

            # convert tensor to numpy array
            if isinstance(image.image, torch.Tensor) and not keep_tensor:
                image = image.image.detach().cpu().numpy().transpose(1, 2, 0)
                self.image = ((image + 1.0) * 127.5).round().astype(np.uint8)
            else:
                self.image = image.image
            return
        else:
            raise TypeError(f'Unsupported image type: {type(image)}')

        self.orig_shape = self.size


    @property
    def size(self) -> Tuple[int, int]:
        '''The size of the image in (width, height).'''
        if isinstance(self.image, torch.Tensor):
            return tuple(self.image.shape[1:][::-1])
        else:
            return tuple(self.image.shape[:2][::-1])


    def resize(self, new_shape: tuple, keep_ratio=False, align_shorter_edge=False, max_size=None) -> "ImageDTO":
        """
        Resize the image to a specified shape. Restoring bbox is supported.

        Args:
            new_shape (tuple): The target shape (width, height) for resizing.
            keep_ratio (bool): Whether to keep the aspect ratio of the image.
            align_shorter_edge (bool): Whether to align the shorter edge to the target shape.
            max_size (int): The maximum size of the resized image.

        Returns:
            The resized image data.
        """
        result = ImageDTO(None)
        result.orig_shape = self.orig_shape

        img_shape = self.size

        # update new_shape
        new_shape = to_2tuple(new_shape)
        if keep_ratio:
            orig_ratio = img_shape[1] / img_shape[0]
            new_ratio = new_shape[1] / new_shape[0]

            if (orig_ratio > new_ratio) ^ align_shorter_edge:
                new_shape = (int(round(new_shape[1] / orig_ratio)), new_shape[1])
            else:
                new_shape = (new_shape[0], int(round(new_shape[0] * orig_ratio)))

            if max_size is not None:
                if orig_ratio >= 1 and new_shape[1] > max_size:
                    new_shape = (int(round(max_size / orig_ratio)), max_size)
                elif orig_ratio < 1 and new_shape[0] > max_size:
                    new_shape = (max_size, int(round(max_size * orig_ratio)))

        # apply resize
        if isinstance(self.image, torch.Tensor):
            result.image = F_v.resize(self.image, (new_shape[1], new_shape[0]), interpolation=F_v.InterpolationMode.BILINEAR, antialias=False)
        else:
            result.image = cv2.resize(self.image, new_shape, interpolation=cv2.INTER_LINEAR)

        # update scale and left_top
        scale = (new_shape[0] / img_shape[0], new_shape[1] / img_shape[1])
        result.scale = (self.scale[0] * scale[0], self.scale[1] * scale[1])
        result.left_top = (self.left_top[0] * scale[0], self.left_top[1] * scale[1])

        return result
    

    def pad(self, new_shape=None, center=True, fill=114, stride=None) -> "ImageDTO":
        """
        Pad the image to a specified shape. Restoring bbox is supported.

        Args:
            new_shape (tuple): The target shape (width, height) for padding.
            center (bool): Whether to center the image or align to top-left.
            fill (int): The value to fill the padding.
            stride (int): When specified, padding will only to make the image shape divisible by the stride.

        Returns:
            The padded image data.
        """
        assert (new_shape is not None) ^ (stride is not None), 'Must and can only specify one of new_shape or stride.'

        result = ImageDTO(None)
        result.orig_shape = self.orig_shape

        img_shape = self.size
        if new_shape is not None:
            new_shape = to_2tuple(new_shape)
            dw, dh = new_shape[0] - img_shape[0], new_shape[1] - img_shape[1]
        else:
            dw, dh = (-img_shape[0]) % stride, (-img_shape[1]) % stride
        
        top, left = 0, 0
        if dw > 0 or dh > 0:
            if center:
                top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
                left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
            else:
                top, bottom = 0, dh
                left, right = 0, dw
            
            if isinstance(self.image, torch.Tensor):
                result.image = F_v.pad(self.image, (left, top, right, bottom), fill=fill / 127.5 - 1.0)
            else:
                result.image = cv2.copyMakeBorder(self.image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill)
        else:
            result.image = self.image
        
        result.scale = self.scale
        result.left_top = (self.left_top[0] + left, self.left_top[1] + top)

        return result


    def align(self, new_shape, kps_src, kps_dst, fill=(135, 133, 132)) -> "ImageDTO":
        """
        Align the image by transforming the source keypoints to the destination keypoints. Restoring bbox is not supported.

        Args:
            new_shape (tuple): The target shape (width, height) for aligning.
            kps_src (np.ndarray): The source keypoints with shape (num_keypoints, 2).
            kps_dst (np.ndarray): The destination keypoints with shape (num_keypoints, 2).
            fill (tuple): The value to fill the padding. Default to gray.

        Returns:
            The aligned image data.
        """
        result = ImageDTO(None)
        result.orig_shape = self.orig_shape

        # caculate affine matrix
        kps_src = np.array(kps_src).reshape(-1, 2)
        kps_dst = np.array(kps_dst).reshape(-1, 2)
        M_affine = cv2.estimateAffinePartial2D(kps_src, kps_dst, method=cv2.LMEDS)[0]

        new_shape = to_2tuple(new_shape)
        fill = to_3tuple(fill)

        if isinstance(self.image, torch.Tensor):
            fill = tuple(i / 127.5 - 1.0 for i in fill)
            M_affine = torch.cat([torch.from_numpy(M_affine).float(), torch.tensor([[0.0, 0.0, 1.0]])], dim=0)

            # convert cv2 affine matrix to pytorch affine matrix
            M_norm = torch.tensor(
                [[2.0 / self.image.shape[2], 0.0 , -1.0],
                [0.0 , 2.0 / self.image.shape[1], -1.0]]
            )
            M_inv_norm = torch.tensor(
                [[new_shape[0] / 2.0, 0.0, new_shape[0] / 2.0],
                [0.0, new_shape[1] / 2.0, new_shape[1] / 2.0],
                [0.0, 0.0, 1.0]]
            )
            M_affine = torch.mm(torch.mm(M_norm, torch.inverse(M_affine)), M_inv_norm).unsqueeze(0)

            # apply affine transform. Use _apply_grid_transform instead of grid_sample to allow filling value
            grid = F.affine_grid(M_affine, [1, self.image.shape[0], new_shape[1], new_shape[0]], align_corners=False)
            aligned_img = F_t._apply_grid_transform(self.image, grid, mode="bilinear", fill=fill)
        else:
            aligned_img = cv2.warpAffine(self.image, M_affine, new_shape, borderValue=fill)
        
        result.image = aligned_img
        return result


    def to_tensor(self, rgb2bgr=False, mean=0.0, std=255.0, timm_form=False) -> torch.Tensor:
        """
        Convert the image data to a tensor. Normalize to range (0, 1) by default.

        Args:
            rgb2bgr (bool): Whether to change RGB to BGR. Set to True if your model is trained with BGR format.
            mean (list, optional): The mean values for normalization.
            std (list, optional): The std values for normalization.
            timm_form (bool): Whether the mean and std values are in range (0, 1) like in timm.

        Returns:
            The image data in tensor format and the resize ratio if size is not None.
        """
        mean = to_3tuple(mean)
        std = to_3tuple(std)

        if timm_form:
            mean = tuple(i * 255.0 for i in mean)
            std = tuple(i * 255.0 for i in std)

        # convert image to tensor or use the kept tensor
        if isinstance(self.image, torch.Tensor):
            tensor = self.image.unsqueeze(0)
            mean = tuple(m / 127.5 - 1.0 for m in mean)
            std = tuple(s / 127.5 for s in std)
        else:
            tensor = torch.from_numpy(self.image).permute(2, 0, 1).float().unsqueeze(0)

        if rgb2bgr:
            tensor = tensor[:, [2, 1, 0], :, :]

        if any(m != 0.0 for m in mean) or any(s != 1.0 for s in std):
            F_v.normalize(tensor, mean=mean, std=std, inplace=True)
        
        return tensor

    
    def to_image(self, to_pil=False) -> np.ndarray | Image.Image:
        """
        Convert the image data to a numpy array (in bgr format) or a PIL Image and save.

        Args:
            to_pil (bool): Whether to convert to PIL Image.

        Returns:
            The image data in numpy array or PIL Image format.
        """
        if isinstance(self.image, torch.Tensor):
            image = self.image.detach().cpu().numpy().transpose(1, 2, 0)
            image = ((image + 1.0) * 127.5).round().astype(np.uint8)
        else:
            image = self.image

        if to_pil:
            image = Image.fromarray(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image

    
    def restore_keypoints(self, keypoints: np.ndarray | torch.Tensor, uniformed=False) -> np.ndarray | torch.Tensor:
        """
        Restore and clamp the keypoints such as bounding boxes or landmarks to the original image size.

        Args:
            keypoints (Tensor): The bounding boxes or landmarks with shape (num_boxes, num_keypoints * 2) or
                (num_boxes, num_keypoints, 2) in the resized image.
            uniformed (bool): Whether the keypoints are uniformed.

        Returns:
            The bounding boxes or landmarks in the original image.
        """
        # Avoid `input tensor and written-to tensor refer to a single memory` error
        keypoints = keypoints.clone() if isinstance(keypoints, torch.Tensor) else np.copy(keypoints)
        if len(keypoints) == 0:
            return keypoints

        reshaped = False
        if len(keypoints.shape) == 2:
            assert keypoints.shape[-1] % 2 == 0, 'The last dimension of 2D keypoints should be even.'
            keypoints = keypoints.reshape(keypoints.shape[0], -1, 2)
            reshaped = True
        elif len(keypoints.shape) == 3:
            assert keypoints.shape[-1] == 2, 'The last dimension of 3D keypoints should be 2.'
        else:
            raise ValueError(f'Unsupported keypoints shape: {keypoints.shape}')
    
        if uniformed:
            img_shape = self.size
            keypoints[:, :, 0] = keypoints[:, :, 0] * img_shape[0]
            keypoints[:, :, 1] = keypoints[:, :, 1] * img_shape[1]
        
        keypoints[:, :, 0] = (keypoints[:, :, 0] - self.left_top[0]) / self.scale[0]
        keypoints[:, :, 1] = (keypoints[:, :, 1] - self.left_top[1]) / self.scale[1]

        # clamp keypoints
        clamp_func = torch.clamp if isinstance(keypoints, torch.Tensor) else np.clip
        keypoints[..., 0] = clamp_func(keypoints[..., 0], 0, self.orig_shape[0])
        keypoints[..., 1] = clamp_func(keypoints[..., 1], 0, self.orig_shape[1])

        if reshaped:
            keypoints = keypoints.reshape(keypoints.shape[0], -1)

        return keypoints


    def save(self, file_path: str, params=None, auto_mkdir=True) -> bool:
        """
        Save the image data to the specified path.

        Args:
            file_path (str): The path to save the image.
            params (None or list): Same as opencv's `imwrite` interface.
            auto_mkdir (bool): If the parent folder of `file_path` does not exist,
                whether to create it automatically.

        Returns:
            Whether the image is saved successfully.
        """

        if auto_mkdir:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        image = self.to_image()
        return cv2.imwrite(file_path, image, params)