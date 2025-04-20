# This weights is converted using tf2onnx and onnx2torch from
# https://github.com/serengil/deepface/blob/master/deepface/models/facial_recognition/Facenet.py

from typing import List
import torch
import torch.nn as nn

from facexlib.utils.image_dto import ImageDTO


class InceptionABlock(nn.Module):
    def __init__(self, dim, hidden_dim, scale=0.17):
        super().__init__()
        self.branch_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.up_conv = nn.Conv2d(3 * hidden_dim, dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.scale = scale

    def forward(self, x):
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        output = torch.cat([branch_0, branch_1, branch_2], 1)
        output = self.up_conv(output) * self.scale + x
        output = self.relu(output)
        return output
    

class ReductionABlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch_0 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 256, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        x = self.pool(x)
        output = torch.cat([branch_0, branch_1, x], 1)
        return output


class InceptionBBlock(nn.Module):
    def __init__(self, dim, hidden_dim, scale=0.1):
        super().__init__()
        self.branch_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 7), padding=(0, 3)),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(),
        )
        self.up_conv = nn.Conv2d(2 * hidden_dim, dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.scale = scale

    def forward(self, x):
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        output = torch.cat([branch_0, branch_1], 1)
        output = self.up_conv(output) * self.scale + x
        output = self.relu(output)
        return output
    

class ReductionBBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch_0 = nn.Sequential(
            nn.Conv2d(896, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 384, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(896, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(896, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        x = self.pool(x)
        output = torch.cat([branch_0, branch_1, branch_2, x], 1)
        return output


class InceptionCBlock(nn.Module):
    def __init__(self, dim, hidden_dim, scale=0.2, output=False):
        super().__init__()
        self.branch_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
        )
        self.up_conv = nn.Conv2d(2 * hidden_dim, dim, kernel_size=1)
        self.output = output
        if not output:
            self.relu = nn.ReLU()
        self.scale = scale

    def forward(self, x):
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        output = torch.cat([branch_0, branch_1], 1)
        output = self.up_conv(output) * self.scale + x
        if not self.output:
            output = self.relu(output)
        return output


class InceptionResNetV1(nn.Module):
    def __init__(self, dimension):
        super(InceptionResNetV1, self).__init__()
        self.input_size = (160, 160)

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 80, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(80, 192, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(192, 256, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.backbone_a = nn.Sequential(
            *[InceptionABlock(256, 32) for _ in range(5)],
            ReductionABlock(),
        )

        self.backbone_b = nn.Sequential(
            *[InceptionBBlock(896, 128) for _ in range(10)],
            ReductionBBlock(),
        )

        self.backbone_c = nn.Sequential(
            *[InceptionCBlock(1792, 192) for _ in range(5)],
            InceptionCBlock(1792, 192, scale=1, output=True),
        )

        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1792, dimension, bias=False),
            nn.BatchNorm1d(dimension),
        )
    

    def get_feat(self, imgs: List[ImageDTO]):
        '''
        Get embedding features of aligned face images.

        Args:
            imgs (List[ImageDTO]): aligned face images.
        '''
        imgs = [img.resize(self.input_size, keep_ratio=True).pad(self.input_size, fill=0)
                .to_tensor(rgb2bgr=True) for img in imgs]
        imgs = torch.cat(imgs, dim=0).to(next(self.parameters()).device)
        net_out = self.forward(imgs)
        return net_out


    def get(self, imgs, bboxes, eyes_kkps=None):
        '''
        Get embedding features of unaligned face images.

        Args:
            imgs (list or str or tensor): unaligned face images.
            bboxes (list or tensor): bounding boxes with shape (batch_size, 4).
            eyes_kkps (list or tensor, optional): landmark keypoints of eyes with shape (batch_size, 4).
        '''
        only_one_img = False
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
            bboxes = [bboxes]
            eyes_kkps = [eyes_kkps]
            only_one_img = True
        
        if eyes_kkps is None:
            eyes_kkps = [None] * len(imgs)
        
        aligned_imgs = [ImageDTO(img).crop_align(bbox, eyes_kps) for img, bbox, eyes_kps in zip(imgs, bboxes, eyes_kkps)]
        
        outputs = self.get_feat(aligned_imgs)
        return outputs[0] if only_one_img else outputs


    def forward(self, x):
        x = self.input_layer(x)
        x = self.backbone_a(x)
        x = self.backbone_b(x)
        x = self.backbone_c(x)
        x = self.output_layer(x)
        return x
