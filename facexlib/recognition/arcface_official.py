import torch
import torch.nn as nn
import cv2

class ResBlock(nn.Module):
    def __init__(self, in_chans, scale=1, downsample=False):
        super().__init__()

        out_chans = in_chans * scale

        self.bach_norm = nn.BatchNorm2d(in_chans)
        self.conv_1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.prelu = nn.PReLU(out_chans)

        self.use_conv_skip = downsample
        if downsample:
            self.conv_2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=2, padding=1)
            self.conv_skip = nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=2)
        else:
            self.conv_2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)


    def forward(self, x):
        if self.use_conv_skip:
            resudial = self.conv_skip(x)
        else:
            resudial = x

        x = self.bach_norm(x)
        x = self.conv_1(x)
        x = self.prelu(x)
        x = self.conv_2(x)
        x = x + resudial
        return x


class ResGroup(nn.Module):
    def __init__(self, num_blocks, in_chans, no_scale=False):
        super().__init__()

        self.input_layer = ResBlock(in_chans, scale = 1 if no_scale else 2, downsample=True)

        out_chans = in_chans if no_scale else in_chans * 2
        self.layers = nn.Sequential(
            *[ResBlock(out_chans) for _ in range(num_blocks - 1)]
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layers(x)
        return x


class ArcFace(nn.Module):
    def __init__(self, layer_num):
        super().__init__()

        self.input_mean = 127.5
        self.input_std = 127.5
        
        self.input_size = (112, 112)

        # models
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.prelu = nn.PReLU(64)

        self.layer_group_1 = ResGroup(layer_num[0], 64, no_scale=True)
        self.layer_group_2 = ResGroup(layer_num[1], 64)
        self.layer_group_3 = ResGroup(layer_num[2], 128)
        self.layer_group_4 = ResGroup(layer_num[3], 256)

        self.batch_norm = nn.BatchNorm2d(512)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(in_features=25088, out_features=512)
        self.batch_norm_output = nn.BatchNorm1d(512)


    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]

        blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, self.input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        
        blob = torch.from_numpy(blob).to(next(self.parameters()).device)
        net_out = self.forward(blob)
        return net_out


    # must transform to [0, 1] before input
    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)

        x = self.layer_group_1(x)
        x = self.layer_group_2(x)
        x = self.layer_group_3(x)
        x = self.layer_group_4(x)

        x = self.batch_norm(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.batch_norm_output(x)
        return x