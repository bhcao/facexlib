from typing import List
import torch
import torch.nn.functional as F
from torch import nn
import math

from facexlib.utils import ImageDTO
from .utils import mtlface_dst


class SPPModule(nn.Module):
    def __init__(self, pool_mode='avg', sizes=(1, 2, 3, 6)):
        super().__init__()
        if pool_mode == 'avg':
            pool_layer = nn.AdaptiveAvgPool2d
        elif pool_mode == 'max':
            pool_layer = nn.AdaptiveMaxPool2d
        else:
            raise NotImplementedError

        self.pool_blocks = nn.ModuleList([
            nn.Sequential(pool_layer(size), nn.Flatten()) for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.pool_blocks]
        x = torch.cat(xs, dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x


class AttentionModule(nn.Module):
    def __init__(self, channels=512, reduction=16):
        super(AttentionModule, self).__init__()
        kernel_size = 7
        pool_size = (1, 2, 3)
        self.avg_spp = SPPModule('avg', pool_size)
        self.max_spp = SPPModule('max', pool_size)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                      dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )

        _channels = channels * int(sum([x ** 2 for x in pool_size])) * 2
        self.channel = nn.Sequential(
            nn.Conv2d(
                _channels, _channels // reduction, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                _channels // reduction, channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )
        self.norm = nn.Identity()
        self.act = nn.Identity()

    def forward(self, x):
        channel_input = torch.cat([self.avg_spp(x), self.max_spp(x)], dim=1)
        channel_scale = self.channel(channel_input)

        spatial_input = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        spatial_scale = self.spatial(spatial_input)
        scale = (channel_scale + spatial_scale) * 0.5

        x_id = x * scale
        x_id = self.act(self.norm(x_id))

        return x_id


def get_block(unit_module, in_channel, depth, num_units, stride=2):
    layers = [unit_module(in_channel, depth, stride)] + [unit_module(depth, depth, 1) for _ in range(num_units - 1)]
    return nn.Sequential(*layers)


class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False), nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class IResNet(nn.Module):
    dropout_ratio = 0.4

    def __init__(self, input_size, num_layers, mode='ir', amp=False):
        super(IResNet, self).__init__()
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"

        if mode == 'ir':
            unit_module = bottleneck_IR
        else:
            raise NotImplementedError

        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))

        block1 = get_block(unit_module, in_channel=64, depth=64, num_units=num_layers[0])
        block2 = get_block(unit_module, in_channel=64, depth=128, num_units=num_layers[1])
        block3 = get_block(unit_module, in_channel=128, depth=256, num_units=num_layers[2])
        block4 = get_block(unit_module, in_channel=256, depth=512, num_units=num_layers[3])
        self.body = nn.Sequential(block1, block2, block3, block4)

        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(p=self.dropout_ratio, inplace=True)
        self.fc = nn.Linear(512 * (input_size // 16) ** 2, 512)
        self.features = nn.BatchNorm1d(512, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        self.amp = amp

        self._initialize_weights()

    def forward(self, x):
        with torch.cuda.amp.autocast(self.amp):
            x = self.input_layer(x)
            x = self.body(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.amp else x)
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()


class MTLFaceEncoder(nn.Module):
    def __init__(self, input_size=112, device='cpu'):
        super(MTLFaceEncoder, self).__init__()
        self.input_size = input_size
        self.device = device

        self.n_styles = math.ceil(math.log(input_size, 2)) * 2 - 2
        facenet = IResNet(input_size, [3, 4, 14, 3], 'ir') # IR-50
        self.input_layer = facenet.input_layer
        self.block1 = facenet.body[0]
        self.block2 = facenet.body[1]
        self.block3 = facenet.body[2]
        self.block4 = facenet.body[3]
        self.output_layer = nn.Sequential(facenet.bn2, nn.Flatten(), facenet.dropout, facenet.fc, facenet.features)
        self.fsm = AttentionModule()


    def forward(self, x):
        x = self.input_layer(x)
        c1 = self.block1(x)
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        x = self.block4(c3)
        x_id = self.fsm(x)
        x_vec = F.normalize(self.output_layer(x_id), dim=1)
        return x_vec
    

    def get_feat(self, imgs: List[ImageDTO]):
        '''
        Get embedding features of aligned face images.

        Args:
            imgs (List[ImageDTO]): aligned face images.
        '''
    
        imgs = torch.concat([img.resize(self.input_size).to_tensor(std=127.5, mean=127.5) for img in imgs], 
                            dim=0).to(self.device)
        x_vec1 = self.forward(imgs)
        x_vec2 = self.forward(torch.flip(imgs, dims=(3,)))
        x_vec = F.normalize(x_vec1 + x_vec2, dim=1)
        return x_vec
    

    def get(self, imgs, landmarks):
        '''
        Get embedding features of unaligned face images. Align face images using landmark keypoints first.

        Args:
            imgs (list or str or tensor): unaligned face images.
            landmarks (list or tensor): landmark keypoints with shape (batch_size, 10).
        '''
        only_one_img = False
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
            landmarks = [landmarks]
            only_one_img = True
        
        aligned_imgs = [ImageDTO(img).kps_align(self.input_size, landmark, mtlface_dst, fill=0)
                        for img, landmark in zip(imgs, landmarks)]
        
        outputs = self.get_feat(aligned_imgs)
        return outputs[0] if only_one_img else outputs


