"""
Code adapted from timm https://github.com/huggingface/pytorch-image-models

Modifications and additions for mivolo by / Copyright 2023, Irina Tolstykh, Maxim Kuprashevich
"""

import torch
import torch.nn as nn
from timm.layers.bottleneck_attn import PosEmbedRel
from timm.layers.helpers import make_divisible
from timm.layers.mlp import Mlp
from timm.layers.trace_utils import _assert
from timm.layers.weight_init import trunc_normal_
from timm.models._builder import build_model_with_cfg
from timm.models._registry import register_model
from timm.models.volo import VOLO
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

__all__ = ["MiVOLOModel"]  # model_registry will add each entrypoint fn to this

class CrossBottleneckAttn(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        feat_size=None,
        stride=1,
        num_heads=4,
        dim_head=None,
        qk_ratio=1.0,
        qkv_bias=False,
        scale_pos_embed=False,
    ):
        super().__init__()
        assert feat_size is not None, "A concrete feature size matching expected input (H, W) is required"
        dim_out = dim_out or dim
        assert dim_out % num_heads == 0

        self.num_heads = num_heads
        self.dim_head_qk = dim_head or make_divisible(dim_out * qk_ratio, divisor=8) // num_heads
        self.dim_head_v = dim_out // self.num_heads
        self.dim_out_qk = num_heads * self.dim_head_qk
        self.dim_out_v = num_heads * self.dim_head_v
        self.scale = self.dim_head_qk**-0.5
        self.scale_pos_embed = scale_pos_embed

        self.qkv_f = nn.Conv2d(dim, self.dim_out_qk * 2 + self.dim_out_v, 1, bias=qkv_bias)
        self.qkv_p = nn.Conv2d(dim, self.dim_out_qk * 2 + self.dim_out_v, 1, bias=qkv_bias)

        # NOTE I'm only supporting relative pos embedding for now
        self.pos_embed = PosEmbedRel(feat_size, dim_head=self.dim_head_qk, scale=self.scale)

        self.norm = nn.LayerNorm([self.dim_out_v * 2, *feat_size])
        mlp_ratio = 4
        self.mlp = Mlp(
            in_features=self.dim_out_v * 2,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            out_features=dim_out,
            drop=0,
            use_conv=True,
        )

        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.qkv_f.weight, std=self.qkv_f.weight.shape[1] ** -0.5)  # fan-in
        trunc_normal_(self.qkv_p.weight, std=self.qkv_p.weight.shape[1] ** -0.5)  # fan-in
        trunc_normal_(self.pos_embed.height_rel, std=self.scale)
        trunc_normal_(self.pos_embed.width_rel, std=self.scale)

    def get_qkv(self, x, qvk_conv):
        B, C, H, W = x.shape

        x = qvk_conv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W

        q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)

        q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
        k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
        v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)

        return q, k, v

    def apply_attn(self, q, k, v, B, H, W, dropout=None):
        if self.scale_pos_embed:
            attn = (q @ k + self.pos_embed(q)) * self.scale  # B * num_heads, H * W, H * W
        else:
            attn = (q @ k) * self.scale + self.pos_embed(q)
        attn = attn.softmax(dim=-1)
        if dropout:
            attn = dropout(attn)

        out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
        return out

    def forward(self, x):
        B, C, H, W = x.shape

        dim = int(C / 2)
        x1 = x[:, :dim, :, :]
        x2 = x[:, dim:, :, :]

        _assert(H == self.pos_embed.height, "")
        _assert(W == self.pos_embed.width, "")

        q_f, k_f, v_f = self.get_qkv(x1, self.qkv_f)
        q_p, k_p, v_p = self.get_qkv(x2, self.qkv_p)

        # person to face
        out_f = self.apply_attn(q_f, k_p, v_p, B, H, W)
        # face to person
        out_p = self.apply_attn(q_p, k_f, v_f, B, H, W)

        x_pf = torch.cat((out_f, out_p), dim=1)  # B, dim_out * 2, H, W
        x_pf = self.norm(x_pf)
        x_pf = self.mlp(x_pf)  # B, dim_out, H, W

        out = self.pool(x_pf)
        return out

def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.96,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": None,
        "classifier": ("head", "aux_head"),
        **kwargs,
    }


default_cfgs = {
    "mivolo_d1_224": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d1_224_84.2.pth.tar", crop_pct=0.96
    ),
    "mivolo_d1_384": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d1_384_85.2.pth.tar",
        crop_pct=1.0,
        input_size=(3, 384, 384),
    ),
    "mivolo_d2_224": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d2_224_85.2.pth.tar", crop_pct=0.96
    ),
    "mivolo_d2_384": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d2_384_86.0.pth.tar",
        crop_pct=1.0,
        input_size=(3, 384, 384),
    ),
    "mivolo_d3_224": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d3_224_85.4.pth.tar", crop_pct=0.96
    ),
    "mivolo_d3_448": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d3_448_86.3.pth.tar",
        crop_pct=1.0,
        input_size=(3, 448, 448),
    ),
    "mivolo_d4_224": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d4_224_85.7.pth.tar", crop_pct=0.96
    ),
    "mivolo_d4_448": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d4_448_86.79.pth.tar",
        crop_pct=1.15,
        input_size=(3, 448, 448),
    ),
    "mivolo_d5_224": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d5_224_86.10.pth.tar", crop_pct=0.96
    ),
    "mivolo_d5_448": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d5_448_87.0.pth.tar",
        crop_pct=1.15,
        input_size=(3, 448, 448),
    ),
    "mivolo_d5_512": _cfg(
        url="https://github.com/sail-sg/volo/releases/download/volo_1/d5_512_87.07.pth.tar",
        crop_pct=1.15,
        input_size=(3, 512, 512),
    ),
}


def get_output_size(input_shape, conv_layer):
    padding = conv_layer.padding
    dilation = conv_layer.dilation
    kernel_size = conv_layer.kernel_size
    stride = conv_layer.stride

    output_size = [
        ((input_shape[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i]) + 1 for i in range(2)
    ]
    return output_size


def get_output_size_module(input_size, stem):
    output_size = input_size

    for module in stem:
        if isinstance(module, nn.Conv2d):
            output_size = [
                (
                    (output_size[i] + 2 * module.padding[i] - module.dilation[i] * (module.kernel_size[i] - 1) - 1)
                    // module.stride[i]
                )
                + 1
                for i in range(2)
            ]

    return output_size


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self, img_size=224, stem_conv=False, stem_stride=1, patch_size=8, in_chans=3, hidden_dim=64, embed_dim=384
    ):
        super().__init__()
        assert patch_size in [4, 8, 16]
        assert in_chans in [3, 6]
        self.with_persons_model = in_chans == 6
        self.use_cross_attn = True

        if stem_conv:
            if not self.with_persons_model:
                self.conv = self.create_stem(stem_stride, in_chans, hidden_dim)
            else:
                self.conv = True  # just to match interface
                # split
                self.conv1 = self.create_stem(stem_stride, 3, hidden_dim)
                self.conv2 = self.create_stem(stem_stride, 3, hidden_dim)
        else:
            self.conv = None

        if self.with_persons_model:

            self.proj1 = nn.Conv2d(
                hidden_dim, embed_dim, kernel_size=patch_size // stem_stride, stride=patch_size // stem_stride
            )
            self.proj2 = nn.Conv2d(
                hidden_dim, embed_dim, kernel_size=patch_size // stem_stride, stride=patch_size // stem_stride
            )

            stem_out_shape = get_output_size_module((img_size, img_size), self.conv1)
            self.proj_output_size = get_output_size(stem_out_shape, self.proj1)

            self.map = CrossBottleneckAttn(embed_dim, dim_out=embed_dim, num_heads=1, feat_size=self.proj_output_size)

        else:
            self.proj = nn.Conv2d(
                hidden_dim, embed_dim, kernel_size=patch_size // stem_stride, stride=patch_size // stem_stride
            )

        self.patch_dim = img_size // patch_size
        self.num_patches = self.patch_dim**2

    def create_stem(self, stem_stride, in_chans, hidden_dim):
        return nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride, padding=3, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.conv is not None:
            if self.with_persons_model:
                x1 = x[:, :3]
                x2 = x[:, 3:]

                x1 = self.conv1(x1)
                x1 = self.proj1(x1)

                x2 = self.conv2(x2)
                x2 = self.proj2(x2)

                x = torch.cat([x1, x2], dim=1)
                x = self.map(x)
            else:
                x = self.conv(x)
                x = self.proj(x)  # B, C, H, W

        return x


class MiVOLOModel(VOLO):
    """
    Vision Outlooker, the main class of our model
    """

    def __init__(
        self,
        layers,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        patch_size=8,
        stem_hidden_dim=64,
        embed_dims=None,
        num_heads=None,
        downsamples=(True, False, False, False),
        outlook_attention=(True, False, False, False),
        mlp_ratio=3.0,
        qkv_bias=False,
        drop_rate=0.0,
        pos_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        post_layers=("ca", "ca"),
        use_aux_head=True,
        use_mix_token=False,
        pooling_scale=2,
    ):
        super().__init__(
            layers,
            img_size,
            in_chans,
            num_classes,
            global_pool,
            patch_size,
            stem_hidden_dim,
            embed_dims,
            num_heads,
            downsamples,
            outlook_attention,
            mlp_ratio,
            qkv_bias,
            drop_rate,
            pos_drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            post_layers,
            use_aux_head,
            use_mix_token,
            pooling_scale,
        )

        im_size = img_size[0] if isinstance(img_size, tuple) else img_size
        self.patch_embed = PatchEmbed(
            img_size=im_size,
            stem_conv=True,
            stem_stride=2,
            patch_size=patch_size,
            in_chans=in_chans,
            hidden_dim=stem_hidden_dim,
            embed_dim=embed_dims[0],
        )

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def forward_features(self, x):
        x = self.patch_embed(x).permute(0, 2, 3, 1)  # B,C,H,W-> B,H,W,C

        # step2: tokens learning in the two stages
        x = self.forward_tokens(x)

        # step3: post network, apply class attention or not
        if self.post_network is not None:
            x = self.forward_cls(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False, targets=None, epoch=None):
        if self.global_pool == "avg":
            out = x.mean(dim=1)
        elif self.global_pool == "token":
            out = x[:, 0]
        else:
            out = x
        if pre_logits:
            return out

        features = out
        fds_enabled = hasattr(self, "_fds_forward")
        if fds_enabled:
            features = self._fds_forward(features, targets, epoch)

        out = self.head(features)
        if self.aux_head is not None:
            # generate classes in all feature tokens, see token labeling
            aux = self.aux_head(x[:, 1:])
            out = out + 0.5 * aux.max(1)[0]

        return (out, features) if (fds_enabled and self.training) else out

    def forward(self, x, targets=None, epoch=None):
        """simplified forward (without mix token training)"""
        x = self.forward_features(x)
        x = self.forward_head(x, targets=targets, epoch=epoch)
        return x


def _create_mivolo(variant, pretrained=False, **kwargs):
    if kwargs.get("features_only", None):
        raise RuntimeError("features_only not implemented for Vision Transformer models.")
    return build_model_with_cfg(MiVOLOModel, variant, pretrained, **kwargs)


@register_model
def mivolo_d1_224(pretrained=False, **kwargs):
    model_args = dict(layers=(4, 4, 8, 2), embed_dims=(192, 384, 384, 384), num_heads=(6, 12, 12, 12), **kwargs)
    model = _create_mivolo("mivolo_d1_224", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d1_384(pretrained=False, **kwargs):
    model_args = dict(layers=(4, 4, 8, 2), embed_dims=(192, 384, 384, 384), num_heads=(6, 12, 12, 12), **kwargs)
    model = _create_mivolo("mivolo_d1_384", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d2_224(pretrained=False, **kwargs):
    model_args = dict(layers=(6, 4, 10, 4), embed_dims=(256, 512, 512, 512), num_heads=(8, 16, 16, 16), **kwargs)
    model = _create_mivolo("mivolo_d2_224", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d2_384(pretrained=False, **kwargs):
    model_args = dict(layers=(6, 4, 10, 4), embed_dims=(256, 512, 512, 512), num_heads=(8, 16, 16, 16), **kwargs)
    model = _create_mivolo("mivolo_d2_384", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d3_224(pretrained=False, **kwargs):
    model_args = dict(layers=(8, 8, 16, 4), embed_dims=(256, 512, 512, 512), num_heads=(8, 16, 16, 16), **kwargs)
    model = _create_mivolo("mivolo_d3_224", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d3_448(pretrained=False, **kwargs):
    model_args = dict(layers=(8, 8, 16, 4), embed_dims=(256, 512, 512, 512), num_heads=(8, 16, 16, 16), **kwargs)
    model = _create_mivolo("mivolo_d3_448", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d4_224(pretrained=False, **kwargs):
    model_args = dict(layers=(8, 8, 16, 4), embed_dims=(384, 768, 768, 768), num_heads=(12, 16, 16, 16), **kwargs)
    model = _create_mivolo("mivolo_d4_224", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d4_448(pretrained=False, **kwargs):
    """VOLO-D4 model, Params: 193M"""
    model_args = dict(layers=(8, 8, 16, 4), embed_dims=(384, 768, 768, 768), num_heads=(12, 16, 16, 16), **kwargs)
    model = _create_mivolo("mivolo_d4_448", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d5_224(pretrained=False, **kwargs):
    model_args = dict(
        layers=(12, 12, 20, 4),
        embed_dims=(384, 768, 768, 768),
        num_heads=(12, 16, 16, 16),
        mlp_ratio=4,
        stem_hidden_dim=128,
        **kwargs
    )
    model = _create_mivolo("mivolo_d5_224", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d5_448(pretrained=False, **kwargs):
    model_args = dict(
        layers=(12, 12, 20, 4),
        embed_dims=(384, 768, 768, 768),
        num_heads=(12, 16, 16, 16),
        mlp_ratio=4,
        stem_hidden_dim=128,
        **kwargs
    )
    model = _create_mivolo("mivolo_d5_448", pretrained=pretrained, **model_args)
    return model


@register_model
def mivolo_d5_512(pretrained=False, **kwargs):
    model_args = dict(
        layers=(12, 12, 20, 4),
        embed_dims=(384, 768, 768, 768),
        num_heads=(12, 16, 16, 16),
        mlp_ratio=4,
        stem_hidden_dim=128,
        **kwargs
    )
    model = _create_mivolo("mivolo_d5_512", pretrained=pretrained, **model_args)
    return model
