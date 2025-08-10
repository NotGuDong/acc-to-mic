import torch
import torch.nn as nn
from torch.nn import Conv2d
from timm.models.layers import DropPath
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvBlock(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.conv1 = Conv2d(dim, dim, kernel_size=(7, 7), padding=(3, 3), groups=dim)
        self.lm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # 1x1卷积
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path =DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        # 调换通道顺序
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.lm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.layer_scale is not None:
            x = self.layer_scale * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class MyConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4), bias=False)
        self.ln1 = LayerNorm(96, eps=1e-6, data_format="channels_first")

        self.layer1 = nn.Sequential(
            ConvBlock(96),
            ConvBlock(96),
            ConvBlock(96)
        )

        self.layer2 = nn.Sequential(
            LayerNorm(96, eps=1e-6, data_format="channels_first"),
            Conv2d(96, 192, kernel_size=(2, 2), stride=(2, 2), bias=False),

            ConvBlock(192),
            ConvBlock(192),
            ConvBlock(192)
        )
        self.layer3 = nn.Sequential(
            LayerNorm(192, eps=1e-6, data_format="channels_first"),
            Conv2d(192, 384, kernel_size=(2, 2), stride=(2, 2), bias=False),
            ConvBlock(384),
            ConvBlock(384),
            ConvBlock(384),
            ConvBlock(384),
            ConvBlock(384),
            ConvBlock(384),
            ConvBlock(384),
            ConvBlock(384),
            ConvBlock(384)
        )
        self.layer4 = nn.Sequential(
            LayerNorm(384, eps=1e-6, data_format="channels_first"),
            Conv2d(384, 768, kernel_size=(2, 2), stride=(2, 2), bias=False),
            ConvBlock(768),
            ConvBlock(768),
            ConvBlock(768),
        )
        self.ln2 = LayerNorm(768, eps=1e-6)
        self.fc = nn.Linear(768, 1000, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.ln2(x.mean([-2, -1]))  # GAP
        x = self.fc(x)
        return x


