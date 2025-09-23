import numpy as np
import torch
from torch import nn


class DownUpConv(nn.Module):
    """基础卷积块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, input_channels=2, output_channels=1):
        super().__init__()

        # 编码器部分
        self.encoder1 = DownUpConv(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DownUpConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DownUpConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # 中间部分
        self.bottleneck = DownUpConv(256, 512)

        # 解码器部分
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DownUpConv(512, 256)  # 512 = 256 + 256(跳跃连接)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DownUpConv(256, 128)  # 256 = 128 + 128(跳跃连接)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DownUpConv(128, 64)  # 128 = 64 + 64(跳跃连接)

        # 输出层
        self.outconv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # 编码器
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        # 中间层
        bottleneck = self.bottleneck(self.pool3(enc3))

        # 解码器
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # 输出
        return self.outconv(dec1)
