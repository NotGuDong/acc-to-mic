import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, ReLU


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=(3, 3), bias=False, stride=(stride, stride), padding=(1, 1))
        self.bn2 = BatchNorm2d(out_channels)
        self.conv3 = Conv2d(out_channels, out_channels * 4, kernel_size=(1, 1), bias=False)
        self.bn3 = BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            Conv2d(in_channels, out_channels * 4, kernel_size=(1, 1), stride=(stride, stride), bias=False),
            BatchNorm2d(out_channels * 4)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn2 = BatchNorm2d(out_channels)
        self.conv3 = Conv2d(out_channels, in_channels, kernel_size=(1, 1), bias=False)
        self.bn3 = BatchNorm2d(in_channels)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += x
        out = self.relu(out)
        return out


class MyResnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = nn.Sequential(
            ConvBlock(64, 64, 1),
            IdentityBlock(256, 64),
            IdentityBlock(256, 64)
        )
        self.layer2 = nn.Sequential(
            ConvBlock(256, 128, 2),
            IdentityBlock(512, 128),
            IdentityBlock(512, 128),
            IdentityBlock(512, 128)
        )
        self.layer3 = nn.Sequential(
            ConvBlock(512, 256, 2),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256)
        )
        self.layer4 = nn.Sequential(
            ConvBlock(1024, 512, 2),
            IdentityBlock(2048, 512),
            IdentityBlock(2048, 512),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# models = Resnet50()
# # print(models)
#
# input = torch.ones((64, 3, 32, 32))
# output = models(input)
# print(output.shape)
