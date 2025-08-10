from __future__ import print_function
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self, classes):
        super(VAE, self).__init__()
        # 特征提取
        prev_channels = 3
        latent_dims = 2048
        modules = []
        img_length = 256
        for cur_channels in [16, 32, 64, 128, 256, 512, 1024, 2048]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels,
                              cur_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1), nn.BatchNorm2d(cur_channels),
                    nn.ReLU()))
            prev_channels = cur_channels
            img_length //= 2
        self.encoder = nn.Sequential(*modules)
        self.mean_linear = nn.Linear(prev_channels * img_length * img_length,
                                     latent_dims)
        self.var_linear = nn.Linear(prev_channels * img_length * img_length,
                                    latent_dims)
        self.latent_dim = latent_dims

        # 分类器
        self.class_classifier = nn.Sequential(
            nn.Linear(2048, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, 256, 256)
        encoded = self.encoder(input_data)
        encoded = torch.flatten(encoded, 1)
        mean = self.mean_linear(encoded)
        logvar = self.var_linear(encoded)
        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        feature = eps * std + mean
        feature = feature.view(feature.size(0), -1)
        class_output = self.class_classifier(feature)

        return class_output