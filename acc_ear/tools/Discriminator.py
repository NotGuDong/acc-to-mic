import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_channels=2, input_size=(240, 240)):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # 输入层: (batch, 2, 30, 30)
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # (32, 15, 15)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 4, 4)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (256, 2, 2)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 展平层
            nn.Flatten(),

            # # 全连接层
            # nn.Linear(256 * 2 * 2, 512),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.4),

            # nn.Linear(512, 256),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.4),

            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),

            nn.Linear(64, 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),

            # 输出层
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, combine_spec):
        patches = self._extract_patches(combine_spec)
        if patches is not None:
            # 直接处理所有patch，获取预测结果 (num_patches, 1)
            all_validity = self.model(patches)

            # 计算原始batch_size
            b = combine_spec.shape[0]
            num_patches_per_sample = patches.shape[0] // b

            # 重塑为 (b, num_patches_per_sample, 1)
            validity_reshaped = all_validity.view(b, num_patches_per_sample, 1)

            # 对每个batch的所有patch结果取平均
            batch_results = validity_reshaped.mean(dim=1).squeeze(-1)

            return batch_results
        else:
            return torch.zeros(combine_spec.shape[0], device=combine_spec.device)

    def _extract_patches(self, images, patch_size=30):
        """提取所有可能的patch_size x patch_size patch"""
        patches = []
        b, c, h, w = images.shape

        # 确保能整除patch_size，否则裁剪
        if h % patch_size != 0:
            images = images[:, :, :h - (h % patch_size), :]
        if w % patch_size != 0:
            images = images[:, :, :, :w - (w % patch_size)]
        b, c, h, w = images.shape  # 更新后的形状

        # 计算步长（非重叠patch）
        h_steps = h // patch_size
        w_steps = w // patch_size

        # 提取所有patch
        for i in range(h_steps):
            for j in range(w_steps):
                patch = images[:, :,
                        i * patch_size:(i + 1) * patch_size,
                        j * patch_size:(j + 1) * patch_size]
                patches.append(patch)

        if patches:
            return torch.cat(patches, dim=0)
        return None