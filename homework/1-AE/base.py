import torch
import torch.nn as nn


class DownsampleBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.net(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.net(x)
