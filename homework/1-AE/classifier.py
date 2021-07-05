import torch
import torch.nn as nn
from base import DownsampleBlock


class Classifier(nn.Module):
    def __init__(self, n_classes=10, mode='mnist'):
        super().__init__()
        self.n_classes = n_classes
        self.hidden = 2 * 2 * 32
        if mode == 'mnist':
            self.encode = nn.Sequential(
                DownsampleBlock(1, 16),  # 64 x 64 x 1 ->  32 x 32 x 16
                DownsampleBlock(16, 32),  # 32 x 32 x 16 -> 16 x 16 x 32
                DownsampleBlock(32, 64),  # 16 x 16 x 32 -> 8 x 8 x 64
                DownsampleBlock(64, 32),  # 8 x 8 x 64 -> 4 x 4 x 32
                DownsampleBlock(32, 32),  # 4 x 4 x 32 -> 2 x 2 x 32
                nn.Flatten()
            )
            self.classify = nn.Sequential(
                nn.Linear(2 * 2 * 32, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, self.n_classes)
            )
        else:
            self.encode = nn.Sequential(
                DownsampleBlock(1, 32),  # 64 x 64 x 1 ->  32 x 32 x 32
                DownsampleBlock(32, 64),  # 32 x 32 x 32 -> 16 x 16 x 64
                DownsampleBlock(64, 128),  # 16 x 16 x 64 -> 8 x 8 x 128
                DownsampleBlock(128, 256),  # 8 x 8 x 128 -> 4 x 4 x 256
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.Flatten()
            )
            self.classify = nn.Sequential(
                nn.Linear(4 * 4 * 256, 1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, self.n_classes)
            )

    def forward(self, x):
        x = self.encode(x)
        x = self.classify(x)
        return x

    def get_activations(self, x):
        return self.encode(x)