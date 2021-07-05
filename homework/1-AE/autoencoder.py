import torch
import torch.nn as nn
import torch.nn.functional as F
from base import DownsampleBlock, UpsampleBlock


class AutoEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            DownsampleBlock(1, 32),     # 64 x 64 x 1 ->  32 x 32 x 16
            DownsampleBlock(32, 64),    # 32 x 32 x 16 -> 16 x 16 x 32
            DownsampleBlock(64, 128),   # 16 x 16 x 32 -> 8 x 8 x 64
            DownsampleBlock(128, 256),  # 8 x 8 x 64 -> 4 x 4 x 256
            DownsampleBlock(256, 128),  # 4 x 4 x 256 -> 2 x 2 x 128
            DownsampleBlock(128, hidden_size) # 2 x 2 x 128 -> 1 x 1 x hidden
        )

        self.decoder = nn.Sequential(
            UpsampleBlock(self.hidden_size, 16),  # 1 x 1 x hidden -> 2 x 2 x 4
            UpsampleBlock(16, 32),            # 2 x 2 x 4 -> 4 x 4 x 8
            UpsampleBlock(32, 64),           # 4 x 4 x 8 -> 8 x 8 x 16
            UpsampleBlock(64, 128),          # 8 x 8 x 16 -> 16 x 16 x 32
            UpsampleBlock(128, 256),          # 16 x 16 x 32 -> 32 x 32 x 64
            UpsampleBlock(256, 256),         # 32 x 32 x 64 -> 64 x 64 x 128
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

    def get_latent_features(self, x):
        z = self.encoder(x)
        return z