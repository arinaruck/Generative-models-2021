import torch
from torch import nn
import torch.nn.functional as F
from blocks import UpBlock, DownBlock


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        layers = [nn.Conv2d(*config.initial)]
        layers += [DownBlock(*params) for params in config.downsampling]
        layers += [nn.Flatten()]

        self.encode = nn.Sequential(*layers)
        out_side = config.out_side
        out_ch = config.out_ch
        hidden = config.hidden
        self.mean = nn.Linear(out_side ** 2 * out_ch, hidden)
        self.logvar = nn.Linear(out_side ** 2 * out_ch, hidden)

    @staticmethod
    def reparametrize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs):
        batch_size = inputs.size(0)
        hidden = self.encode(inputs)
        mean, logvar = self.mean(hidden), self.logvar(hidden)
        z = self.reparametrize(mean, logvar)
        return z, mean, logvar


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.out_side = config.out_side
        self.out_ch = config.out_ch
        hidden = config.hidden
        self.dense = nn.Sequential(
            nn.Linear(hidden, self.out_side ** 2 * self.out_ch),
            nn.LeakyReLU(0.2, True),
        )

        layers = [UpBlock(*params) for params in config.upsampling]
        layers += [nn.Conv2d(*config.final), nn.Tanh()]
        self.decode = nn.Sequential(*layers)

    def forward(self, hidden):
        b_sz = hidden.size(0)
        x = self.dense(hidden)
        x = x.view(b_sz, self.out_ch, self.out_side, self.out_side)
        x = self.decode(x)
        return x


class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.hidden = config.hidden
        self.device = config.device
        self.to(config.device)

    def forward(self, x):
        z, mean, logvar = self.encoder(x)
        x = self.decoder(z)
        return x, mean, logvar

    def encode(self, x):
        z, _, _ = self.encoder(x)
        return z

    def decode(self, z):
        return self.decoder(z)

    @torch.no_grad()
    def sample(self, n):
        sample = torch.randn(n, self.hidden).to(self.device)
        return self.decode(sample)