import torch
import torch.nn as nn
from torch.nn import functional as F
from math import log, pi, exp

logabs = lambda x: torch.log(torch.abs(x))


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(out_ch, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(in_ch, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(out_ch, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_ch, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_ch, affine=True, track_running_stats=True))

        self.block[0].weight.data.normal_(0, 0.05)
        self.block[0].bias.data.zero_()
        self.block[3].weight.data.normal_(0, 0.05)
        self.block[3].bias.data.zero_()

    def forward(self, x):
        x = x + self.block(x)
        return x


def add_dims(x):
    return x.unsqueeze(2).unsqueeze(3)


class InvConv2dLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        weight = torch.randn(in_channels, in_channels)
        q, _ = torch.qr(weight)
        q_LU, pivots = q.lu()
        w_p, w_l, w_u = torch.lu_unpack(q_LU, pivots)

        w_s = torch.diag(w_u)
        w_u = torch.triu(w_u, 1)
        u_mask = torch.triu(torch.ones_like(w_u), 1)
        l_mask = u_mask.T

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", u_mask)
        self.register_buffer("l_mask", l_mask)
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
                self.w_p
                @ (self.w_l * self.l_mask + self.l_eye)
                @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return add_dims(weight)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, add_dims(weight.squeeze().inverse()))


class ZeroInitedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)
        return out