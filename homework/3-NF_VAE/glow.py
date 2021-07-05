import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import linalg as la
from blocks import InvConv2dLU, ZeroInitedConv2d, gaussian_log_p, gaussian_sample, logabs


# based on https://github.com/rosinality/glow-pytorch/blob/master/model.py
class ActNorm(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.initialized = False

    def initialize(self, input):
        with torch.no_grad():
            channels = input.size(1)
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = flatten.mean(1).view(1, channels, 1, 1)
            std = flatten.std(1).view(1, channels, 1, 1)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape
        if not self.initialized:
            self.initialize(input)
            self.initialized = True
        log_abs = logabs(self.scale)
        logdet = height * width * torch.sum(log_abs)
        return self.scale * (input + self.loc), logdet

    def reverse(self, output):
        return output / self.scale - self.loc


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroInitedConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channels, affine=True):
        super().__init__()
        self.actnorm = ActNorm(in_channels)
        self.invconv = InvConv2dLU(in_channels)
        self.coupling = AffineCoupling(in_channels, affine=affine)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine))

        self.split = split

        if split:
            self.prior = ZeroInitedConv2d(in_channel * 2, in_channel * 4)

        else:
            self.prior = ZeroInitedConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        for flow in self.flows:
            out, curr_logdet = flow(out)
            logdet += curr_logdet

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)

            else:
                zero = torch.zeros_like(input)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed


class Glow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block = nn.ModuleList()
        n_channel = config.in_channels
        self.n_block = config.n_blocks
        self.n_flow = config.n_flows
        self.blocks = nn.ModuleList()
        for i in range(self.n_block - 1):
            self.blocks.append(Block(n_channel, config.n_flow,
                                     affine=config.affine))
            n_channel *= 2
        self.blocks.append(Block(n_channel, config.n_flow,
                                 split=False, affine=config.affine))
        self.device = config.device

    def init_with_data(self, batch):
        b_sz, n_channels, input_size, _ = batch.size()
        self.z_shapes = self.calc_z_shapes(n_channels, input_size,
                                           self.n_flow, self.n_block)

    def calc_z_shapes(self, n_channels, input_size, n_flows, n_blocks):
        z_shapes = []
        for i in range(n_blocks - 1):
            input_size //= 2
            n_channels *= 2
            z_shapes.append((n_channels, input_size, input_size))
        input_size //= 2
        z_shapes.append((n_channels * 4, input_size, input_size))
        return z_shapes

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        input = z_list[-1]
        for i, block in enumerate(self.blocks[::-1]):
            input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)
        return input

    @torch.no_grad()
    def sample(self, n, t=0.5):
        z_sample = []
        for z in self.z_shapes:
            z_new = torch.randn(n, *z) * t
            z_sample.append(z_new.to(self.device))
        return self.reverse(z_sample)