import torch
from torch import nn
from torch.nn import functional as F
from blocks import InvConv2dLU, UpBlock, DownBlock, ResBlock, gaussian_log_p, gaussian_sample, logabs


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        layers = [nn.Conv2d(*config.initial)]
        layers += [DownBlock(*params) for params in config.downsampling]
        layers += [nn.Flatten()]

        self.encode = nn.Sequential(*layers)
        out_side = config.out_side
        out_channels = config.out_channels
        hidden = config.z_dim
        self.mean = nn.Linear(out_side ** 2 * out_channels, hidden)
        self.log_sd = nn.Linear(out_side ** 2 * out_channels, hidden)

    @staticmethod
    def reparametrize(mu, log_sd):
        std = torch.exp(log_sd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs):
        batch_size = inputs.size(0)
        hidden = self.encode(inputs)
        mean, log_sd = self.mean(hidden), self.log_sd(hidden)
        z = self.reparametrize(mean, log_sd)
        return z, mean, log_sd


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.out_side = config.out_side
        self.in_side = config.in_side
        self.out_channels = config.out_channels
        self.in_channels = config.in_channels
        hidden = config.z_dim
        self.dense = nn.Sequential(
            nn.Linear(hidden, self.out_side ** 2 * self.out_channels),
            nn.LeakyReLU(0.2, True),
        )

        layers = [UpBlock(*params) for params in config.upsampling]
        layers += [nn.Conv2d(*config.final), nn.Tanh()]
        self.decode = nn.Sequential(*layers)
        self.mean = nn.Linear(self.in_side ** 2 * self.in_channels // 4, self.in_side ** 2 * self.in_channels)
        self.log_sd = nn.Linear(self.in_side ** 2 * self.in_channels // 4, self.in_side ** 2 * self.in_channels)

    def forward(self, hidden):
        b_sz = hidden.size(0)
        x = self.dense(hidden)
        x = x.view(b_sz, self.out_channels, self.out_side, self.out_side)
        x = self.decode(x).view(b_sz, -1)
        mean = self.mean(x).view(b_sz, self.in_channels, self.in_side, self.in_side)
        log_sd = self.log_sd(x).view(b_sz, self.in_channels, self.in_side, self.in_side)
        return mean, log_sd


class VAPNEV(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.cond_glow = GlowSimple(config)
        self.hidden = config.z_dim
        self.device = config.device
        self.to(config.device)

    def forward(self, x):
        b_sz = x.size(0)
        z, _, _ = self.encoder(x)
        mean, log_sd = self.decoder(z)
        y, log_det = self.cond_glow(x, z)
        log_p = gaussian_log_p(y, mean, log_sd)
        return y, log_p, log_det

    @torch.no_grad()
    def reverse(self, x, t=0.5):
        z, _, _ = self.encoder(x)
        print('z:', torch.isfinite(z).all())
        mean, log_sd = self.decoder(z)
        print('mean:', torch.isfinite(mean).all(), 'log sd:', torch.isfinite(log_sd).all())
        y = gaussian_sample(torch.randn_like(mean), mean, log_sd)
        print('y:', torch.isfinite(y).all())
        result = self.cond_glow.reverse(y, z)
        print('result:', torch.isfinite(result).all())
        return result

    @torch.no_grad()
    def sample(self, n, t=0.5):
        z = torch.randn(n, self.hidden).to(self.device) * t
        mean, log_sd = self.decoder(z)
        y = gaussian_sample(torch.randn_like(mean), mean, log_sd)
        return self.cond_glow.reverse(y, z)


class ConditionalProj(nn.Module):
    def __init__(self, config, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.out_channels = out_channels
        self.out_side = config.out_side
        self.x_net = nn.Sequential(
            ResBlock(in_channels, hidden_channels),
            nn.LeakyReLU(0.2),
            ResBlock(in_channels, hidden_channels),
            nn.LeakyReLU(0.2)
        )

        self.dense = nn.Sequential(
            nn.Linear(config.z_dim, self.out_side ** 2 * self.out_channels),
            nn.LeakyReLU(0.2, True),
        )

        layers = [UpBlock(*params) for params in config.upsampling_cond]
        layers += [nn.Conv2d(*config.final_cond)]
        self.z_net = nn.Sequential(*layers)

        param_d = (1, in_channels, 1, 1)
        self.alpha = nn.Parameter(torch.zeros(*param_d))
        self.beta1 = nn.Parameter(torch.zeros(*param_d))
        self.beta2 = nn.Parameter(torch.zeros(*param_d))
        self.bias = nn.Parameter(torch.zeros(*param_d))

    def forward(self, x, z):
        b_sz, ch, _, _ = x.size()
        z = self.dense(z)
        z = z.view(b_sz, self.out_channels, self.out_side, self.out_side)
        x_x = self.x_net(x)
        x_z = self.z_net(z)
        x = self.alpha * x_x * x_z + self.beta1 * x_x + self.beta2 * x_z + self.bias
        return x


class ConditionalCoupling(nn.Module):
    def __init__(self, config, in_channels, out_channels, hidden_channels=16):
        super().__init__()
        self.m = ConditionalProj(config, in_channels // 2, out_channels, hidden_channels)
        self.log_s = ConditionalProj(config, in_channels // 2, out_channels, hidden_channels)

    def forward(self, x, z):
        assert torch.isfinite(x).all(), 'coupling input x nan'
        assert torch.isfinite(z).all(), 'coupling input z nan'
        in_a, in_b = x.chunk(2, 1)
        m, log_s = self.m(in_a, z), self.log_s(in_a, z)
        assert torch.isfinite(log_s).all() and torch.isfinite(m).all(), 'log_s or m nan'
        s = F.sigmoid(log_s + 2)
        out_b = in_b * s + m
        logdet = torch.sum(torch.log(s).view(x.shape[0], -1), 1)
        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, x, z):
        assert torch.isfinite(x).all() and torch.isfinite(z).all(), 'coupling reverse input nan'
        out_a, out_b = x.chunk(2, 1)
        m, log_s = self.m(out_a, z), self.log_s(out_a, z)
        assert torch.isfinite(log_s).all() and torch.isfinite(m).all(), 'log_s or m nan'
        s = F.sigmoid(log_s + 2)
        assert torch.isfinite(s).all(), 's nan'
        in_b = out_b / (s + 1e-6) - m
        assert torch.isfinite(in_b).all(), f'in_b nan, s_min={s.min()}, s_max={s.max()}'
        return torch.cat([out_a, in_b], 1)


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


class Flow(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        self.actnorm = ActNorm(in_channels)
        self.invconv = InvConv2dLU(in_channels)
        self.coupling = ConditionalCoupling(config, in_channels, out_channels)

    def forward(self, x, z):
        out, norm_logdet = self.actnorm(x)
        out, conv_logdet = self.invconv(out)
        out, coupling_logdet = self.coupling(out, z)
        logdet = norm_logdet + conv_logdet + coupling_logdet
        return out, logdet

    def reverse(self, output, z):
        assert torch.isfinite(output).all(), 'flow reverse input nan'
        input = self.coupling.reverse(output, z)
        assert torch.isfinite(input).all(), 'coupling reverse output nan'
        input = self.invconv.reverse(input)
        assert torch.isfinite(input).all(), 'invconv reverse output nan'
        input = self.actnorm.reverse(input)
        assert torch.isfinite(input).all(), 'actnorm reverse output nan'
        return input


class Block(nn.Module):
    def __init__(self, config, in_channels, out_channels, n_flows):
        super().__init__()
        squeeze_dim = in_channels * 4
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(Flow(config, squeeze_dim, out_channels))

    def forward(self, x, z):
        b_size, n_channel, height, width = x.shape
        squeezed = x.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
        logdet = 0

        for flow in self.flows:
            out, curr_logdet = flow(out, z)
            logdet += curr_logdet

        unsqueezed = out.view(b_size, n_channel, 2, 2, height // 2, width // 2)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        out = unsqueezed.contiguous().view(b_size, n_channel, height, width)
        return out, logdet

    def reverse(self, output, z):
        b_size, n_channel, height, width = output.shape
        squeezed = output.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        input = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        for flow in self.flows[::-1]:
            input = flow.reverse(input, z)

        unsqueezed = input.view(b_size, n_channel, 2, 2, height // 2, width // 2)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(b_size, n_channel, height, width)
        return unsqueezed


class GlowSimple(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block = nn.ModuleList()
        self.in_channels = config.in_channels
        self.n_blocks = config.n_blocks
        self.blocks = nn.ModuleList()
        n_channels = config.in_channels
        for i in range(self.n_blocks):
            self.blocks.append(Block(config, n_channels, config.out_channels, config.n_flows))
        self.device = config.device

    def forward(self, x, z):
        out, logdet = x, 0
        for block in self.blocks:
            out, curr_logdet = block(out, z)
            logdet += curr_logdet
        return out, logdet

    def reverse(self, x, z):
        input = x
        for i, block in enumerate(self.blocks[::-1]):
            input = block.reverse(input, z)
        return input