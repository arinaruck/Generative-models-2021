import wandb
import torch
from torch import nn
from torch.optim import Adam
from loss import StarGANLoss
from utils import permute_labels, compute_gradient_penalty


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()

        layers = []
        layers += [DownBlock(*params) for params in config.downsampling]
        layers += [ResBlock(*params) for params in config.residual]
        layers += [UpBlock(*params) for params in config.upsampling]
        layers += [nn.Conv2d(*config.final), nn.Tanh()]

        self.network = nn.Sequential(*layers)

    def forward(self, x, labels):
        labels = labels.view(labels.size(0), labels.size(1), 1, 1)
        labels = labels.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, labels], dim=1)
        return self.network(x)


class Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        layers += [CriticBlock(*params) for params in config.downsampling]
        self.encode = nn.Sequential(*layers)
        self.src_out = nn.Conv2d(*config.src_out)
        self.cls_out = nn.Conv2d(*config.cls_out)

    def forward(self, x):
        b_sz = x.size(0)
        encoded = self.encode(x)
        src_out = self.src_out(encoded)
        cls_out = self.cls_out(encoded)
        return src_out, cls_out.view(b_sz, -1)


class StarGAN:
    def __init__(self, config):
        self.G = Generator(config.generator)
        self.D = Critic(config.critic)

        self.device = config.device

        self.optimizerD = Adam(self.D.parameters(), lr=config.lr_D, betas=(0.5, 0.999))
        self.optimizerG = Adam(self.G.parameters(), lr=config.lr_G, betas=(0.5, 0.999))
        self.criterion = StarGANLoss(config.device)
        self.to(config.device)

    def train(self):
        self.G.train()
        self.D.train()

    def eval(self):
        self.G.eval()
        self.D.eval()

    def to(self, device):
        self.D.to(device)
        self.G.to(device)

    def trainG(self, image, label):
        self.optimizerG.zero_grad()
        new_label = permute_labels(label)
        generated = self.G(image, new_label)
        reconstructed = self.G(generated, label)
        src_out, cls_out = self.D(generated)
        loss_dict, loss = self.criterion.generator_loss(image, reconstructed, src_out, cls_out, new_label)
        wandb.log(loss_dict)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), 10.0)
        self.optimizerG.step()
        self.optimizerD.zero_grad()

    def trainD(self, image, label):
        self.optimizerD.zero_grad()
        new_label = permute_labels(label)
        generated = self.G(image, new_label).detach()
        src_out_gen, _ = self.D(generated)
        src_out, cls_out = self.D(image)
        gp = compute_gradient_penalty(self.D, generated, image, self.device)
        loss_dict, loss = self.criterion.discriminator_loss(image, src_out_gen, src_out, cls_out, label, gp)
        wandb.log(loss_dict)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), 10.0)
        self.optimizerD.step()
        self.optimizerG.zero_grad()

    @torch.no_grad()
    def generate(self, image, label):
        generated = self.G(image, label)
        return generated


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_ch, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_ch, affine=True, track_running_stats=True))

    def forward(self, x):
        x = x + self.block(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(out_ch, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class CriticBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, leaky):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.LeakyReLU(leaky, inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x