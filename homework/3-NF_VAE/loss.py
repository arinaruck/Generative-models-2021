import torch.nn as nn
import torch


class VAELoss():
    def __init__(self, device, lmbd_kl=1):
        self.MSE = nn.MSELoss(reduction='sum').to(device)
        self.lmbd_kl = lmbd_kl

    def __call__(self, x_gen, x_real, mu, logvar):
        b_sz = x_gen.shape[0]
        mse = self.MSE(x_gen.view(b_sz, -1), x_real.view(b_sz, -1))
        kl =  -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = mse + self.lmbd_kl * kl
        loss_dict = {'total_loss': total_loss.item(),
                     'reconstruction_loss': mse.item(), 'kl': kl.item()}
        return loss_dict, total_loss