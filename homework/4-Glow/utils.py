import wandb
import random
import os
import numpy as np
import torch


@torch.no_grad()
def log_images(model, epoch, dataloader, device, model_type='VAE', mode='reconstruct'):
    model.eval()
    image, label = next(iter(dataloader))
    b_sz = image.size(0)
    real_image = (image + 1) / 2
    real_image = real_image.numpy()
    image = image.to(device)
    if model_type == 'VAE':
        fake_image, _, _ = model(image)
    elif model_type == 'Glow':
        log_p, logdet, z_out = model(image)
        fake_image = model.reverse(z_out, reconstruct=True)
    fake_image = fake_image.cpu().numpy()
    fake_image = (fake_image + 1) / 2

    sampled_image = model.sample(b_sz)
    sampled_image = sampled_image.cpu().numpy()
    sampled_image = (sampled_image + 1) / 2
    wandb.log({'epoch': epoch,
               'real images': [
                   wandb.Image(real_image[i].transpose(1, 2, 0),
                               caption=f"real, epoch: {epoch}")
                   for i in range(b_sz)],
               'fake images': [
                   wandb.Image(fake_image[i].transpose(1, 2, 0),
                               caption=f"fake, epoch: {epoch}")
                   for i in range(b_sz)],
               'sampled images': [
                   wandb.Image(sampled_image[i].transpose(1, 2, 0),
                               caption=f"sampled, epoch: {epoch}")
                   for i in range(b_sz)]})
    model.train()


def seed_all(seed=1):
    """
    fixes seeds everywhere
    :param seed: random seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)