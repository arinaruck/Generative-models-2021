import torch
from torch import nn
import os
import random
import numpy as np


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
        b_sz = real_samples.size(0)

        # Calculate interpolation
        alpha = torch.rand(b_sz, 1, 1, 1)
        alpha = alpha.expand_as(real_samples).to(device)
        interpolated = (alpha * real_samples.data + (1 - alpha) * fake_samples.data).requires_grad_(True)

        prob_interpolated, _ = critic(interpolated)
        prob_interpolated = prob_interpolated.view(b_sz, -1).mean(-1)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(real_samples.shape[0], requires_grad=False, device=device),
            # gradients w.t. output. 1 is default
            create_graph=True,
            retain_graph=True,  # keep all gradients for further optimization steps
            only_inputs=True,
        )[0]

        gradients = gradients.view(b_sz, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


def permute_labels(labels):
    b_sz = labels.size(0)
    idx = torch.randperm(b_sz)
    return labels[idx]


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


def filter_idx(dataset, idx):
    subset_idx = []
    for i, (_, lbl) in enumerate(dataset):
        if sum(lbl[idx]) > 0:
            subset_idx.append(i)
    return subset_idx