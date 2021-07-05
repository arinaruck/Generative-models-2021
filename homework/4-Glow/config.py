from easydict import EasyDict as edict
import torch

IN_CH = 3
H, W = 128, 128


def make_config():
    config = {
        'in_channels': 3,
        'n_flows': 8,
        'n_blocks': 3,
        'lu': True,
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'lr': 2e-4,
        'batch': 64,
        'iter': 200000,
        'n_flow': 16,
        'n_block': 3,
        'conv_lu': True,
        'affine': True, #False,
        'n_bits': 5,
        'img_size': 64,
        'temp': 0.7,
        'n_sample': 64,
        'path': '/content/dataset'
    }
    config = edict(config)
    return config