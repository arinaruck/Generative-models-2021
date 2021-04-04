from easydict import EasyDict as edict
import torch

IN_CH = 3
H, W = 128, 128


def make_config(model):
    if model == 'VAE':
        config = {
            'downsampling':
                [
                    #  (in_ch, out_ch, k_sz, strd, pad)
                    (32, 64, 4, 2, 1),
                    (64, 128, 4, 2, 1),
                    (128, 256, 4, 2, 1),
                    (256, 512, 4, 2, 1),

                ],
            'upsampling':
                [
                    (512, 256, 3, 1, 1),
                    (256, 128, 3, 1, 1),
                    (128, 64, 3, 1, 1),
                    (64, 32, 3, 1, 1),
                ],
            'initial': (IN_CH, 32, 7, 1, 3),
            'final': (32, IN_CH, 7, 1, 3),
            'out_ch': 512,
            'out_side': H // 2 ** 4,
            'hidden': 256,
            'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            'lr': 3e-4,
            'batch_size': 32
        }
    elif model == 'Glow':
        config = {
            'in_channels': 3,
            'n_flows': 8,
            'n_blocks': 3,
            'lu': True,
            'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            'lr': 2e-4,
            'batch': 64,
            'n_flow': 16,
            'n_block': 3,
            'conv_lu': True,
            'affine': True,
            'n_bits': 5,
            'img_size': 64,
            'temp': 0.7,
            'n_sample': 64

        }
    elif model == 'vapnev':
        config = {
            'downsampling':
                [
                    #  (in_ch, out_ch, k_sz, strd, pad)
                    (16, 32, 4, 2, 1),
                    (32, 64, 4, 2, 1),
                    (64, 128, 4, 2, 1),
                    (128, 256, 4, 2, 1),

                ],
            'upsampling':
                [
                    (256, 128, 3, 1, 1),
                    (128, 64, 3, 1, 1),
                    (64, 32, 3, 1, 1),
                ],
            'upsampling_cond':
                [
                    (256, 128, 3, 1, 1),
                    (128, 64, 3, 1, 1),
                    (64, 32, 3, 1, 1),
                ],
            'initial': (IN_CH, 16, 7, 1, 3),
            'final': (32, IN_CH, 7, 1, 3),
            'final_cond': (32, 2 * IN_CH, 7, 1, 3),
            'in_channels': 3,
            'out_channels': 256,
            'out_side': 64 // 2 ** 4,
            'in_side': 64,
            'z_dim': 128,
            'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            'lr': 3e-4,
            'batch_size': 16,
            'n_flows': 3,
            'n_blocks': 2,
            'n_bits': 5,
            'temp': 0.7,
            'n_sample': 64
        }
    else:
        print(f'There is no config for model: {model}')
        return
    return edict(config)