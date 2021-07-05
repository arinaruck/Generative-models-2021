from easydict import EasyDict as edict
import torch

N_CLS = 5
IN_CH = 3 + N_CLS
H, W = 128, 128


def make_config():
    config = {
        'generator':
            {
            'downsampling':
                [
                #  (in_ch, out_ch, k_sz, strd, pad)
                    (IN_CH, 64,     7,  1,  3),
                    (64,    128,    4,  2,  1),
                    (128,   256,    4,  2,  1),

                ],
            'residual':
                [
                    (256, )
                ] * 6,
            'upsampling':
                [
                #  (in_ch, out_ch, k_sz, strd, pad)
                    (256,   128,    3,  1,  1),
                    (128,   64,     3,  1,  1),
                ],
            'final':
                    (64,    3,      7,  1,  3),
            },
        'critic':
            {
                'downsampling':
                    [
                        (3,     64,     4,  2,  1,  0.01),
                        (64,    128,    4,  2,  1,  0.01),
                        (128,   256,    4,  2,  1,  0.01),
                        (256,   512,    4,  2,  1,  0.01),
                        (512,   1024,   4,  2,  1,  0.01),
                        (1024,  2048,   4,  2,  1,  0.01),
                    ],
                'src_out':
                        (2048,  1,      3,  1,  1),
                'cls_out':
                        (2048, N_CLS,   (H // 64, W // 64), 1,  0)
            },
        'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        'lr_D': 5e-4,
        'lr_G': 1e-4
    }
    config = edict(config)
    return config