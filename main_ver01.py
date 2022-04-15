from email.policy import default
import os
import numpy as np
import torch

from src.DGAProcess import LearningProcess

from src.DGALossV2 import DGALoss
from src.DGANet import DGANet
from src.DGANetV2 import DGANetV2

from termcolor import cprint


import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--id', type=str, default=None)
parser.add_argument('-c', type=int, default=-1)
parser.add_argument('--lr', type=float, default=-1.0)
parser.add_argument('--v_windows', nargs='+', type=int, default=[64])
args = parser.parse_args()
print(args.__dict__)

params = {
    'result_dir': os.path.join('/root/denoise/results', args.id),
    'test_dir': os.path.join('/root/denoise/results', args.id, 'tests'),

    'dataset': {
        'train_seqs': [
            'MH_01_easy',
            'MH_03_medium',
            'MH_05_difficult',
            'V1_02_medium',
            'V2_01_easy',
            'V2_03_difficult'
        ],
        'val_seqs': [
            'MH_02_easy',
            'MH_04_difficult',
            'V2_02_medium',
            'V1_03_difficult',
            'V1_01_easy',
        ],
        'test_seqs': [
            'MH_02_easy',
            'MH_04_difficult',
            'V2_02_medium',
            'V1_03_difficult',
            'V1_01_easy',
        ],
        # size of trajectory during training
        'data_dir':     "/root/denoise/data",
        'predata_dir':  os.path.join('/root/denoise/results', 'predata'),
        'N': 32 * 500, # should be integer * 'max_train_freq'
        'min_train_freq': 16,
        'max_train_freq': 32,
        'v_window': 16,
    },

    'net_class': DGANet,
    'net': {
        'in_dim': 6,
        'out_dim': 6,
        'c0': 16,
        'dropout': 0.1,
        'ks': [7, 7, 7, 7],
        'ds': [4, 4, 4],
        'momentum': 0.1,
        'gyro_std': [1*np.pi/180, 2*np.pi/180, 5*np.pi/180],
        'acc_std': [2.0e-3, 2.0e-3, 2.0e-3],
    },

    'train': {
        'optimizer_class': torch.optim.Adam,
        'optimizer': {
            'lr': 0.01,
            'weight_decay': 1e-1,
            'amsgrad': False,
        },
        'loss_class': DGALoss,
        'loss': {
            'min_N': 4, # int(np.log2(dataset_params['min_train_freq'])),
            'max_N': 5, # int(np.log2(dataset_params['max_train_freq'])),
            'w':  1e6,
            'huber': 0.005,
            'dt': 0.005,
            'v_window': 16,
        },
        'scheduler_class': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        'scheduler': {
            'T_0': 600,
            'T_mult': 2,
            'eta_min': 1e-3,
        },
        'dataloader': {
            'batch_size': 6,
            'pin_memory': False,
            'num_workers': 0,
            'shuffle': False,
        },

        # frequency of validation step
        'freq_val': 20,
        # total number of epochs
        'n_epochs': 400,
    }
}

# params['net_class'] = DGANetV2
# params['net'] = {
#     'in_dim':  6,
#     'out_dim': 6,
#     'channel': [32, 64, 64, 128, 128, 256],
#     'kernal':  [5,  5,  5,  5,   5,   5],
#     'dilation':[2,  4,  8,  16,   32,   64],
#     'stride':  [1,  1,  1,  1,   1,   1],
#     'momentum': 0.1,
#     'gyro_std': [1*np.pi/180, 2*np.pi/180, 5*np.pi/180],
#     'acc_std': [2.0e-3, 2.0e-3, 2.0e-3],
# }

if __name__ == '__main__':
    if args.c >= 0:
        params['net']['c0'] = args.c
        cprint('Note :: params/net/c0 = %d' % args.c, 'green')
    if args.lr >= 0:
        params['train']['optimizer']['lr'] = args.lr
        cprint('Note :: params/train/optimizer/lr = %d' % args.lr, 'green')
    if args.v_windows != [64]:
        params['train']['loss']['v_windows'] = args.v_windows
        params['dataset']['v_windows'] = args.v_windows
        cprint('Note :: params/train/loss/v_window = %d' % args.v_windows, 'green')

    process = LearningProcess(params, args.mode)


    if args.mode == 'train':
        cprint('\n========== Train ==========\n', 'cyan', attrs=['bold'])
        process.train()
    elif args.mode == 'test':
        cprint('\n========== Test ==========\n', 'cyan', attrs=['bold'])
        process.test()
    elif args.mode == 'anal':
        cprint('\n========== Analysis ==========\n', 'cyan', attrs=['bold'])
        process.analyze()
    else:
        cprint("argument 'mode' must be one of ['train', 'test', 'anal']", 'red')
