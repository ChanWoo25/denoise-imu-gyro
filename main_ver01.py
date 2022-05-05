import os
import numpy as np
import torch

from src.DGAProcess import LearningProcess

from src.DGALoss import DGALoss
from src.DGANet import DGANet

from termcolor import cprint


import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--id', type=str, default=None)
parser.add_argument('--c0', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--dv', nargs='+', type=int, default=[16, 32])
parser.add_argument('--dv_normed', nargs='+', type=int, default=[32, 64])
args = parser.parse_args()
print(args.__dict__)


params = {
    'debug': args.debug,
    'result_dir': os.path.join('/root/denoise/results', args.id),
    'test_dir': os.path.join('/root/denoise/results', args.id, 'tests'),
    'id': args.id,

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
        'c0': args.c0,
        'dropout': 0.1,
        'ks': [7, 7, 7, 7],
        'ds': [4, 4, 4],
        'momentum': 0.1,
        'gyro_std': [1*np.pi/180, 2*np.pi/180, 5*np.pi/180],
        'acc_std': [2.0e-3, 2.0e-3, 2.0e-3], #? Is this proper range ?#
    },

    'train': {
        'optimizer_class': torch.optim.Adam,
        'optimizer': {
            'lr': args.lr,
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
            'dv': args.dv,
            'dv_normed': args.dv_normed,
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
        'freq_val': 40,
        # total number of epochs
        'n_epochs': 1000,
    }
}

if __name__ == '__main__':

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
