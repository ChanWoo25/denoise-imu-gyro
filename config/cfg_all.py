import torch
import numpy as np
import os

from src.DGALoss import DGALoss, DGALossVer2
from src.DGANet import DGANet, DGANetVer2, DGANetVer3
from src.DGNet import DGNetVer1
from src.DGLoss import DGLossVer1, DGLossVer2


def parse():
    import argparse
    parser = argparse.ArgumentParser()
    # 'Log & Save & Load' Stuffs
    parser.add_argument('--id', type=str, default='test')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--machine', type=str, default='server')

    # 'Train & Test' Stuffs
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--input_type', type=str, default='window')
    parser.add_argument('--c0', type=int, default=16)
    parser.add_argument('--train_batch_size', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=3200)
    parser.add_argument('--goal_epoch', type=int, default=400)
    parser.add_argument('--dv', nargs='+', type=int, default=[16, 32])
    parser.add_argument('--dv_normed', nargs='+', type=int, default=[32, 64])
    # Loss related
    parser.add_argument('--gnll_ratio', type=float, default=1.0)
    # Optimization related
    parser.add_argument('--lr', type=float, default=0.01)

    # NCP Hyper Parameters
    parser.add_argument('--n_inter', type=int, default=20)
    parser.add_argument('--n_command', type=int, default=10)
    parser.add_argument('--out_sensory', type=int, default=4)
    parser.add_argument('--out_inter', type=int, default=5)
    parser.add_argument('--rec_command', type=int, default=6)
    parser.add_argument('--in_motor', type=int, default=4)
    parser.add_argument('--ode_unfolds', type=int, default=6)

    return parser.parse_args()


def configure():
    args = parse()
    print(' ====== [Program Arguments] ====== \n', \
                      args.__dict__, \
          ' ================================= \n')

    euroc_data_dir = os.path.join('/root', 'project', 'datasets', 'EUROC')
    project_dir = os.path.join('/root', 'project', 'results', 'Once')
    if args.machine == 'server':
        euroc_data_dir = os.path.join('/home/leecw', 'project', 'datasets', 'EUROC')
        project_dir = os.path.join('/home/leecw', 'project', 'results', 'Once')


    if args.net_version == 'acc_ver1':
        Net =   DGANet
        Loss =  DGALossVer2
    elif args.net_version == 'acc_ver2':
        Net =   DGANetVer2
        Loss =  DGALossVer2
    elif args.net_version == 'acc_ver3':
        Net =   DGANetVer3
        Loss =  DGALossVer2
    elif args.net_version == 'ori_ver1':
        Net =   DGNetVer1
        Loss =  DGLossVer1
    elif args.net_version == 'ori_ver2':
        Net =   DGNetVer1
        Loss =  DGLossVer2

    params = {
        'net_version': args.net_version,
        'result_dir': os.path.join(project_dir, args.id),
        'test_dir': os.path.join(project_dir, args.id, 'tests'),
        'figure_dir': os.path.join(project_dir, 'figures'),
        'ckpt_path': args.ckpt_path,
        'id': args.id,
        'input_type': args.input_type,
        'ode_unfolds': args.ode_unfolds,
        'project_dir': project_dir,

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
            'data_dir':     euroc_data_dir,
            'predata_dir':  os.path.join(project_dir, 'predata'),
            'N': args.seq_len, # should be integer * 'max_train_freq'
            'min_train_freq': 16,
            'max_train_freq': 32,
            'v_window': 16,
        },

        'net_class': Net,
        'net': {
            'in_dim': 6,
            'out_dim': 16,
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
            'loss_class': Loss,
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
                'batch_size': args.train_batch_size,
                'pin_memory': False,
                'num_workers': 0,
                'shuffle': False,
            },

            # frequency of validation step
            'freq_val': 40,
            # total number of epochs
            'n_epochs': args.goal_epoch,
        },

        'test': {
            'loss_class': Loss,
            'loss': {
                'min_N': 4, # int(np.log2(dataset_params['min_train_freq'])),
                'max_N': 5, # int(np.log2(dataset_params['max_train_freq'])),
                'w':  1e6,
                'huber': 0.005,
                'dt': 0.005,
                'dv': args.dv,
                'dv_normed': args.dv_normed,
            },
            'dataloader': {
                'batch_size': 1,
                'pin_memory': False,
                'num_workers': 0,
                'shuffle': False,
            },
        },

        'ncp': {
            # Number of inter neurons
            'inter_neurons': args.n_inter,
            # Number of command neurons
            'command_neurons': args.n_command,
            # Number of motor neurons
            'motor_neurons': 3,
            # How many outgoing synapses has each sensory neuron
            'sensory_fanout': args.out_sensory,
            # How many outgoing synapses has each inter neuron
            'inter_fanout': args.out_inter,
            # Now many recurrent synapses are in the command neuron layer
            'recurrent_command_synapses': args.rec_command,
            # How many incoming synapses has each motor neuron
            'motor_fanin': args.in_motor,
        },
    }

    return params
