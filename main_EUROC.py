import os
from unittest import result
import torch
import src.learning as lr
import src.networks as sn
import src.losses as sl
import src.dataset as ds
import numpy as np


import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--is_train', type=int, default=False)
parser.add_argument('--id', type=str, default=None)
args = parser.parse_args()
print(args.__dict__)

import sys
data_dir = "data"
result_dir = "result"
if sys.platform.startswith('win'):
    data_dir = "data"
    result_dir = "result"
elif sys.platform.startswith('linux'):
    data_dir = "/root/Data/EUROC"
    result_dir = "/root/Data/Result/DenoiseIMU"
id = args.id
# load_id = "220315"

address = os.path.join(result_dir, id)
# pretrained = os.path.join(result_dir, load_id)


# test a given network
# or test the last trained network
# address = "last"
################################################################################
# Network parameters
################################################################################
net_class = sn.GyroNet
net_params = {
    'in_dim': 6,
    'out_dim': 3,
    'c0': 16,
    'dropout': 0.1,
    'ks': [7, 7, 7, 7],
    'ds': [4, 4, 4],
    'momentum': 0.1,
    'gyro_std': [1*np.pi/180, 2*np.pi/180, 5*np.pi/180],
}
################################################################################
# Dataset parameters
################################################################################
dataset_class = ds.EUROCDataset
dataset_params = {
    # where are raw data ?
    'data_dir': data_dir,
    # where record preloaded data ?
    'predata_dir': result_dir,
    # set train, val and test sequence
    'train_seqs': [
        'MH_01_easy',
        'MH_03_medium',
        'MH_05_difficult',
        'V1_02_medium',
        'V2_01_easy',
        'V2_03_difficult'
        ],
    'val_seqs': [
        'MH_01_easy',
        'MH_03_medium',
        'MH_05_difficult',
        'V1_02_medium',
        'V2_01_easy',
        'V2_03_difficult',
        ],
    'test_seqs': [
        'MH_02_easy',
        'MH_04_difficult',
        'V2_02_medium',
        'V1_03_difficult',
        'V1_01_easy',
        ],
    # size of trajectory during training
    'N': 32 * 500, # should be integer * 'max_train_freq'
    'min_train_freq': 16,
    'max_train_freq': 32,
}
################################################################################
# Training parameters
################################################################################
train_params = {
    'is_train':True,
    'optimizer_class': torch.optim.Adam,
    'optimizer': {
        'lr': 0.01,
        'weight_decay': 1e-1,
        'amsgrad': False,
    },
    'loss_class': sl.GyroLoss,
    'loss': {
        'min_N': int(np.log2(dataset_params['min_train_freq'])),
        'max_N': int(np.log2(dataset_params['max_train_freq'])),
        'w':  1e6,
        'target': 'rotation matrix',
        'huber': 0.005,
        'dt': 0.005,
    },
    'scheduler_class': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'scheduler': {
        'T_0': 600,
        'T_mult': 2,
        'eta_min': 1e-3,
    },
    'dataloader': {
        'batch_size': 10,
        'pin_memory': False,
        'num_workers': 0,
        'shuffle': False,
    },
    # frequency of validation step
    'freq_val': 600,
    # total number of epochs
    'n_epochs': 6000,
    # where record results ?
    'res_dir': address,
    # where record Tensorboard log ?
    'tb_dir': address,
}

test_params = {
    'is_train':False,
    'optimizer_class': torch.optim.Adam,
    'optimizer': {
        'lr': 0.01,
        'weight_decay': 1e-1,
        'amsgrad': False,
    },
    'loss_class': sl.GyroLoss,
    'loss': {
        'min_N': int(np.log2(dataset_params['min_train_freq'])),
        'max_N': int(np.log2(dataset_params['max_train_freq'])),
        'w':  1e6,
        'target': 'rotation matrix',
        'huber': 0.005,
        'dt': 0.005,
    },
    'scheduler_class': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'scheduler': {
        'T_0': 600,
        'T_mult': 2,
        'eta_min': 1e-3,
    },
    'dataloader': {
        'batch_size': 10,
        'pin_memory': False,
        'num_workers': 0,
        'shuffle': False,
    },
    # frequency of validation step
    'freq_val': 600,
    # total number of epochs
    'n_epochs': 1800,
    # where record results ?
    'res_dir': address,
    # where record Tensorboard log ?
    'tb_dir': address,
}


if __name__ == '__main__':
    if args.is_train:
        print("Train")
        if os.path.exists(address):
            pass
            # print("[FATAL] -- Already trained");exit(1)
        else:
            os.mkdir(address)
        learning_process = lr.GyroLearningBasedProcessing(train_params, net_class, net_params, address, train_params['loss']['dt'])
        learning_process.train(dataset_class, dataset_params)
    else:
        print("Test")
        if not os.path.exists(address):
            print("[FATAL] -- There is no pretrained");exit(1)
        learning_process = lr.GyroLearningBasedProcessing(test_params, net_class, net_params, address, dt=train_params['loss']['dt'])
        learning_process.test(dataset_class, dataset_params)
