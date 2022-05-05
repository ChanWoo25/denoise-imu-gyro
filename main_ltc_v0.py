import os
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

import kerasncp as kncp
from kerasncp.torch import LTCCell
from kerasncp import wirings
print('kncp version:', kncp.__version__)

from src.DGAProcess import LearningProcess
from src.DGALoss import DGALoss
from src.DGANet import DGANet
from src.DgaSequence import DgaRawSequence, DgaWinSequence
from src.DgaDataset import DgaDataset
from src.DgaPreNet import DgaPreNet

################


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

euroc_data_dir = os.path.join('/root', 'project', 'datasets', 'EUROC')
ltc_results_dir = os.path.join('/root', 'project', 'results', 'LTC')

params = {
    'debug': args.debug,
    'result_dir': os.path.join(ltc_results_dir, args.id),
    'test_dir': os.path.join(ltc_results_dir, args.id, 'tests'),
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
        'data_dir':     euroc_data_dir,
        'predata_dir':  os.path.join(ltc_results_dir, 'predata'),
        'N': 32 * 100, # should be integer * 'max_train_freq'
        'min_train_freq': 16,
        'max_train_freq': 32,
        'v_window': 16,
    },

    'net_class': DGANet,
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
            'batch_size': 2,
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

# LightningModule for training a RNNSequence module
class SequenceLearner(pl.LightningModule):
    def __init__(self, model, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        us, gt = batch

        print('us:', us.shape, us.device)
        print('gt:', gt.shape, gt.device)
        q_gt = gt[:, :, 4:8]

        q_hat = self.model.forward(us)
        print('q_hat:', q_hat.shape, q_hat.device)
        print('q_gt:', q_gt.shape, q_gt.device)

        q_hat = q_hat.view_as(q_gt)
        loss = nn.MSELoss()(q_hat, q_gt)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        us, gt = batch
        q_gt = gt[:, :, 4:8]
        q_hat = self.model.forward(us)
        q_hat = q_hat.view_as(q_gt)
        loss = nn.MSELoss()(q_hat, q_gt)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def optimizer_step(
        self,
        current_epoch,
        batch_nb,
        optimizer,
        optimizer_idx,
        closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        optimizer.optimizer.step(closure=closure)
        # Apply weight constraints
        self.model.rnn_cell.apply_weight_constraints()


in_features = 6
out_features = 4
N = 16000  # Length of the time-series

# Input feature is a sine and a cosine wave
data_x = np.stack(
    [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))], axis=1
)
data_x = np.expand_dims(data_x, axis=0).astype(np.float32)  # Add batch dimension
# Target output is a sine with double the frequency of the input signal
data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]).astype(np.float32)
data_x = torch.Tensor(data_x)
data_y = torch.Tensor(data_y)
print("data_x.size: ", str(data_x.size()))
print("data_y.size: ", str(data_y.size()))


dataset_train = DgaDataset(params, mode='train')
dataset_test  = DgaDataset(params, mode='test')
dataloader_train = data.DataLoader(dataset_train, **params['train']['dataloader'])
dataloader_test = data.DataLoader(dataset_test, **params['train']['dataloader'])

#######################
ncp_wiring = kncp.wirings.NCP(
    inter_neurons=20,  # Number of inter neurons
    command_neurons=10,  # Number of command neurons
    motor_neurons=4,  # Number of motor neurons
    sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
    inter_fanout=5,  # How many outgoing synapses has each inter neuron
    recurrent_command_synapses=6,  # Now many recurrent synapses are in the
    # command neuron layer
    motor_fanin=4,  # How many incoming synapses has each motor neuron
)
ncp_cell = LTCCell(ncp_wiring, in_features)

dga_pre_net = DgaPreNet(params).cuda()
#######################

# wiring = kncp.wirings.FullyConnected(8, out_features)  # 16 units, 8 motor neurons
# ltc_cell = LTCCell(wiring, in_features)

ltc_sequence = DgaRawSequence(
    ncp_cell
)
# ltc_sequence = DgaWinSequence(
#     ncp_cell,
#     dga_pre_net
# )

learn = SequenceLearner(ltc_sequence, lr=0.01)
trainer = pl.Trainer(
    logger=pl.loggers.CSVLogger("log"),
    max_epochs=400,
    progress_bar_refresh_rate=1,
    gradient_clip_val=1,  # Clip gradient to stabilize training
    gpus=1,
)

trainer.fit(learn, dataloader_train)

results = trainer.test(learn, dataloader_test)

