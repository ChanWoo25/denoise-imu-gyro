import os
from unittest import result
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt

import kerasncp as kncp
from kerasncp.torch import LTCCell
from kerasncp import wirings

print('kncp version:', kncp.__version__)

from src.DGAProcess import LearningProcess
from src.DgaLoss import DgaLoss
from src.DGANet import DGANet
from src.DgaSequence import DgaRawSequence, DgaWinSequence
from src.DgaDataset import DgaDataset
from src.DgaPreNet import DgaPreNet
from src.lie_algebra import SO3
from src.utils import bmtm, vnorm, fast_acc_integration
################


import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--machine', type=str, default='desktop')
parser.add_argument('--id', type=str, default='test')
parser.add_argument('--input_type', type=str, default='window')
parser.add_argument('--resume_path', type=str, default=None)
parser.add_argument('--test_path', type=str, default=None)
parser.add_argument('--c0', type=int, default=16)
parser.add_argument('--train_batch_size', type=int, default=6)
parser.add_argument('--seq_len', type=int, default=3200)
parser.add_argument('--goal_epoch', type=int, default=400)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--dv', nargs='+', type=int, default=[16, 32])
parser.add_argument('--dv_normed', nargs='+', type=int, default=[32, 64])


# Wiring Hyper paramters
parser.add_argument('--n_inter', type=int, default=20)
parser.add_argument('--n_command', type=int, default=10)
# parser.add_argument('--n_motor', type=int, default=0)
parser.add_argument('--out_sensory', type=int, default=4)
parser.add_argument('--out_inter', type=int, default=5)
parser.add_argument('--rec_command', type=int, default=6)
parser.add_argument('--in_motor', type=int, default=4)
# LTC Cell Hyper parameters
parser.add_argument('--ode_unfolds', type=int, default=6)
# Parse
args = parser.parse_args()
print(args.__dict__)

euroc_data_dir = os.path.join('/root', 'project', 'datasets', 'EUROC')
ltc_results_dir = os.path.join('/root', 'project', 'results', 'LTC')
if args.machine == 'server':
    euroc_data_dir = os.path.join('/home/leecw', 'project', 'datasets', 'EUROC')
    ltc_results_dir = os.path.join('/home/leecw', 'project', 'results', 'LTC')


params = {
    'debug': args.debug,
    'result_dir': os.path.join(ltc_results_dir, args.id),
    'test_dir': os.path.join(ltc_results_dir, args.id, 'tests'),
    'figure_dir': os.path.join(ltc_results_dir, 'figures'),
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
        'N': args.seq_len, # should be integer * 'max_train_freq'
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
        'loss_class': DgaLoss,
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
        'n_epochs': 1000,
    },

    'test': {
        'optimizer_class': torch.optim.Adam,
        'optimizer': {
            'lr': args.lr,
            'weight_decay': 1e-1,
            'amsgrad': False,
        },
        'loss_class': DgaLoss,
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
            'batch_size': 1,
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

# def integrate_with_quaternions_superfast(w:torch.Tensor):

#     qs = None
#     if w.shape[-1] == 3:
#         qs = SO3.qnorm(SO3.exp(w) * 0.005)

#     imu_qs = SO3.qnorm(SO3.qexp(raw_us[:, :3].cuda().double() * 0.005))
#     net_qs = SO3.qnorm(SO3.qexp(net_us[:, :3].cuda().double() * 0.005))
#     Rot0 = SO3.qnorm(quat_gt[:2].cuda().double())
#     imu_qs[0] = Rot0[0]
#     net_qs[0] = Rot0[0]

#     N = np.log2(imu_qs.shape[0])
#     for i in range(int(N)):
#         k = 2**i
#         imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:-k], imu_qs[k:]))
#         net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:-k], net_qs[k:]))

#     if int(N) < N:
#         k = 2**int(N)
#         k2 = imu_qs[k:].shape[0]
#         # print("imu_qs: ", imu_qs.shape)
#         # print("k: %d, k2: %d"%(k, k2))
#         # print("imu_qs: qmul with %d x %d" % (imu_qs[:k2].shape[0], imu_qs[k:].shape[0]))
#         # print("net_qs: qmul with %d x %d" % (net_qs[:k2].shape[0], net_qs[k:].shape[0]))
#         imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:k2], imu_qs[k:]))
#         net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:k2], net_qs[k:]))

#     imu_Rots = SO3.from_quaternion(imu_qs).float()
#     net_Rots = SO3.from_quaternion(net_qs).float()
#     return net_qs.cpu(), imu_Rots, net_Rots


# LightningModule for training a RNNSequence module
class SequenceLearner(pl.LightningModule):
    def __init__(self, model, loss, lr, nf):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.figsize = (20, 12)
        self.dt = 0.005 # (s)
        self.nf = nf

        # Loss
        self.w = params['train']['loss']['w']
        self.sl = torch.nn.SmoothL1Loss()
        self.sln = torch.nn.SmoothL1Loss(reduction='none')
        self.huber = params['train']['loss']['huber']

    def f_huber(self, rs):
        """Huber loss function"""
        loss = self.w * self.sl(rs/self.huber, torch.zeros_like(rs)) * (self.huber**2)
        return loss

    def training_step(self, batch, batch_idx):
        us, dw_16_gt, dw_32_gt = batch
        us = us.float()
        dw_16_gt = dw_16_gt.float()
        dw_32_gt = dw_32_gt.float()

        self.model.set_nf(nf['train']['mean'], nf['train']['std'])
        w_hat = self.model.forward(us)

        loss = self.loss(w_hat, dw_16_gt, dw_32_gt)

        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        seq, ts, us, dw_16_gt, dw_32_gt, q_gt = batch
        seq = seq[0]
        ts = ts.float()
        us = us.float()
        dw_16_gt = dw_16_gt.float()
        dw_32_gt = dw_32_gt.float()
        q_gt = q_gt.float()

        self.model.set_nf(nf['test']['mean'], nf['test']['std'])
        w_hat = self.model.forward(us)

        loss = self.loss(w_hat, dw_16_gt, dw_32_gt)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        seq, ts, us, dw_16_gt, dw_32_gt, q_gt = batch
        seq = seq[0]
        ts = ts.float()
        us = us.float()
        dw_16_gt = dw_16_gt.float()
        dw_32_gt = dw_32_gt.float()
        q_gt = q_gt.float()

        self.model.set_nf(nf['test']['mean'], nf['test']['std'])
        w_hat = self.model.forward(us)

        loss = self.loss(w_hat, dw_16_gt, dw_32_gt)

        ### Visualize
        q_hat = SO3.qnorm(SO3.qexp(w_hat.squeeze().double() * self.dt))
        N = np.log2(q_hat.shape[0])
        for i in range(int(N)):
            k = 2**i
            q_hat[k:] = SO3.qnorm(SO3.qmul(q_hat[:-k], q_hat[k:]))

        if int(N) < N:
            k = 2**int(N)
            k2 = q_hat[k:].shape[0]
            q_hat[k:] = SO3.qnorm(SO3.qmul(q_hat[:k2], q_hat[k:]))

        rot_hat = SO3.from_quaternion(q_hat)
        rpy_hat = SO3.to_rpy(rot_hat).cpu().numpy()

        rot_gt = SO3.from_quaternion(q_gt.squeeze().double())
        rpy_gt = -SO3.to_rpy(rot_gt).cpu().numpy() # TODO: Resolve reflection issue

        offset = rpy_gt[0] - rpy_hat[0]
        rpy_hat += offset

        def rad2deg(x):
            return x * (180. / np.pi)

        self.plot_orientation(seq, rad2deg(rpy_hat), rad2deg(rpy_gt))
        ###

        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **params['train']['scheduler'])
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [scheduler]

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


    def plot_orientation(self, seq, rpy_hat:np.ndarray, rpy_gt:np.ndarray):
        figure_dir = os.path.join(params['figure_dir'], seq, 'rpy')
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize, dpi=250)
        fig.suptitle('$SO(3)$ Orientation Estimation / %s / %s' % (seq, args.id), fontsize=20)

        axs[0].set(ylabel='roll (deg)', title='Orientation estimation')
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        N = rpy_hat.shape[0]
        ts = list(range(N))
        ts = np.array(ts) * 0.005
        rpy_gt = rpy_gt[:N]
        for i in range(3):
            axs[i].plot(ts, rpy_hat[:, i]%360, color='blue', label=r'net IMU')
            axs[i].plot(ts, rpy_gt[:, i]%360, color='black', label=r'ground truth')
            axs[i].set_xlim(ts[0], ts[-1])

        _path = os.path.join(figure_dir, args.id + '.png')
        self.savefig(axs, fig, _path)
        plt.close(fig)

    def savefig(self, axs, fig, path):
        if isinstance(axs, np.ndarray):
            for i in range(len(axs)):
                axs[i].grid()
                axs[i].legend()
        else:
            axs.grid()
            axs.legend()
        fig.savefig(path)

#####
# N = 16000  # Length of the time-series
# # Input feature is a sine and a cosine wave
# data_x = np.stack(
#     [np.sin(np.linspace(0, 3 * np.pi, N)), np.cos(np.linspace(0, 3 * np.pi, N))], axis=1
# )
# data_x = np.expand_dims(data_x, axis=0).astype(np.float32)  # Add batch dimension
# # Target output is a sine with double the frequency of the input signal
# data_y = np.sin(np.linspace(0, 6 * np.pi, N)).reshape([1, N, 1]).astype(np.float32)
# data_x = torch.Tensor(data_x)
# data_y = torch.Tensor(data_y)
# print("data_x.size: ", str(data_x.size()))
# print("data_y.size: ", str(data_y.size()))
#####

dataset_train = DgaDataset(params, mode='train')
dataset_test  = DgaDataset(params, mode='test')
dataloader_train = data.DataLoader(dataset_train, **params['train']['dataloader'])
dataloader_test = data.DataLoader(dataset_test, **params['test']['dataloader'])
train_mean, train_std = dataset_train.get_dataset_nf()
test_mean, test_std = dataset_test.get_dataset_nf()
nf = {
    'train': {
        'mean': train_mean,
        'std': train_std
    },
    'test':{
        'mean': test_mean,
        'std': test_std
    }
}

loss = DgaLoss(params)

#######################
in_features = 16 if args.input_type == 'window' else 6
out_features = 3
ncp_wiring = kncp.wirings.NCP(
    # Number of inter neurons
    inter_neurons=args.n_inter,
    # Number of command neurons
    command_neurons=args.n_command,
    # Number of motor neurons
    motor_neurons=out_features,
    # How many outgoing synapses has each sensory neuron
    sensory_fanout=args.out_sensory,
    # How many outgoing synapses has each inter neuron
    inter_fanout=args.out_inter,
    # Now many recurrent synapses are in the command neuron layer
    recurrent_command_synapses=args.rec_command,
    # How many incoming synapses has each motor neuron
    motor_fanin=args.in_motor,
)
ncp_cell = LTCCell(ncp_wiring, in_features, ode_unfolds=args.ode_unfolds)
dga_pre_net = DgaPreNet(params).cuda()
#######################

# wiring = kncp.wirings.FullyConnected(8, out_features)  # 16 units, 8 motor neurons
# ltc_cell = LTCCell(wiring, in_features)

ltc_sequence = None
if args.input_type == 'raw':
    ltc_sequence = DgaRawSequence(ncp_cell)
elif args.input_type == 'window':
    ltc_sequence = DgaWinSequence(ncp_cell, dga_pre_net)

learn = SequenceLearner(ltc_sequence, loss, lr=params['train']['optimizer']['lr'], nf=nf)

# Checkpointing
from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    dirpath='./log/%s/'%args.id,
    filename='{epoch}-{val_loss:.2f}',
    save_last=True,
    save_top_k=10
)

trainer = pl.Trainer(
    logger=pl.loggers.CSVLogger(ltc_results_dir, name=args.id),
    max_epochs=args.goal_epoch,
    progress_bar_refresh_rate=1,
    gradient_clip_val=1,  # Clip gradient to stabilize training
    gpus=1,
    check_val_every_n_epoch=10,
    log_every_n_steps=10,
    num_sanity_val_steps=0,
    callbacks=[checkpoint_callback]
)

if args.mode == 'train' or args.mode == 'both':
    trainer.fit(learn, dataloader_train, dataloader_test, ckpt_path=args.resume_path)

if args.mode == 'test' or args.mode == 'both':
    results = trainer.test(learn, dataloader_test, ckpt_path=args.test_path)
    print(type(results))

