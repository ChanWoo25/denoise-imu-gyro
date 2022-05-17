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

from src.DgaLoss import DgaLoss
from src.DGANet import DGANet
from src.DgaSequence import DgaRawSequence, DgaWinSequence
from src.DgaDataset import DgaDataset
from src.DgaPreNet import DgaPreNet
from src.lie_algebra import SO3
from config.ltc import configure
################

params = configure()


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
        fig.suptitle('$SO(3)$ Orientation Estimation / %s / %s' % (seq, params['id']), fontsize=20)

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

        _path = os.path.join(figure_dir, params['id'] + '.png')
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



input_type = params['input_type']

#######################
in_features = 16 if input_type == 'window' else 6
out_features = 3
ncp_wiring = kncp.wirings.NCP(**params['ncp'])
ncp_cell = LTCCell(ncp_wiring, in_features, ode_unfolds=params['ode_unfolds'])
dga_pre_net = DgaPreNet(params).cuda()
#######################

# wiring = kncp.wirings.FullyConnected(8, out_features)  # 16 units, 8 motor neurons
# ltc_cell = LTCCell(wiring, in_features)

ltc_sequence = None
if input_type == 'raw':
    ltc_sequence = DgaRawSequence(ncp_cell)
elif input_type == 'window':
    ltc_sequence = DgaWinSequence(ncp_cell, dga_pre_net)

loss = params['loss_class'](params)
learner = SequenceLearner(
    model=ltc_sequence,
    loss=loss,
    lr=params['train']['optimizer']['lr'],
    nf=nf
)

## Checkpoint Callback
from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    dirpath='./log/%s/'%params['id'],
    filename='{epoch}-{val_loss:.2f}',
    save_last=True,
    save_top_k=10
)

trainer = pl.Trainer(
    logger=pl.loggers.CSVLogger(params['ltc_results_dir'], name=params['id']),
    max_epochs=params['goal_epoch'],
    progress_bar_refresh_rate=1,
    gradient_clip_val=1,  # Clip gradient to stabilize training
    gpus=1,
    check_val_every_n_epoch=10,
    log_every_n_steps=10,
    num_sanity_val_steps=0,
    callbacks=[checkpoint_callback]
)

if params['mode'] == 'train':
    trainer.fit(learner, dataloader_train, dataloader_test, ckpt_path=params['resume_path'])

elif params['mode'] == 'test':
    results = trainer.test(learner, dataloader_test, ckpt_path=params['test_path'])
    print(type(results))

