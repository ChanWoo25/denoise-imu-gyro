import numpy as np
import torch
from src.utils import bmmt, bmv, bmtv, bbmv, bmtm
from src.lie_algebra import SO3
import matplotlib.pyplot as plt

class DGALoss(torch.nn.Module):
    """Loss for low-frequency orientation increment"""

    def __init__(self, w, min_N, max_N, dt, huber):
        super().__init__()
        # windows sizes
        self.min_N = min_N
        self.max_N = max_N
        self.min_train_freq = 2 ** self.min_N
        self.max_train_freq = 2 ** self.max_N

        # sampling time
        self.dt = dt # (s)

        # weights on loss
        self.w = w
        self.sl = torch.nn.SmoothL1Loss()

        self.huber = huber
        self.weight = torch.ones(1, 1, self.min_train_freq).cuda() / self.min_train_freq
        self.N0 = 5 # remove first N0 increment in loss due not account padding

    def f_huber(self, rs):
        """Huber loss function"""
        loss = self.w * self.sl(rs/self.huber, torch.zeros_like(rs)) * (self.huber**2)
        return loss

    def forward(self, w_hat, a_hat, xs, dv):
        """Forward errors with rotation matrices"""
        # print("DGA Loss :: forward() :: Debugging")
        N = xs.shape[0]

        ### Gyro Loss
        Xs = SO3.exp(xs[:, ::self.min_train_freq].reshape(-1, 3).double())
        # print("xs:", xs.shape, xs.dtype)
        # xs: torch.Size([6, 16000, 3]) torch.float32
        w_hat = self.dt*w_hat.reshape(-1, 3).double()
        Omegas = SO3.exp(w_hat[:, :3])

        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
        rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
        gyro_loss_16 = self.f_huber(rs)

        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
            Xs = Xs[::2].bmm(Xs[1::2])
            rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
            gyro_loss_32 = self.f_huber(rs)/(2**(k - self.min_N + 1))
        ###

        ### Accel Loss
        dv = dv[:, ::self.min_train_freq].reshape(-1, 3).double()
        # print("dv:", dv.shape, dv.dtype)
        # print("a_hat:", a_hat.shape, a_hat.dtype)
        # dv: torch.Size([6000, 3]) torch.float64
        # a_hat: torch.Size([6, 16000, 3]) torch.float32
        dv_hat = self.dt * a_hat.reshape(-1, 3).double()
        # print("dv_hat:", dv_hat.shape, dv_hat.dtype)
        # torch.Size([96000, 3]) torch.float64

        for k in range(self.min_N): # 4
            dv_hat = dv_hat[::2] + dv_hat[1::2]
        # print("dv_hat:", dv_hat.shape, dv_hat.dtype)
        # dv_hat: torch.Size([6000, 3]) torch.float64
        acc_loss_16 = self.sl(dv, dv_hat) * 10.0

        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            dv_hat = dv_hat[::2] + dv_hat[1::2]
            dv = dv[::2] + dv[1::2]
            acc_loss_32 = self.sl(dv, dv_hat) * 10.0
        ###
        # print("Gyro Loss16:", gyro_loss_16.item())
        # print("Gyro Loss32:", gyro_loss_32.item())
        # print("Acc  Loss16:",  acc_loss_16.item())
        # print("Acc  Loss32:",  acc_loss_32.item())

        ratio = np.array([gyro_loss_16.item(), gyro_loss_32.item(), acc_loss_16.item(), acc_loss_16.item()])
        ratio /= ratio.sum()
        ratio *= 100.0
        # print("Ratio :: (%.2f, %.2f, %.2f, %.2f)" % (ratio[0], ratio[1], ratio[2], ratio[3]))

        """ -- Training --
            DGA Loss :: forward() :: Debugging
            xs: torch.Size([6, 16000, 3]) torch.float32
            dv: torch.Size([6000, 3]) torch.float64
            a_hat: torch.Size([6, 16000, 3]) torch.float32
            dv_hat: torch.Size([96000, 3]) torch.float64
            dv_hat: torch.Size([6000, 3]) torch.float64
            Gyro Loss16: 0.09049987956560995
            Gyro Loss32: 0.11279626840789501
            Acc  Loss16: 0.0027867559808852733
            Acc  Loss32: 0.010718861934038657
            Ratio :: (43.33, 54.00, 1.33, 1.33)
        """
        return (gyro_loss_16 + gyro_loss_32 + acc_loss_16 + acc_loss_32)


