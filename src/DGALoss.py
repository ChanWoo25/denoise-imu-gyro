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

    def forward(self, xs, w_hat):
        """Forward errors with rotation matrices"""
        N = xs.shape[0]
        Xs = SO3.exp(xs[:, ::self.min_train_freq].reshape(-1, 3).double())
        w_hat = self.dt*w_hat.reshape(-1, 3).double()
        Omegas = SO3.exp(w_hat[:, :3])

        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
        rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
        loss = self.f_huber(rs)

        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
            Xs = Xs[::2].bmm(Xs[1::2])
            rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
            loss = loss + self.f_huber(rs)/(2**(k - self.min_N + 1))

        return loss

