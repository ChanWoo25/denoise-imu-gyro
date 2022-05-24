import torch
import numpy as np
from src.utils import bmtm, vnorm, fast_acc_integration
from src.lie_algebra import SO3

class DGLossVer1(torch.nn.Module):
    """Loss for low-frequency orientation increment"""

    def __init__(self, params): # w, min_N, max_N, dt, huber
        super().__init__()

        # windows sizes
        self.min_N = params['train']['loss']['min_N']
        self.max_N = params['train']['loss']['max_N']
        self.min_train_freq = 2 ** self.min_N
        self.max_train_freq = 2 ** self.max_N

        # sampling time
        self.dt = params['train']['loss']['dt'] # (s)

        # weights on loss
        self.w = params['train']['loss']['w']
        self.sl = torch.nn.SmoothL1Loss()
        self.sln = torch.nn.SmoothL1Loss(reduction='none')

        self.huber = params['train']['loss']['huber']
        self.weight = torch.ones(1, 1, self.min_train_freq).cuda() / self.min_train_freq
        self.N0 = 5 # remove first N0 increment in loss due not account padding

        self.dv        = params['train']['loss']['dv']
        self.dv_normed = params['train']['loss']['dv_normed']

    def f_huber(self, rs):
        """Huber loss function"""
        loss = self.w * self.sl(rs/self.huber, torch.zeros_like(rs)) * (self.huber**2)
        return loss

    def gyro_loss(self, w_hat, dw_16):
        """Forward errors with rotation matrices"""
        N = dw_16.shape[0]
        drot_16 = SO3.exp(dw_16[:, ::self.min_train_freq].reshape(-1, 3).double())
        dw_hat = self.dt * w_hat.reshape(-1, 3).double()
        drot_hat = SO3.exp(dw_hat)

        for k in range(self.min_N):
            drot_hat = drot_hat[::2].bmm(drot_hat[1::2])
        rs = SO3.log(bmtm(drot_hat, drot_16)).reshape(N, -1, 3)[:, self.N0:]
        gyro_loss_16 = self.f_huber(rs)

        ## compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            drot_hat = drot_hat[::2].bmm(drot_hat[1::2])
            drot_32 = drot_16[::2].bmm(drot_16[1::2])
            rs = SO3.log(bmtm(drot_hat, drot_32)).reshape(N, -1, 3)[:, self.N0:]
            gyro_loss_32 = self.f_huber(rs)/(2**(k - self.min_N + 1)) / 2.0

        return gyro_loss_16, gyro_loss_32

    def forward(self, w_hat, dw_16, show=False):
        gyro16, gyro32 = self.gyro_loss(w_hat, dw_16)
        if show:
            print('Gyro 16 Loss:', gyro16.item())
            print('Gyro 32 Loss:', gyro32.item())
        return gyro16 + gyro32



class DGLossVer2(torch.nn.Module):
    """Loss for low-frequency orientation increment"""

    def __init__(self, params): # w, min_N, max_N, dt, huber
        super().__init__()

        # windows sizes
        self.min_N = params['train']['loss']['min_N']
        self.max_N = params['train']['loss']['max_N']
        self.min_train_freq = 2 ** self.min_N
        self.max_train_freq = 2 ** self.max_N

        # sampling time
        self.dt = params['train']['loss']['dt'] # (s)

        # weights on loss
        self.w = params['train']['loss']['w']
        self.sl = torch.nn.SmoothL1Loss()
        self.sln = torch.nn.SmoothL1Loss(reduction='none')

        self.huber = params['train']['loss']['huber']
        self.weight = torch.ones(1, 1, self.min_train_freq).cuda() / self.min_train_freq
        self.N0 = 5 # remove first N0 increment in loss due not account padding

        self.dv             = params['train']['loss']['dv']
        self.dv_normed      = params['train']['loss']['dv_normed']

    def f_huber(self, rs):
        """Huber loss function"""
        loss = self.w * self.sl(rs/self.huber, torch.zeros_like(rs)) * (self.huber**2)
        return loss

    def gyro_loss(self, w_hat, dw_16):
        """Forward errors with rotation matrices"""
        N = dw_16.shape[0]
        drot_16 = SO3.exp(dw_16[:, ::self.min_train_freq].reshape(-1, 3).double())
        dw_hat = self.dt * w_hat.reshape(-1, 3).double()
        drot_hat = SO3.exp(dw_hat)

        for k in range(self.min_N):
            drot_hat = drot_hat[::2].bmm(drot_hat[1::2])
        rs = SO3.log(bmtm(drot_hat, drot_16)).reshape(N, -1, 3)[:, self.N0:]
        gyro_loss_16 = self.f_huber(rs)

        ## compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            drot_hat = drot_hat[::2].bmm(drot_hat[1::2])
            drot_32 = drot_16[::2].bmm(drot_16[1::2])
            rs = SO3.log(bmtm(drot_hat, drot_32)).reshape(N, -1, 3)[:, self.N0:]
            gyro_loss_32 = self.f_huber(rs)/(2**(k - self.min_N + 1)) / 2.0

        return gyro_loss_16, gyro_loss_32

    def gaussian_nll_loss(self, w_hat, w_gt, w_mean, w_std):
        eps = 1e-6
        w_var = w_std ** 2
        w_var[w_var<eps] = eps
        w_gap = w_gt - w_hat
        _first = torch.log(w_var)
        _second = (w_gap - w_mean)**2 / w_var
        _loss = (_first + _second) / 2.0
        return torch.mean(_loss)

    def forward(self, w_hat, dw_16, w_gt, w_mean, w_std, show=False):
        gyro16, gyro32 = self.gyro_loss(w_hat, dw_16)
        gnll = self.gaussian_nll_loss(w_hat, w_gt, w_mean, w_std)
        if show:
            print('Gyro 16 Loss:', gyro16.item())
            print('Gyro 32 Loss:', gyro32.item())
            print('Gaussian Loss:', gnll.item())
        return gyro16 + gyro32 + gnll
