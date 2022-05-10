import torch
import numpy as np
from src.utils import bmtm, vnorm, fast_acc_integration
from src.lie_algebra import SO3

class DgaLoss(torch.nn.Module):
    """Loss for low-frequency orientation increment"""

    def __init__(self, params): # w, min_N, max_N, dt, huber
        super().__init__()
        self.dt     = 0.005
        self.w      = params['train']['loss']['w']
        self.huber  = params['train']['loss']['huber']
        self.sl = torch.nn.SmoothL1Loss()

    def f_huber(self, rs):
        """Huber loss function"""
        loss = self.w * self.sl(rs/self.huber, torch.zeros_like(rs)) * (self.huber**2)
        return loss

    def forward(self, w_hat, dw_16_gt, dw_32_gt):
        batch_size = w_hat.shape[0]

        ## Loss computation
        drot_16_gt = SO3.exp(dw_16_gt[:, ::16].reshape(-1, 3).double())
        drot_32_gt = SO3.exp(dw_32_gt[:, ::32].reshape(-1, 3).double())

        ## dw 16 Loss
        drot_hat = SO3.exp(self.dt * w_hat.reshape(-1, 3).double())
        for _ in range(4): # 2 ** 4 = 16
            drot_hat = drot_hat[::2].bmm(drot_hat[1::2])
        dw_16_loss = SO3.log(bmtm(drot_hat, drot_16_gt))
        dw_16_loss = dw_16_loss.reshape(batch_size, -1, 3)[:, 5:] # 처음 10개는 패딩 관련으로 무효 처리
        dw_16_loss = self.f_huber(dw_16_loss)

        ## dw 32 Loss
        drot_hat = drot_hat[::2].bmm(drot_hat[1::2])
        dw_32_loss = SO3.log(bmtm(drot_hat, drot_32_gt))
        dw_32_loss = dw_32_loss.reshape(batch_size, -1, 3)[:, 2:] # 처음 5개는 패딩 관련으로 무효 처리
        dw_32_loss = self.f_huber(dw_32_loss) / 2.0

        return dw_16_loss + dw_32_loss
