from re import L
import numpy as np
import torch
from src.utils import bmmt, bmv, bmtv, bbmv, bmtm
from src.lie_algebra import SO3
import matplotlib.pyplot as plt

class DGALoss(torch.nn.Module):
    """Loss for low-frequency orientation increment"""

    def __init__(self, w, min_N, max_N, dt, huber, v_window):
        super().__init__()
        # windows sizes
        self.min_N = min_N
        self.max_N = max_N
        self.min_train_freq = 2 ** self.min_N
        self.max_train_freq = 2 ** self.max_N
        self.v_window = v_window

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

    def re_batch(self, x:torch.Tensor, padding_size, note=None):
        """2padding_size만큼의 길이를 갖는 윈도우 사이즈의 미니 배치를 x.shape[0]개 만큼 만들어 반환한다.
            2차원 -> 3차원
            즉, when x == [N, 6] => x_rebatch == [N, pad_size, 6]
        """
        N = x.shape[0]
        k = x.shape[-1]

        pad = x[0].expand(padding_size, k).clone()
        if note == 'accel':
            pad[:,3:6] = 0.0

        x = torch.cat([pad, x], dim=0)
        x_arr = []
        for i in range(padding_size+1):
            x_arr.append(x[i:i+N])
        x = torch.stack(x_arr, dim=1)

        assert x.shape == torch.Size([N, padding_size+1, k])
        return x

    def forward(self, w_hat, a_hat, xs, dv, vs_gt_norm, mode='train'):
        """Forward errors with rotation matrices"""
        batch_size = int(w_hat.shape[0])
        N = int(w_hat.shape[1])
        padding_size = self.v_window - 1


        ### Gyro Loss
        Xs = SO3.exp(xs[:, ::self.min_train_freq].reshape(-1, 3).double())
        # print("xs:", xs.shape, xs.dtype)
        # xs: torch.Size([6, 16000, 3]) torch.float32
        w_hat = self.dt*w_hat.reshape(-1, 3).double()
        Omegas = SO3.exp(w_hat[:, :3])

        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
        rs = SO3.log(bmtm(Omegas, Xs)).reshape(batch_size, -1, 3)[:, self.N0:]
        gyro_loss_16 = self.f_huber(rs)

        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
            Xs = Xs[::2].bmm(Xs[1::2])
            rs = SO3.log(bmtm(Omegas, Xs)).reshape(batch_size, -1, 3)[:, self.N0:]
            gyro_loss_32 = self.f_huber(rs)/(2**(k - self.min_N + 1))
        ###

        ### Accel Loss
        ## a_hat을 511개 쌓아서 만든 새로운 _vs 생성
        _vs = torch.zeros_like(a_hat).cuda()
        _vs_norm = []
        _dv_hat = ((a_hat[:, 1:] + a_hat[:, :-1]) * self.dt).cuda() # torch.Size([6, 15999, 3])
        for i in range(1, N):
            _vs[:, i:] += _dv_hat[:, 0:N-i]
        _vs = torch.cat([_vs[:, 0, :].unsqueeze(1).expand(batch_size, padding_size, 3), _vs], dim=1) # torch.Size([6, 16510, 3])
        for i in range(N):
            window_batch = _vs[:, i:i+padding_size+1] # torch.Size([6, 511, 3])
            _std, _mean = torch.std_mean(window_batch, unbiased=False, dim=1)
            _normalized = (window_batch[:, -1, :] - _mean) # torch.Size([6, 3]) # 220414 평균만 제거 하는 쪽으로 선회
            _normalized = _normalized.unsqueeze(1)                # torch.Size([6, 1, 3])
            if i == 0:
                _vs_norm.append(torch.zeros_like(_normalized, dtype=torch.float32))
            else:
                _vs_norm.append(_normalized)
        _vs_norm = torch.cat(_vs_norm, dim=1) # torch.Size([6, 16000, 3])
        # print("_vs_norm:", _vs_norm.shape, _vs_norm.dtype)
        # print('vs_gt_norm:', vs_gt_norm.shape)
        acc_norm_loss = (vs_gt_norm - _vs_norm) ** 2 # torch.Size([6, 16000, 3])
        acc_norm_loss = acc_norm_loss.mean()

        if mode is 'val':
            print('loss: gyro16(%4.4f), gyro32(%4.4f), accel(%4.4f)'  % (gyro_loss_16, gyro_loss_32, acc_norm_loss))



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
        return (gyro_loss_16 + gyro_loss_32 + acc_norm_loss)


