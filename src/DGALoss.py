import torch
import numpy as np
from src.utils import bmtm, vnorm, fast_acc_integration
from src.lie_algebra import SO3

class DGALoss(torch.nn.Module):
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
        # print('# GYRO LOSS:')
        N = dw_16.shape[0]
        drot_16 = SO3.exp(dw_16[:, ::self.min_train_freq].reshape(-1, 3).double())
        # print('- dw_16:', dw_16.shape, dw_16.shape, dw_16.device)

        # print('- w_hat:', w_hat.shape, w_hat.shape, w_hat.device)
        dw_hat = self.dt * w_hat.reshape(-1, 3).double()
        # print('- dw_hat:', dw_hat.shape, dw_hat.shape, dw_hat.device)
        drot_hat = SO3.exp(dw_hat)
        # print('- drot_hat:', drot_hat.shape, drot_hat.shape, drot_hat.device)

        for k in range(self.min_N):
            drot_hat = drot_hat[::2].bmm(drot_hat[1::2])
        rs = SO3.log(bmtm(drot_hat, drot_16)).reshape(N, -1, 3)[:, self.N0:]
        # print('- drot_hat:', drot_hat.shape, drot_hat.shape, drot_hat.device)
        # print('- rs:', drot_16.shape, drot_16.shape, drot_16.device)
        gyro_loss_16 = self.f_huber(rs)

        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            drot_hat = drot_hat[::2].bmm(drot_hat[1::2])
            drot_32 = drot_16[::2].bmm(drot_16[1::2])
            # print('- drot_hat:', drot_hat.shape, drot_hat.shape, drot_hat.device)
            # print('- rs:', drot_32.shape, drot_32.shape, drot_32.device)
            rs = SO3.log(bmtm(drot_hat, drot_32)).reshape(N, -1, 3)[:, self.N0:]
            gyro_loss_32 = self.f_huber(rs)/(2**(k - self.min_N + 1)) / 2.0
        ###

        # print('- gyro loss 16:', gyro_loss_16)
        # print('- gyro loss 32:', gyro_loss_32)
        return gyro_loss_16 + gyro_loss_32

    def accel_loss(self, a_hat, gt_dv_normed):
        # print('# ACCEL LOSS')
        v_hat = fast_acc_integration(a_hat) # torch.Size([6, 16000, 3]) True cuda:0
        # dv_hat = (( (a_hat[:, 1:] + a_hat[:, :-1]) / 2.0) * self.dt)
        # print('dv_hat:', dv_hat.shape, dv_hat.requires_grad)


        # gt_dv_keys = [*gt_dv.keys()]
        # gt_dv_keys.sort()

        gt_dv_normed_keys = []
        for key in self.dv_normed:
            gt_dv_normed_keys.append(int(key))
        gt_dv_normed_keys.sort()

        ## dv loss (당장에 수치적으로 크게 이상한 부분은 없다.)
        # dv_loss = {}
        # dv = dv_hat[:, ::2] + dv_hat[:, 1::2]
        # print('- dv:', dv.shape, dv.dtype, dv.device, dv.requires_grad)
        # d = 2

        # for key in gt_dv_keys:
        #     window_size = int(key)
        #     print('\n dv window_size:', window_size)
        #     value = gt_dv[key].detach()

        #     while d != window_size:
        #         d *= 2
        #         dv = dv[:, ::2] + dv[:, 1::2]

        #     value = value[:, ::window_size]
        #     print(' - gt_dv:', value.shape, value.dtype, value.device)
        #     print(' - dv:', dv.shape, dv.dtype, dv.device)
        #     _loss = (dv - value)**2
        #     _loss = _loss.mean(dim=1).sum()
        #     print(' - loss:', _loss)
        #     dv_loss[key] = _loss

        # print('dv_loss:')
        # for k, v in dv_loss.items():
        #     print('  - %s' % k, v)
        #     losses.append(v)

        ## dv_normed loss
        batch_size = a_hat.shape[0]
        N = a_hat.shape[1]

        # # Debugging
        # test = dv_hat[0, :10, 0]
        # test_v = 0.0
        # print('test: ', dv_hat[0, :10, 0].tolist())
        # print('test_v: ', end='')
        # for i in range(10):
        #     test_v += test[i]
        #     print(test_v.item(), end=', ')
        # print()

        loss = torch.tensor(0.0, requires_grad=True).cuda()

        for key in gt_dv_normed_keys:
            window_size = key
            gt_normed_v = gt_dv_normed[str(key)].detach().cuda()

            ## Integration
            # v_hat = dv_hat.clone().cuda()
            # # print('- v_hat:', v_hat.shape, v_hat.dtype, v_hat.device, v_hat.requires_grad)
            # for i in range(1, v_hat.shape[1]):
            #     v_hat[:, i, :] += v_hat[:, i-1, :]

            normed_v = vnorm(v_hat, window_size)

            # _mean = torch.zeros(v_hat.shape[0], 1, v_hat.shape[2]).cuda()
            # for i in range(v_hat.shape[1]):
            #     i0 = max(0, i+1-window_size)
            #     iN = i+1
            #     _mean = torch.cat([_mean, v_hat[:, i0:iN, :].mean(dim=1, keepdim=True)], dim=1)
            # v_hat = torch.cat([torch.zeros(v_hat.shape[0], 1, v_hat.shape[2]).cuda(), v_hat], dim=1)
            # normed_v = v_hat - _mean

            _loss = self.sln(gt_normed_v, normed_v) * np.log(window_size) * 10.0
            _loss = _loss.mean(dim=1).sum()
            loss += _loss
            # print('- accel loss %d:' % window_size, _loss)

            # est_normed_v = torch.cat([v_hat[:, 0, :].unsqueeze(1).expand(batch_size, padding_size, 3), v_hat], dim=1)
            # print('- est_normed_v:', est_normed_v.shape, est_normed_v.requires_grad)
            # for i in range(window_size-1, 0, -1):
            #     est_normed_v[:, i-1:i-1+N, :] += v_hat
            # est_normed_v = est_normed_v[:, :N, :]
            # est_normed_v = est_normed_v / float(window_size)
            # est_normed_v = v_hat - est_normed_v
            # print('- est_normed_v:', est_normed_v.shape, est_normed_v.requires_grad)
            # print(est_normed_v[0, :20, 0].tolist())

            # loss_normed = (est_normed_v - gt_normed_v) ** 2
            # print('- loss_normed:', loss_normed.shape)
            # loss_normed = loss_normed.mean(dim=1).sum()

            # print('- loss: ', loss_normed.item())
            # dv_normed_loss[key] = loss_normed

        ### Noise Loss - Window size is 51 (51//2 == 25[pad])
        avg_window = 15
        pad = avg_window // 2
        def avg(arr):
            avg_window = pad + 1 + pad
            batch = arr.shape[0]
            N = arr.shape[1]
            M = arr.shape[2]
            pad0 = arr[:, 0, :].unsqueeze(1).expand(batch, pad, M)
            padN = arr[:, -1, :].unsqueeze(1).expand(batch, pad, M)
            tmp_arr = torch.cat([pad0, arr, padN], dim=1)
            avg_arr = torch.zeros_like(arr)
            for i in range(avg_window):
                avg_arr += tmp_arr[:, i: i+N]
            avg_arr /= avg_window
            return avg_arr
        smoothed_a_hat = avg(a_hat)
        # print('smoothed_a_hat:', smoothed_a_hat.shape) # ([6, 16000, 3])
        gap_a_hat = (smoothed_a_hat - a_hat) ** 2
        gap_a_hat = gap_a_hat.mean(dim=1)
        gap_a_hat = gap_a_hat.mean(dim=0)
        # print('gap_a_hat:', gap_a_hat.shape) # torch.Size([3])
        # gap_a_hat = gap_a_hat
        # gap_x_loss = gap_a_hat[0]
        # gap_y_loss = gap_a_hat[1]
        # gap_z_loss = gap_a_hat[2]
        gap_loss =  torch.Tensor([.0]).cuda() # gap_x_loss + gap_y_loss + gap_z_loss

        return loss, gap_loss

    def forward(self, w_hat, dw_16, a_hat, gt_dv_normed):
        gloss = self.gyro_loss(w_hat, dw_16)
        acc_loss, gap_loss = self.accel_loss(a_hat, gt_dv_normed)
        return gloss, acc_loss, gap_loss
