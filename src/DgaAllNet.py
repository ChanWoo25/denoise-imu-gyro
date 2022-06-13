import sys
if sys.platform.startswith('win'):
    sys.path.append(r"C:\Users\leech\Desktop\imu_ws\denoise-imu-gyro") # My window workspace path
elif sys.platform.startswith('linux'):
    sys.path.append('/home/leecw/project')

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import kerasncp as kncp
from kerasncp.torch import LTCCell

from src.DgaPreNet import DgaPreNet
from src.utils import bmtm, bmtv, bmmt, bbmv
from src.lie_algebra import SO3


class DgaAllNet(nn.Module):
    """nn.Module that unfolds a RNN cell into a sequence"""
    def __init__(
        self,
        params:dict
    ):
        super(DgaAllNet, self).__init__()

        self.gyro_net    = GyroNet(params)
        self.lstm_cell_w = torch.nn.LSTMCell(3, 128)

        self.lstm_w = torch.nn.LSTM(3, 128, 1, batch_first=True)
        self.lstm_q = torch.nn.LSTM(4, 128, 1, batch_first=True)
        self.lstm_T = torch.nn.LSTM(256, 512, 1, batch_first=True)

        self.lstm_cell_q = torch.nn.LSTMCell(4, 128)
        self.lstm_cell_T = torch.nn.LSTMCell(256, 512)
        # self.rnn_cell:LTCCell = rnn_cell

        self.fcn_0 = torch.nn.Linear(512, 128)
        self.fcn_1 = torch.nn.Linear(128, 32)
        self.fcn_2 = torch.nn.Linear(32, 4)

        self.mean_u = torch.nn.Parameter(torch.tensor([[0.0]*6]), requires_grad=False)
        self.std_u = torch.nn.Parameter(torch.tensor([[1.0]*6]), requires_grad=False)

    def set_normalized_factors(self, mean_u, std_u):
        self.mean_u = torch.nn.Parameter(mean_u, requires_grad=False)
        self.std_u = torch.nn.Parameter(std_u, requires_grad=False)

    def forward(self, us, q0):
        # q0: [n_batch, 4] initial state for each sequence

        window_size = 200
        qs = q0.unsqueeze(1).expand(q0.size(0), window_size, 4)

        # Normalize
        x = (us - self.mean_u) / self.std_u

        # w_hat, w_tilde = self.gyro_net(x) # [n_batch, seq_len, 3(r,p,y)]

        device = x.device
        batch_size = x.size(0) # train -> 6 ?
        seq_len = x.size(1) # 16000
        out_dim = x.size(2) # 3

        w_hat, w_tilde = self.gyro_net(x) ; print('w_hat:', w_hat.shape)  # [n_batch, seq_len, 3(r,p,y)]
        w_hat_pad = w_hat[:, 0, :].reshape(batch_size, 1, 3).expand(batch_size, window_size-1, 3)
        w_hat_pad = torch.cat([w_hat_pad, w_hat], dim=1) ; print('w_hat_pad:', w_hat_pad.shape)

        for i in range(seq_len):
            print('\n Idx: (%d / %d)' % (i, seq_len))
            input_w = w_hat_pad[:, i:i+window_size, :] # 이부분의 Graph가 mem에 많은 영향을 끼치지는 않는것으로 보인다.
            output_w, (hidden_w, cell_w) = self.lstm_w(input_w) ; print('output_w:', output_w.shape)
            input_q = qs[:, i:i+window_size, :].detach()
            output_q, (hidden_q, cell_q) = self.lstm_q(input_q) ; print('output_q:', output_q.shape)

            input_T = torch.cat([output_w, output_q], dim=2) ; print('input_T:', input_T.shape)
            output_T, (hidden_T, cell_T) = self.lstm_T(input_T) ; print('output_T:', output_T.shape)

            output_T = self.fcn_0(output_T[:, -1, :]) ; print('output_T:', output_T.shape)
            output_T = self.fcn_1(output_T) ; print('output_T:', output_T.shape)
            output_T = self.fcn_2(output_T) ; print('output_T:', output_T.shape)
            # output_T at this moment means current new delta quaternion

            q = qs[:, -1] + output_T  ; print('q:', q.shape)
            # normalize
            q = q / q.sum(dim=1).unsqueeze(1)
            print('q-sum:', q.sum())
            q = q.unsqueeze(1) ; print('q:', q.shape)
            qs = torch.cat([qs, q], dim=1) ; print('qs:', qs.shape)

        qs = qs[:, window_size-1:, :] ; print('qs:', qs.shape)

        return qs

        ## Normalize
        _mean = self.mean_us.expand_as(x).to(x.device)
        _std  = self.std_us.expand_as(x).to(x.device)
        x = (x - _mean) / _std

        hidden_state = torch.zeros(
            (batch_size, self.rnn_cell.state_size), device=device
        )
        outputs = []
        for t in range(seq_len): # sequence를 Sliding window 1 size로 훑으며 업데이트하는 과정
            inputs = x[:, t]
            new_output, hidden_state = self.rnn_cell.forward(inputs, hidden_state)
            outputs.append(new_output)
            # if t == 0:
            #     print('- Input per time:', inputs.shape, inputs.device)
            #     print('- Hidden per time:', hidden_state.shape, hidden_state.device)
            #     print('- Output per time:', new_output.shape, new_output.device)
        outputs = torch.stack(outputs, dim=1)  # return entire sequence

        outputs = (outputs * _std[:, :, 0:3]) + _mean[:, :, 0:3]
        return outputs



class GyroNet(torch.nn.Module):
    """It's a DNN Model for denoising IMU sensor\'s raw measurements (both gryo & accel)"""
    def __init__(self, params):
        super().__init__()
        print('\n# Initilaize GyroNet ...')

        net_params = params['net']
        self.in_dim  = net_params['in_dim']
        self.out_dim = 3 # net_params['out_dim']
        self.dropout = net_params['dropout']
        self.momentum= net_params['momentum']
        self.gyro_std = net_params['gyro_std']
        self.acc_std = net_params['acc_std']

        # channel dimension
        c0 = net_params['c0']
        c1 = 2*c0
        c2 = 2*c1
        c3 = 2*c2
        # kernel dimension (odd number)
        k0 = net_params['ks'][0]
        k1 = net_params['ks'][1]
        k2 = net_params['ks'][2]
        k3 = net_params['ks'][3]
        # dilation dimension
        d0 = net_params['ds'][0]
        d1 = net_params['ds'][1]
        d2 = net_params['ds'][2]
        # padding
        p0 = (k0-1) + d0*(k1-1) + d0*d1*(k2-1) + d0*d1*d2*(k3-1)

        # todo
        # 1. Bias = False Try
        self.conv1 = torch.nn.Conv1d(self.in_dim, c0, k0, dilation=1)
        self.conv2 = torch.nn.Conv1d(c0, c1, k1, dilation=d0)
        self.conv3 = torch.nn.Conv1d(c1, c2, k2, dilation=d0*d1)
        self.conv4 = torch.nn.Conv1d(c2, c3, k3, dilation=d0*d1*d2)
        self.conv5 = torch.nn.Conv1d(c3, self.out_dim, 1, dilation=1)

        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(self.dropout)
        self.bn1 = torch.nn.BatchNorm1d(c0, momentum=self.momentum)
        self.bn2 = torch.nn.BatchNorm1d(c1, momentum=self.momentum)
        self.bn3 = torch.nn.BatchNorm1d(c2, momentum=self.momentum)
        self.bn4 = torch.nn.BatchNorm1d(c3, momentum=self.momentum)

        self.replicationPad1dS = torch.nn.ReplicationPad1d((p0, 0))
        self.replicationPad1dE = torch.nn.ReplicationPad1d((0, 0))

        ### Parameter member variables
        # Trainable
        self.C_w = torch.nn.Parameter(torch.randn(3, 3)*5e-2)
        self.C_a = torch.nn.Parameter(torch.randn(3, 3)*5e-2)
        # Not trainable
        self.mean_u     = torch.nn.Parameter(torch.zeros(self.in_dim),      requires_grad=False)
        self.std_u      = torch.nn.Parameter(torch.ones(self.in_dim),       requires_grad=False)
        self.gyro_std   = torch.nn.Parameter(torch.Tensor(self.gyro_std),   requires_grad=False)
        self.acc_std    = torch.nn.Parameter(torch.Tensor(self.acc_std),    requires_grad=False)
        self.I3          = torch.nn.Parameter(torch.eye(3),            requires_grad=False)
        self.g          = torch.nn.Parameter(torch.Tensor([0,0,9.81]), requires_grad=False) # Alse can be 9.81 | Is there any difference on results?
        ###


    def forward(self, us:torch.Tensor):
        w_imu = us[:, :, 0:3]
        a_imu = us[:, :, 3:6]

        # Core
        x = us.transpose(1, 2)
        x = self.replicationPad1dS(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = x.transpose(1, 2)
        w_tilde = x[:, :, 0:3]

        # Gyro -- Post-process
        C_w = (self.I3 + self.C_w).expand(us.shape[0], us.shape[1], 3, 3)
        w_imu_calib = bbmv(C_w, w_imu)
        w_tilde = self.gyro_std * w_tilde
        w_hat = w_tilde + w_imu_calib
        # Rot_us: torch.Size([6, 16000, 3]) torch.float32
        # w_tilde: torch.Size([6, 16000, 3]) torch.float32
        # w_hat: torch.Size([6, 16000, 3]) torch.float32

        return w_hat, w_tilde


class DGANetVer2(torch.nn.Module):
    """
        - Body frame과 World frame에 대하여 잡음이 모두 존재할거라는 가정
        - Accel만 추정
    """
    def __init__(self, params):
        super().__init__()
        print('\n# Initilaize DGANetVer2 ...')

        net_params = params['net']

        self.in_dim  = net_params['in_dim']
        self.out_dim = 6
        self.dropout = net_params['dropout']
        self.momentum= net_params['momentum']
        self.gyro_std = net_params['gyro_std']
        self.acc_std = net_params['acc_std']

        # channel dimension
        c0 = net_params['c0']
        c1 = 2*c0
        c2 = 2*c1
        c3 = 2*c2
        # kernel dimension (odd number)
        k0 = net_params['ks'][0]
        k1 = net_params['ks'][1]
        k2 = net_params['ks'][2]
        k3 = net_params['ks'][3]
        # dilation dimension
        d0 = net_params['ds'][0]
        d1 = net_params['ds'][1]
        d2 = net_params['ds'][2]
        # padding
        p0 = (k0-1) + d0*(k1-1) + d0*d1*(k2-1) + d0*d1*d2*(k3-1)

        # todo
        # 1. Bias = False Try
        self.conv1 = torch.nn.Conv1d(self.in_dim, c0, k0, dilation=1)
        self.conv2 = torch.nn.Conv1d(c0, c1, k1, dilation=d0)
        self.conv3 = torch.nn.Conv1d(c1, c2, k2, dilation=d0*d1)
        self.conv4 = torch.nn.Conv1d(c2, c3, k3, dilation=d0*d1*d2)
        self.conv5 = torch.nn.Conv1d(c3, self.out_dim, 1, dilation=1)

        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(self.dropout)
        self.bn1 = torch.nn.BatchNorm1d(c0, momentum=self.momentum)
        self.bn2 = torch.nn.BatchNorm1d(c1, momentum=self.momentum)
        self.bn3 = torch.nn.BatchNorm1d(c2, momentum=self.momentum)
        self.bn4 = torch.nn.BatchNorm1d(c3, momentum=self.momentum)

        self.replicationPad1dS = torch.nn.ReplicationPad1d((p0, 0))
        self.replicationPad1dE = torch.nn.ReplicationPad1d((0, 0))

        ### Parameter member variables
        # Trainable
        self.C_a = torch.nn.Parameter(torch.randn(3, 3)*5e-2)
        # Not trainable
        self.mean_u     = torch.nn.Parameter(torch.zeros(self.in_dim),      requires_grad=False)
        self.std_u      = torch.nn.Parameter(torch.ones(self.in_dim),       requires_grad=False)
        self.gyro_std   = torch.nn.Parameter(torch.Tensor(self.gyro_std),   requires_grad=False)
        self.acc_std    = torch.nn.Parameter(torch.Tensor(self.acc_std),    requires_grad=False)
        self.I3          = torch.nn.Parameter(torch.eye(3),            requires_grad=False)
        self.g          = torch.nn.Parameter(torch.Tensor([0,0,9.81]), requires_grad=False) # Alse can be 9.81 | Is there any difference on results?
        ###

    def set_normalized_factors(self, mean_u, std_u):
        self.mean_u = torch.nn.Parameter(mean_u.cuda(), requires_grad=False)
        self.std_u = torch.nn.Parameter(std_u.cuda(), requires_grad=False)

    def forward(self, x:torch.Tensor, rot_gt:torch.Tensor, mode='train'):
        us = torch.clone(x)
        w_imu = us[:, :, 0:3]
        a_imu = us[:, :, 3:6]

        # Normalize
        x = (x - self.mean_u) / self.std_u
        x = x.transpose(1, 2)

        # Core
        x = self.replicationPad1dS(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = x.transpose(1, 2)
        a_tilde_b = x[:, :, 0:3]
        a_tilde_w = x[:, :, 3:6]

        ## Acc -- Post-process
        C_a = (self.I3 + self.C_a).expand(us.shape[0], us.shape[1], 3, 3)

        a_imu = bbmv(C_a, a_imu)
        a_hat = a_imu - self.acc_std * a_tilde_b
        # print('a_w:', bbmv(rot_gt, a_hat).shape)
        # print('a_t_w:', a_tilde_w.shape)
        # print('self.g:', self.g)
        a_hat = bbmv(rot_gt, a_hat) - self.acc_std *  a_tilde_w - self.g

        if mode == 'train':
            return a_hat
        elif mode == 'test':
            return a_hat, a_tilde_b, a_tilde_w


class DGANetVer3(torch.nn.Module):
    """
        - World frame에 대하여 잡음이 존재할거라는 가정
        - Accel만 추정
    """
    def __init__(self, params):
        super().__init__()
        print('\n# Initilaize DGANetVer2 ...')

        net_params = params['net']

        self.in_dim  = net_params['in_dim']
        self.out_dim = 6
        self.dropout = net_params['dropout']
        self.momentum= net_params['momentum']
        self.gyro_std = net_params['gyro_std']
        self.acc_std = net_params['acc_std']

        # channel dimension
        c0 = net_params['c0']
        c1 = 2*c0
        c2 = 2*c1
        c3 = 2*c2
        # kernel dimension (odd number)
        k0 = net_params['ks'][0]
        k1 = net_params['ks'][1]
        k2 = net_params['ks'][2]
        k3 = net_params['ks'][3]
        # dilation dimension
        d0 = net_params['ds'][0]
        d1 = net_params['ds'][1]
        d2 = net_params['ds'][2]
        # padding
        p0 = (k0-1) + d0*(k1-1) + d0*d1*(k2-1) + d0*d1*d2*(k3-1)

        # todo
        # 1. Bias = False Try
        self.conv1 = torch.nn.Conv1d(self.in_dim, c0, k0, dilation=1)
        self.conv2 = torch.nn.Conv1d(c0, c1, k1, dilation=d0)
        self.conv3 = torch.nn.Conv1d(c1, c2, k2, dilation=d0*d1)
        self.conv4 = torch.nn.Conv1d(c2, c3, k3, dilation=d0*d1*d2)
        self.conv5 = torch.nn.Conv1d(c3, self.out_dim, 1, dilation=1)

        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(self.dropout)
        self.bn1 = torch.nn.BatchNorm1d(c0, momentum=self.momentum)
        self.bn2 = torch.nn.BatchNorm1d(c1, momentum=self.momentum)
        self.bn3 = torch.nn.BatchNorm1d(c2, momentum=self.momentum)
        self.bn4 = torch.nn.BatchNorm1d(c3, momentum=self.momentum)

        self.replicationPad1dS = torch.nn.ReplicationPad1d((p0, 0))
        self.replicationPad1dE = torch.nn.ReplicationPad1d((0, 0))

        ### Parameter member variables
        # Trainable
        self.C_a = torch.nn.Parameter(torch.randn(3, 3)*5e-2)
        # Not trainable
        self.mean_u     = torch.nn.Parameter(torch.zeros(self.in_dim),      requires_grad=False)
        self.std_u      = torch.nn.Parameter(torch.ones(self.in_dim),       requires_grad=False)
        self.gyro_std   = torch.nn.Parameter(torch.Tensor(self.gyro_std),   requires_grad=False)
        self.acc_std    = torch.nn.Parameter(torch.Tensor(self.acc_std),    requires_grad=False)
        self.I3          = torch.nn.Parameter(torch.eye(3),            requires_grad=False)
        self.g          = torch.nn.Parameter(torch.Tensor([0,0,9.81]), requires_grad=False) # Alse can be 9.81 | Is there any difference on results?
        ###

    def set_normalized_factors(self, mean_u, std_u):
        self.mean_u = torch.nn.Parameter(mean_u.cuda(), requires_grad=False)
        self.std_u = torch.nn.Parameter(std_u.cuda(), requires_grad=False)

    def forward(self, x:torch.Tensor, rot_gt:torch.Tensor, mode='train'):
        us = torch.clone(x)
        w_imu = us[:, :, 0:3]
        a_imu = us[:, :, 3:6]

        # Normalize
        x = (x - self.mean_u) / self.std_u
        x = x.transpose(1, 2)

        # Core
        x = self.replicationPad1dS(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.gelu(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = x.transpose(1, 2)
        a_tilde_w = x[:, :, 0:3]

        ## Acc -- Post-process
        C_a = (self.I3 + self.C_a).expand(us.shape[0], us.shape[1], 3, 3)

        a_imu = bbmv(C_a, a_imu)
        a_hat = a_imu
        # print('a_w:', bbmv(rot_gt, a_hat).shape)
        # print('a_t_w:', a_tilde_w.shape)
        # print('self.g:', self.g)
        a_hat = bbmv(rot_gt, a_hat) - self.acc_std * a_tilde_w - self.g

        if mode == 'train':
            return a_hat
        elif mode == 'test':
            return a_hat, a_tilde_w



class DgaWinSequence(nn.Module):
    """nn.Module that unfolds a RNN cell into a sequence"""
    def __init__(
        self,
        rnn_cell,
        dga_net
    ):
        super(DgaWinSequence, self).__init__()
        self.rnn_cell:LTCCell = rnn_cell
        self.dga_net:DgaPreNet = dga_net
        self.mean_us = 0.0
        self.std_us = 1.0

    def set_nf(self, mean_us, std_us):
        self.mean_us = mean_us
        self.std_us = std_us

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        out_dim = x.size(2)

        ## Normalize
        _mean = self.mean_us.expand_as(x).to(x.device)
        _std  = self.std_us.expand_as(x).to(x.device)
        x = (x - _mean) / _std

        in_features = self.dga_net(x)
        # print('in_features:',in_features.shape, in_features.device)

        hidden_state = torch.zeros(
            (batch_size, self.rnn_cell.state_size), device=device
        )
        outputs = []
        for t in range(seq_len): # sequence를 Sliding window 1 size로 훑으며 업데이트하는 과정
            inputs = in_features[:, t]
            new_output, hidden_state = self.rnn_cell.forward(inputs, hidden_state)
            outputs.append(new_output)
        outputs = torch.stack(outputs, dim=1)  # return entire sequence

        outputs = (outputs * _std[:, :, 0:3]) + _mean[:, :, 0:3]
        return outputs



params = {
    'net': {
        'in_dim': 6,
        'out_dim': 3,
        'c0': 16,
        'dropout': 0.1,
        'ks': [7, 7, 7, 7],
        'ds': [4, 4, 4],
        'momentum': 0.1,
        'gyro_std': [1*np.pi/180, 2*np.pi/180, 5*np.pi/180],
        'acc_std': [2.0e-3, 2.0e-3, 2.0e-3],
    }
}

if __name__ == "__main__":

    input = torch.randn((6, 1600, 6)).cuda()
    q0 = torch.randn((6, 4)).cuda()
    q0 = q0 / q0.sum(dim=1).unsqueeze(1)

    rot   = torch.randn((6, 1600, 3, 3)).cuda()
    print(input.shape)

    model = DgaAllNet(params)
    model = model.cuda()

    output = model(input, q0).cpu()
    print(output.shape)
