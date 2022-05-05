import sys
if sys.platform.startswith('win'):
    sys.path.append(r"C:\Users\leech\Desktop\imu_ws\denoise-imu-gyro") # My window workspace path
elif sys.platform.startswith('linux'):
    sys.path.append('/root/project')

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.utils import bmtm, bmtv, bmmt, bbmv
from src.lie_algebra import SO3

class DgaPreNet(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        print('\n# Initilaize DgaPreNet ...')

        net_params = params['net']

        self.in_dim   = net_params['in_dim']
        self.out_dim  = net_params['out_dim']
        self.dropout  = net_params['dropout']
        self.momentum = net_params['momentum']
        self.gyro_std = net_params['gyro_std']
        self.acc_std  = net_params['acc_std']

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

        ### Non-parameter member variables
        self.gt_rots = None
        ###

        print("--- success ---\n")

    def set_normalized_factors(self, mean_u, std_u):
        self.mean_u = torch.nn.Parameter(mean_u.cuda(), requires_grad=False)
        self.std_u = torch.nn.Parameter(std_u.cuda(), requires_grad=False)

    def grab_gt_rots(self, gt_rots:torch.Tensor):
        self.gt_rots = gt_rots
        print("gt_rots' grad:", self.gt_rots.requires_grad)

    def forward(self, x:torch.Tensor):
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

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.gelu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.gelu(x)

        x = self.conv5(x)
        x = x.transpose(1, 2)
        return x
        # w_tilde = x[:, :, 0:3]
        # a_tilde = x[:, :, 3:6]



        ###########################
        # # Gyro -- Post-process
        # C_w = (self.I3 + self.C_w).expand(us.shape[0], us.shape[1], 3, 3)
        # Rot_us = bbmv(C_w, w_imu)
        # w_hat = self.gyro_std * w_tilde + Rot_us
        # # Rot_us: torch.Size([6, 16000, 3]) torch.float32
        # # w_tilde: torch.Size([6, 16000, 3]) torch.float32
        # # w_hat: torch.Size([6, 16000, 3]) torch.float32

        # ## Acc -- Post-process
        # C_a = (self.I3 + self.C_a).expand(us.shape[0], us.shape[1], 3, 3)

        # a_imu = bbmv(C_a, a_imu)
        # a_hat = a_imu + self.acc_std * a_tilde
        # a_hat = bbmv(rot_gt, a_hat)
        # a_hat -= self.g

        # # print("a_hat:", a_hat.shape, a_hat.dtype)
        #     # C_a: torch.Size([6, 16000, 3, 3]) torch.float32
        #     # a_imu: torch.Size([6, 16000, 3]) torch.float32
        #     # self.acc_std: torch.Size([3]) torch.float32
        #     # a_tilde: torch.Size([6, 16000, 3]) torch.float32
        #     # a_hat: torch.Size([6, 16000, 3]) torch.float32
        # return w_hat, a_hat

params = {
    'net': {
        'in_dim': 6,
        'out_dim': 16,
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

    net = DgaPreNet(params).cuda()
    print(net)

    input = torch.randn((1, 16000, 6)).cuda()
    rot   = torch.randn((1, 16000, 3, 3)).cuda()
    print(input.shape)

    net.eval()
    features = net(input, rot)
    print('features:', features.shape, features.dtype, features.device)
    # Output : (batch_size, seq_size, output_dim)

    # w_hat = w_hat.cpu().detach()
    # a_hat = a_hat.cpu().detach()
    # print("w_hat:", w_hat.shape, w_hat.dtype, w_hat.device, w_hat.requires_grad)
    # print("a_hat:", a_hat.shape, a_hat.dtype, a_hat.device, a_hat.requires_grad)
