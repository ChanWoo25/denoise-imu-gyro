import sys
if sys.platform.startswith('win'):
    sys.path.append(r"C:\Users\leech\Desktop\imu_ws\denoise-imu-gyro") # My window workspace path
elif sys.platform.startswith('linux'):
    sys.path.append('/root/denoise')

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.utils import bmtm, bmtv, bmmt, bbmv
from src.lie_algebra import SO3


def vis(xs:torch.Tensor):
    for x in xs:
        print(x, x.shape)
        print()

class GyroNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, c0, dropout, ks, ds, momentum, gyro_std):
        super().__init__()
        print("Initialize Start")
        self.in_dim = in_dim
        self.out_dim = out_dim
        # channel dimension
        c1 = 2*c0
        c2 = 2*c1
        c3 = 2*c2
        # kernel dimension (odd number)
        k0 = ks[0]
        k1 = ks[1]
        k2 = ks[2]
        k3 = ks[3]
        # dilation dimension
        d0 = ds[0]
        d1 = ds[1]
        d2 = ds[2]
        # padding
        p0 = (k0-1) + d0*(k1-1) + d0*d1*(k2-1) + d0*d1*d2*(k3-1)

        # todo
        # 1. Bias = False Try
        self.conv1 = torch.nn.Conv1d(in_dim, c0, k0, dilation=1)
        self.conv2 = torch.nn.Conv1d(c0, c1, k1, dilation=d0)
        self.conv3 = torch.nn.Conv1d(c1, c2, k2, dilation=d0*d1)
        self.conv4 = torch.nn.Conv1d(c2, c3, k3, dilation=d0*d1*d2)
        self.conv5 = torch.nn.Conv1d(c3, out_dim, 1, dilation=1)

        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)
        self.bn1 = torch.nn.BatchNorm1d(c0, momentum=momentum)
        self.bn2 = torch.nn.BatchNorm1d(c1, momentum=momentum)
        self.bn3 = torch.nn.BatchNorm1d(c2, momentum=momentum)
        self.bn4 = torch.nn.BatchNorm1d(c3, momentum=momentum)

        self.replicationPad1dS = torch.nn.ReplicationPad1d((p0, 0))
        self.replicationPad1dE = torch.nn.ReplicationPad1d((0, 0))


        # for normalizing inputs
        self.mean_u = torch.nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        self.std_u  = torch.nn.Parameter(torch.ones(in_dim),  requires_grad=False)

        gyro_std = torch.Tensor(gyro_std)
        self.gyro_std = torch.nn.Parameter(gyro_std, requires_grad=False)

        gyro_Rot = 0.05*torch.randn(3, 3)
        self.gyro_Rot = torch.nn.Parameter(gyro_Rot)
        # self.Id3 = torch.eye(3)
        self.Id3 = torch.nn.Parameter(torch.eye(3), requires_grad=False)
        print("Initialize End")

    def set_normalized_factors(self, mean_u, std_u):
        self.mean_u = torch.nn.Parameter(mean_u.cuda(), requires_grad=False)
        self.std_u = torch.nn.Parameter(std_u.cuda(), requires_grad=False)

    def forward(self, x:torch.Tensor):
        us = torch.clone(x)

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

        # Post-process
        Rots = (self.Id3 + self.gyro_Rot).expand(us.shape[0], us.shape[1], 3, 3)
        Rot_us = bbmv(Rots, us[:, :, :3])
        w_hat = self.gyro_std * x + Rot_us

        # print("Rot_us:", Rot_us.shape, Rot_us.dtype)
        # print("x:", x.shape, x.dtype)
        # print("w_hat:", w_hat.shape, w_hat.dtype)
        # Rot_us: torch.Size([6, 16000, 3]) torch.float32
        # x: torch.Size([6, 16000, 3]) torch.float32
        # w_hat: torch.Size([6, 16000, 3]) torch.float32
        return w_hat

net_params = {
    'in_dim': 6,
    'out_dim': 3,
    'c0': 16,
    'dropout': 0.1,
    'ks': [7, 7, 7, 7],
    'ds': [4, 4, 4],
    'momentum': 0.1,
    'gyro_std': [1*np.pi/180, 2*np.pi/180, 5*np.pi/180],
}

if __name__ == "__main__":

    net = GyroNet(**net_params).cuda()
    print(net)
    vis([net.mean_u, net.std_u, net.gyro_Rot, net.gyro_std, net.Id3])

    input = torch.randn((1, 1, 6)).cuda()
    print(input.shape)

    net.eval()
    output = net(input).detach().cpu()
    print(output, output.shape)
