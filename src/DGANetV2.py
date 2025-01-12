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

from termcolor import cprint


def vis(xs:torch.Tensor):
    for x in xs:
        print(x, x.shape)
        print()


class DGANetV2(torch.nn.Module):
    def __init__(self, in_dim, out_dim, channel, kernal, stride, dilation, momentum, gyro_std, acc_std):
        super().__init__()
        cprint('Initilaize DGANetV2 ...', 'green')
        print("  It's a DNN Model for denoising IMU sensor\'s raw measurements (both gryo & accel)")

        self.in_dim = in_dim
        self.out_dim = out_dim
        # channel dimension
        c, k, s, d = channel, kernal, stride, dilation
        # padding
        p0 = 504

        # todo
        # 1. Bias = False Try
        self.conv0 = torch.nn.Conv1d(in_dim, c[0],    k[0], s[0], dilation=d[0])
        self.conv1 = torch.nn.Conv1d(c[0],   c[1],    k[1], s[1], dilation=d[1])
        self.conv2 = torch.nn.Conv1d(c[1],   c[2],    k[2], s[2], dilation=d[2])
        self.conv3 = torch.nn.Conv1d(c[2],   c[3],    k[3], s[3], dilation=d[3])
        self.conv4 = torch.nn.Conv1d(c[3],   c[4],    k[4], s[4], dilation=d[4])
        self.conv5 = torch.nn.Conv1d(c[4],   c[5],    k[5], s[5], dilation=d[5])
        self.conv6 = torch.nn.Conv1d(c[5], out_dim, kernel_size=1)

        self.gelu = torch.nn.GELU()
        self.bn0 = torch.nn.BatchNorm1d(c[0], momentum=momentum)
        self.bn1 = torch.nn.BatchNorm1d(c[1], momentum=momentum)
        self.bn2 = torch.nn.BatchNorm1d(c[2], momentum=momentum)
        self.bn3 = torch.nn.BatchNorm1d(c[3], momentum=momentum)
        self.bn4 = torch.nn.BatchNorm1d(c[4], momentum=momentum)
        self.bn5 = torch.nn.BatchNorm1d(c[5], momentum=momentum)

        self.replicationPad1dS = torch.nn.ReplicationPad1d((p0, 0))
        self.replicationPad1dE = torch.nn.ReplicationPad1d((0, 0))

        ### Parameter member variables
        # Trainable
        self.C_w = torch.nn.Parameter(torch.randn(3, 3)*5e-2)
        self.C_a = torch.nn.Parameter(torch.randn(3, 3)*5e-2)
        # Not trainable
        self.mean_u     = torch.nn.Parameter(torch.zeros(in_dim),      requires_grad=False)
        self.std_u      = torch.nn.Parameter(torch.ones(in_dim),       requires_grad=False)
        self.gyro_std   = torch.nn.Parameter(torch.Tensor(gyro_std),   requires_grad=False)
        self.acc_std    = torch.nn.Parameter(torch.Tensor(acc_std),    requires_grad=False)
        self.I3          = torch.nn.Parameter(torch.eye(3),            requires_grad=False)
        self.g          = torch.nn.Parameter(torch.Tensor([0,0,9.82]), requires_grad=False) # Alse can be 9.81 | Is there any difference on results?
        ###

    def set_normalized_factors(self, mean_u, std_u):
        self.mean_u = torch.nn.Parameter(mean_u.cuda(), requires_grad=False)
        self.std_u = torch.nn.Parameter(std_u.cuda(), requires_grad=False)

    def forward(self, x:torch.Tensor):
        us = torch.clone(x)
        w_imu = us[:, :, 0:3]
        a_imu = us[:, :, 3:6]

        # Normalize
        x = (x - self.mean_u) / self.std_u
        x = x.transpose(1, 2)

        # Core
        x = self.replicationPad1dS(x)

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.gelu(x)

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
        x = self.bn5(x)
        x = self.gelu(x)

        x = self.conv6(x)
        x = x.transpose(1, 2)
        w_tilde = x[:, :, 0:3]
        a_tilde = x[:, :, 3:6]

        # w -- Post-process
        C_w = (self.I3 + self.C_w).expand(us.shape[0], us.shape[1], 3, 3)
        Rot_us = bbmv(C_w, w_imu)
        # print("gyro_std:", self.gyro_std.shape)
        # print("w_tilde:", w_tilde.shape)
        # print("Rot_us:", Rot_us.shape)
        w_hat = self.gyro_std * w_tilde + Rot_us
        # Rot_us: torch.Size([6, 16000, 3]) torch.float32
        # w_tilde: torch.Size([6, 16000, 3]) torch.float32
        # w_hat: torch.Size([6, 16000, 3]) torch.float32

        # a -- Post-process

        C_a = (self.I3 + self.C_a).expand(us.shape[0], us.shape[1], 3, 3)

        # print("C_a:", C_a.shape, C_a.dtype)
        # print("a_imu:", a_imu.shape, a_imu.dtype)
        # print("self.acc_std:", self.acc_std.shape, self.acc_std.dtype)
        # print("a_tilde:", a_tilde.shape, a_tilde.dtype)
        a_hat = bbmv(C_a, a_imu) + self.acc_std * a_tilde
        # print("a_hat:", a_hat.shape, a_hat.dtype)
            # C_a: torch.Size([6, 16000, 3, 3]) torch.float32
            # a_imu: torch.Size([6, 16000, 3]) torch.float32
            # self.acc_std: torch.Size([3]) torch.float32
            # a_tilde: torch.Size([6, 16000, 3]) torch.float32
            # a_hat: torch.Size([6, 16000, 3]) torch.float32
        return w_hat, a_hat

net_params = {
    'in_dim':  6,
    'out_dim': 6,
    'channel': [32, 64, 64, 128, 128, 256],
    'kernal':  [5,  5,  5,  5,   5,   5],
    'dilation':[2,  4,  8,  16,   32,   64],
    'stride':  [1,  1,  1,  1,   1,   1],
    'momentum': 0.1,
    'gyro_std': [1*np.pi/180, 2*np.pi/180, 5*np.pi/180],
    'acc_std': [2.0e-3, 2.0e-3, 2.0e-3],
}

if __name__ == "__main__":

    net = DGANetV2(**net_params).cuda()
    print(net)
    # vis([net.mean_u, net.std_u, net.gyro_std, net.Id3])

    input = torch.randn((6, 16000, 6)).cuda()
    print(input.shape)

    net.eval()
    w_hat, a_hat = net(input)

    w_hat = w_hat.cpu().detach()
    a_hat = a_hat.cpu().detach()
    print("w_hat:", w_hat.shape, w_hat.dtype, w_hat.device, w_hat.requires_grad)
    print("a_hat:", a_hat.shape, a_hat.dtype, a_hat.device, a_hat.requires_grad)
    # w_hat = w_hat.detach().numpy()
    # a_hat = a_hat.detach().numpy()
