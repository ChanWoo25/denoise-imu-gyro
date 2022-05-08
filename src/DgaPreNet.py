import torch
import sys
sys.path.append('/root/project')

class DgaPreNet(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
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

        self.relu = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)
        self.bn1 = torch.nn.BatchNorm1d(c0, momentum=self.momentum)
        self.bn2 = torch.nn.BatchNorm1d(c1, momentum=self.momentum)
        self.bn3 = torch.nn.BatchNorm1d(c2, momentum=self.momentum)
        self.bn4 = torch.nn.BatchNorm1d(c3, momentum=self.momentum)

        self.replicationPad1dS = torch.nn.ReplicationPad1d((p0, 0))
        self.replicationPad1dE = torch.nn.ReplicationPad1d((0, 0))

        ### Parameter member variables
        # # Trainable
        # self.C_w = torch.nn.Parameter(torch.randn(3, 3)*5e-2)
        # self.C_a = torch.nn.Parameter(torch.randn(3, 3)*5e-2)
        # # Not trainable
        # self.mean_u     = torch.nn.Parameter(torch.zeros(self.in_dim),      requires_grad=False)
        # self.std_u      = torch.nn.Parameter(torch.ones(self.in_dim),       requires_grad=False)
        # self.gyro_std   = torch.nn.Parameter(torch.Tensor(self.gyro_std),   requires_grad=False)
        # self.acc_std    = torch.nn.Parameter(torch.Tensor(self.acc_std),    requires_grad=False)
        # self.I3          = torch.nn.Parameter(torch.eye(3),            requires_grad=False)
        # self.g          = torch.nn.Parameter(torch.Tensor([0,0,9.81]), requires_grad=False) # Alse can be 9.81 | Is there any difference on results?
        # ###

    def forward(self, x:torch.Tensor):
        x = x.transpose(1, 2)
        x = self.replicationPad1dS(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv5(x)
        x = x.transpose(1, 2)
        return x
