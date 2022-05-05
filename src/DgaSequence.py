import sys
if sys.platform.startswith('win'):
    sys.path.append(r"C:\Users\leech\Desktop\imu_ws\denoise-imu-gyro") # My window workspace path
elif sys.platform.startswith('linux'):
    sys.path.append('/root/project')

import numpy as np
import torch
import torch.nn as nn
import kerasncp as kncp
from kerasncp.torch import LTCCell
from src.DgaPreNet import DgaPreNet


class DgaRawSequence(nn.Module):
    """nn.Module that unfolds a RNN cell into a sequence"""
    def __init__(
        self,
        rnn_cell,
    ):
        super(DgaRawSequence, self).__init__()
        self.rnn_cell:LTCCell = rnn_cell

    def forward(self, x):
        print('[DgaRawSequence] -- forward()')
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        out_dim = x.size(2)
        print('- Input Size:', x.shape, x.device)

        hidden_state = torch.zeros(
            (batch_size, self.rnn_cell.state_size), device=device
        )
        outputs = []
        for t in range(seq_len): # sequence를 Sliding window 1 size로 훑으며 업데이트하는 과정
            inputs = x[:, t]
            new_output, hidden_state = self.rnn_cell.forward(inputs, hidden_state)
            outputs.append(new_output)
            if t == 0:
                print('- Input per time:', inputs.shape, inputs.device)
                print('- Hidden per time:', hidden_state.shape, hidden_state.device)
                print('- Output per time:', new_output.shape, new_output.device)
        outputs = torch.stack(outputs, dim=1)  # return entire sequence
        print('- Return outputs:', outputs.shape, outputs.device)
        return outputs
        # return w_hat, a_hat

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

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        out_dim = x.size(2)
        print('input x size: ', x.shape, x.device)

        in_features = self.dga_net(x)
        print('in_features:',in_features.shape, in_features.device)

        hidden_state = torch.zeros(
            (batch_size, self.rnn_cell.state_size), device=device
        )
        outputs = []
        for t in range(seq_len): # sequence를 Sliding window 1 size로 훑으며 업데이트하는 과정
            inputs = in_features[:, t]
            new_output, hidden_state = self.rnn_cell.forward(inputs, hidden_state)
            outputs.append(new_output)
        outputs = torch.stack(outputs, dim=1)  # return entire sequence
        return outputs
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
