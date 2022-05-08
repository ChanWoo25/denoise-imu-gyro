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
        rnn_cell
    ):
        super(DgaRawSequence, self).__init__()
        self.rnn_cell:LTCCell = rnn_cell
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
