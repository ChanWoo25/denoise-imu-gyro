import sys
if sys.platform.startswith('win'):
    sys.path.append(r"C:\Users\leech\Desktop\imu_ws\denoise-imu-gyro") # My window workspace path
elif sys.platform.startswith('linux'):
    sys.path.append('/root/denoise')

from src.utils import pdump, pload, bmtv, bmtm
from src.lie_algebra import SO3
from termcolor import cprint
from torch.utils.data.dataset import Dataset
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch
import sys

class DGADataset(Dataset):

    def __init__(self, train_seqs, val_seqs, test_seqs, data_dir, predata_dir, \
            N, min_train_freq, max_train_freq, v_window, mode):

        super().__init__()

        self.seq_dict = {
            'train': train_seqs,
            'val': val_seqs,
            'test': test_seqs,
        }
        # train_seqs, self.sequences = self.get_sequences(train_seqs, val_seqs, test_seqs)
        self.data_dir = data_dir
        self.predata_dir = predata_dir
        self.N = N
        self.min_train_freq = min_train_freq
        self.max_train_freq = max_train_freq
        self.v_window = v_window
        self.mode = mode
        self.sequences = self.seq_dict[mode]
        self.dt = 0.05

        self.path_normalize_factors = os.path.join(predata_dir, 'nf.p')
        self.mean_u, self.std_u = self.init_normalize_factors(train_seqs)

        self._train = False
        self._val = False

        ## Additional
        # noise density
        self.imu_std = torch.Tensor([8e-5, 1e-3]).float()
        # bias repeatability (without in-run bias stability)
        self.imu_b0 = torch.Tensor([1e-3, 1e-3]).float()
        # sequence size during training
        self.uni = torch.distributions.uniform.Uniform(-torch.ones(1), torch.ones(1))

    def __getitem__(self, i):
        mondict = self.load_seq(i)
        us = mondict['us']
        dxi_ij = mondict['dxi_ij']
        gt_interpolated = mondict['gt_interpolated']
        delta_v_gt = mondict['delta_v_gt']
        vs_gt_norm = mondict['vs_gt_norm']


        N_max = dxi_ij.shape[0]

        if self._train: # random start
            n0 = torch.randint(0, self.max_train_freq, (1, ))
            nend = n0 + self.N
        elif self._val: # end sequence
            n0 = self.max_train_freq + self.N
            nend = N_max - ((N_max - n0) % self.max_train_freq)
        else:  # full sequence
            n0 = 0
            nend = N_max - (N_max % self.max_train_freq)

        us = us[n0: nend]
        dxi_ij = dxi_ij[n0: nend]
        delta_v_gt = delta_v_gt[n0: nend]
        gt_interpolated = gt_interpolated[n0: nend]
        vs_gt_norm = vs_gt_norm[n0: nend]
        return us, dxi_ij, delta_v_gt, gt_interpolated, vs_gt_norm

    def __len__(self):
        return len(self.seq_dict[self.mode])

    def add_noise(self, u):
        """Add Gaussian noise and bias to input"""
        noise = torch.randn_like(u)
        noise[:, :, :3] = noise[:, :, :3] * self.imu_std[0]
        noise[:, :, 3:6] = noise[:, :, 3:6] * self.imu_std[1]

        # bias repeatability (without in run bias stability)
        b0 = self.uni.sample(u[:, 0].shape).cuda()
        b0[:, :, :3] = b0[:, :, :3] * self.imu_b0[0]
        b0[:, :, 3:6] =  b0[:, :, 3:6] * self.imu_b0[1]
        u = u + noise + b0.transpose(1, 2)
        return u

    def init_train(self):
        self._train = True
        self._val = False

    def init_val(self):
        self._train = False
        self._val = True

    def load_seq(self, i):
        return pload(self.predata_dir, self.seq_dict[self.mode][i] + '_v%d.p'%self.v_window)

    def init_normalize_factors(self, seqs):
        if os.path.exists(self.path_normalize_factors):
            mondict = pload(self.path_normalize_factors)
            seqs = mondict['seqs']
            cprint('Load nf.p ...', 'green')
            cprint('  Training sequences must be equel to Normalized sequences below', 'yellow')
            for seq in seqs:
                print('  - %s' % seq)
            return mondict['mean_u'], mondict['std_u']

        cprint('Compute normalized factor ...', 'green')
        print('  Note: Do this only once on training time')

        # first compute mean
        mean_seqs = 0.0
        n_data = 0
        n_pos = 0
        n_neg = 0

        for i, seq in enumerate(seqs):
            pickle_dict = pload(self.predata_dir, seq + '_v%d.p'%self.v_window)
            us = pickle_dict['us']
            sms = pickle_dict['dxi_ij']

            if i == 0:
                mean_seqs = us.sum(dim=0)
                n_pos = sms.sum(dim=0)
                n_neg = sms.shape[0] - sms.sum(dim=0)
            else:
                mean_seqs += us.sum(dim=0)
                n_pos += sms.sum(dim=0)
                n_neg += sms.shape[0] - sms.sum(dim=0)

            n_data += us.shape[0]

        mean_seqs = mean_seqs / n_data
        pos_weight = n_neg / n_pos

        # second compute standard deviation
        for i, seq in enumerate(seqs):
            pickle_dict = pload(self.predata_dir, seq + '_v%d.p'%self.v_window)
            us = pickle_dict['us']
            if i == 0:
                std_u = ((us - mean_seqs) ** 2).sum(dim=0)
            else:
                std_u += ((us - mean_seqs) ** 2).sum(dim=0)
        std_u = (std_u / n_data).sqrt()

        normalize_factors = {
            'mean_u': mean_seqs,
            'std_u': std_u,
            'seqs': seqs
        }
        print('... ended computing normalizing factors')
        print('pos_weight:', pos_weight)
        print('This values most be a training parameters !')
        print('mean_u    :', mean_seqs)
        print('std_u     :', std_u)
        print('n_data    :', n_data)
        pdump(normalize_factors, self.path_normalize_factors)
        return mean_seqs, std_u


    def save_csv(self, sequence):
        print("Save pre-processed gt & imu csv files -- %s" % sequence)
        gt_path = os.path.join(self.GT_DIR, sequence, "gt_interpolated.csv")
        gt_csv = self.gt_processed[sequence][:,0:11]
        header = "time[ns],px,py,pz,qw,qx,qy,qz,vx,vy,vz"
        np.savetxt(gt_path, gt_csv, fmt="%d,%1.9f,%1.9f,%1.9f,%1.9f,%1.9f,%1.9f,%1.9f,%1.9f,%1.9f,%1.9f", header=header)
        print("\tsaved in file \'%s\'" % gt_path, "shape:", gt_csv.shape) # (29992, 11)

        imu_path = os.path.join(self.DEST_DIR, sequence + "_raw_imu_interpolated.csv")
        imu_csv = self.imu_processed[sequence]
        header = "time[ns],wx,wy,wz,ax,ay,az"
        np.savetxt(imu_path, imu_csv, fmt="%d,%1.9f,%1.9f,%1.9f,%1.9f,%1.9f,%1.9f", header=header)
        print("\tsaved in file \'%s\'" % imu_path, "shape:", imu_csv.shape) # (29992, 7)

if __name__ == '__main__':
    params = {
        'predata_dir': '/root/Data/Result/DenoiseIMU',
        # where are raw data ?
        'data_dir': '/root/Data/EUROC',
        # set train, val and test sequence
        'train_seqs': [
            'MH_01_easy',
            'MH_03_medium',
            'MH_05_difficult',
            'V1_02_medium',
            'V2_01_easy',
            'V2_03_difficult'
            ],
        'val_seqs': [
            'MH_01_easy',
            'MH_03_medium',
            'MH_05_difficult',
            'V1_02_medium',
            'V2_01_easy',
            'V2_03_difficult',
            ],
        'test_seqs': [
            'MH_02_easy',
            'MH_04_difficult',
            'V2_02_medium',
            'V1_03_difficult',
            'V1_01_easy',
            ],
        # size of trajectory during training
        'N': 32 * 500, # should be integer * 'max_train_freq'
        'min_train_freq': 16,
        'max_train_freq': 32,
    }

    dataset_train = DGADataset(**params, mode='train')
