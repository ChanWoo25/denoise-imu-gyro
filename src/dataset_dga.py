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

class BaseDataset(Dataset):

    def __init__(self, predata_dir, train_seqs, val_seqs, test_seqs, mode, N,
        min_train_freq=128, max_train_freq=512, dt=0.005):
        super().__init__()
        # where record pre loaded data
        self.predata_dir = predata_dir
        self.path_normalize_factors = os.path.join(predata_dir, 'nf.p')

        self.mode = mode
        if mode is 'parse':
            print("Parse mode -- parent class is not initiated.")
            return

        # choose between training, validation or test sequences
        train_seqs, self.sequences = self.get_sequences(train_seqs, val_seqs, test_seqs)
        # get and compute value for normalizing inputs
        self.mean_u, self.std_u = self.init_normalize_factors(train_seqs)
        self.mode = mode  # train, val or test
        self._train = False
        self._val = False
        # noise density
        self.imu_std = torch.Tensor([8e-5, 1e-3]).float()
        # bias repeatability (without in-run bias stability)
        self.imu_b0 = torch.Tensor([1e-3, 1e-3]).float()
        # IMU sampling time
        self.dt = dt # (s)
        # sequence size during training
        self.N = N # power of 2
        self.min_train_freq = min_train_freq
        self.max_train_freq = max_train_freq
        self.uni = torch.distributions.uniform.Uniform(-torch.ones(1), torch.ones(1))

    def get_sequences(self, train_seqs, val_seqs, test_seqs):
        """Choose sequence list depending on dataset mode"""
        sequences_dict = {
            'train': train_seqs,
            'val': val_seqs,
            'test': test_seqs,
        }
        return sequences_dict['train'], sequences_dict[self.mode]

    def __getitem__(self, i):
        mondict = self.load_seq(i)
        N_max = mondict['xs'].shape[0]
        if self._train: # random start
            n0 = torch.randint(0, self.max_train_freq, (1, ))
            nend = n0 + self.N
        elif self._val: # end sequence
            n0 = self.max_train_freq + self.N
            nend = N_max - ((N_max - n0) % self.max_train_freq)
        else:  # full sequence
            n0 = 0
            nend = N_max - (N_max % self.max_train_freq)
        u = mondict['us'][n0: nend]
        x = mondict['xs'][n0: nend]
        dv = mondict['dv_gt'][n0: nend]
        return u, x, dv

    def __len__(self):
        return len(self.sequences)

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

    def length(self):
        return self._length

    def load_seq(self, i):
        return pload(self.predata_dir, self.sequences[i] + '.p')

    def load_gt(self, i):
        return pload(self.predata_dir, self.sequences[i] + '_gt.p')

    def init_normalize_factors(self, train_seqs):
        if os.path.exists(self.path_normalize_factors):
            mondict = pload(self.path_normalize_factors)
            return mondict['mean_u'], mondict['std_u']

        path = os.path.join(self.predata_dir, train_seqs[0] + '.p')
        if not os.path.exists(path):
            print("init_normalize_factors not computed")
            return 0, 0

        print('Start computing normalizing factors ...')
        cprint("Do it only on training sequences, it is vital!", 'yellow')
        # first compute mean
        num_data = 0

        for i, sequence in enumerate(train_seqs):
            pickle_dict = pload(self.predata_dir, sequence + '.p')
            us = pickle_dict['us']
            sms = pickle_dict['xs']
            if i == 0:
                mean_u = us.sum(dim=0)
                num_positive = sms.sum(dim=0)
                num_negative = sms.shape[0] - sms.sum(dim=0)
            else:
                mean_u += us.sum(dim=0)
                num_positive += sms.sum(dim=0)
                num_negative += sms.shape[0] - sms.sum(dim=0)
            num_data += us.shape[0]
        mean_u = mean_u / num_data
        pos_weight = num_negative / num_positive

        # second compute standard deviation
        for i, sequence in enumerate(train_seqs):
            pickle_dict = pload(self.predata_dir, sequence + '.p')
            us = pickle_dict['us']
            if i == 0:
                std_u = ((us - mean_u) ** 2).sum(dim=0)
            else:
                std_u += ((us - mean_u) ** 2).sum(dim=0)
        std_u = (std_u / num_data).sqrt()
        normalize_factors = {
            'mean_u': mean_u,
            'std_u': std_u,
        }
        print('... ended computing normalizing factors')
        print('pos_weight:', pos_weight)
        print('This values most be a training parameters !')
        print('mean_u    :', mean_u)
        print('std_u     :', std_u)
        print('num_data  :', num_data)
        pdump(normalize_factors, self.path_normalize_factors)
        return mean_u, std_u

    def read_data(self, data_dir):
        raise NotImplementedError

    @staticmethod
    def interpolate(x, t, t_int):
        """
        Interpolate ground truth at the sensor timestamps
        """

        # vector interpolation
        x_int = np.zeros((t_int.shape[0], x.shape[1]))
        for i in range(x.shape[1]):
            if i in [4, 5, 6, 7]:
                continue
            x_int[:, i] = np.interp(t_int, t, x[:, i])
        # quaternion interpolation
        t_int = torch.Tensor(t_int - t[0])
        t = torch.Tensor(t - t[0])
        qs = SO3.qnorm(torch.Tensor(x[:, 4:8]))
        x_int[:, 4:8] = SO3.qinterp(qs, t, t_int).numpy()
        return x_int


class EUROCDataset(BaseDataset):
    """
        Dataloader for the EUROC Data Set.
    """

    def __init__(self, data_dir, predata_dir, train_seqs, val_seqs,
                test_seqs, mode, N, min_train_freq, max_train_freq, dt=0.005):
        super().__init__(predata_dir, train_seqs, val_seqs, test_seqs, mode, N, min_train_freq, max_train_freq, dt)
        self.data_dir = data_dir
        self.DEST_DIR = predata_dir
        self.IN_DIR = predata_dir
        self.OUT_DIR = os.path.join(self.DEST_DIR, "estimate")
        self.GT_DIR = os.path.join(self.DEST_DIR, "gt")

        if mode == 'parse':
            self.sequences = ['MH_02_easy', 'MH_04_difficult']
        self.gt_processed = {} # np.ndarray
        self.imu_processed = {}
        self.min_train_freq = min_train_freq

        for sequence in self.sequences:
            self.preprocess(sequence)
            if self.mode == 'parse':
                self.save_csv(sequence)

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

    def preprocess(self, sequence):
        # if self.mode is not 'parse':
        #     return

        print("Preprocess -- %s" % sequence)
        path_imu = os.path.join(self.data_dir, sequence, "mav0", "imu0", "data.csv")
        path_gt = os.path.join(self.data_dir, sequence, "mav0", "state_groundtruth_estimate0", "data.csv")

        predata_path = os.path.join(self.predata_dir, sequence + '.p')
        predata_gt_path = os.path.join(self.predata_dir, sequence + '_gt.p')
        if self.mode is not 'parse' and os.path.exists(predata_path) and os.path.exists(predata_gt_path):
            print("\tPass")
            return

        imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1)
        gt = np.genfromtxt(path_gt, delimiter=",", skip_header=1)

        # time synchronization between IMU and ground truth
        t0 = np.max([gt[0, 0], imu[0, 0]])
        t_end = np.min([gt[-1, 0], imu[-1, 0]])

        # start index
        idx0_imu = np.searchsorted(imu[:, 0], t0)
        idx0_gt = np.searchsorted(gt[:, 0], t0)

        # end index
        idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
        idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')

        # subsample
        imu = imu[idx0_imu: idx_end_imu]
        gt = gt[idx0_gt: idx_end_gt]
        ts = imu[:, 0]/1e9

        # interpolate
        gt = self.interpolate(gt, gt[:, 0]/1e9, ts)
        self.gt_processed[sequence] = np.copy(gt)
        self.imu_processed[sequence] = np.copy(imu)

        # take ground truth position
        p_gt = gt[:, 1:4]
        p_gt = p_gt - p_gt[0]

        # take ground true quaternion pose
        q_gt = torch.Tensor(gt[:, 4:8]).double()
        q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
        Rot_gt = SO3.from_quaternion(q_gt.cuda(), ordering='wxyz').cpu()

        # convert from numpy
        p_gt = torch.Tensor(p_gt).double()
        v_gt = torch.tensor(gt[:, 8:11]).double()
        imu = torch.Tensor(imu[:, 1:]).double()

        # compute pre-integration factors for all training
        mtf = self.min_train_freq
        dRot_ij = bmtm(Rot_gt[:-mtf], Rot_gt[mtf:])
        dRot_ij = SO3.dnormalize(dRot_ij.cuda())
        dxi_ij = SO3.log(dRot_ij).cpu()
        # print("\tdxi_ij -- ", dxi_ij.shape, dxi_ij.dtype) # [29976, 3]

        delta_v_gt = v_gt[:-mtf] - v_gt[mtf:]

        # save for all training
        print("\tSaved in pickle file \'%s\'" % predata_path)
        print("\txs:", dxi_ij.shape, dxi_ij.dtype)
        print("\tus:", imu.shape, imu.dtype)
        print("\tdv_gt:", delta_v_gt.shape, delta_v_gt.dtype)
        mondict = {
            'xs': dxi_ij.float(),
            'us': imu.float(),
            'dv_gt': delta_v_gt.float(),
        }
        pdump(mondict, self.predata_dir, sequence + ".p")

        # save ground truth
        print("\tSaved in pickle file \'%s\'" % predata_gt_path)
        mondict = {
            'ts': ts,
            'qs': q_gt.float(),
            'vs': v_gt.float(),
            'ps': p_gt.float(),
        }
        pdump(mondict, self.predata_dir, sequence + "_gt.p")

if __name__ == '__main__':
    EUROC_DATASET_PARAMS = {
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

    dataset_train = EUROCDataset(**EUROC_DATASET_PARAMS, mode='parse')
