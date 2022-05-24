import sys

from scipy.fftpack import next_fast_len
if sys.platform.startswith('win'):
    sys.path.append(r"C:\Users\leech\Desktop\imu_ws\denoise-imu-gyro") # My window workspace path
elif sys.platform.startswith('linux'):
    sys.path.append('/root/denoise')

from src.utils import pdump, pload, bmtv, bmtm
from src.lie_algebra import SO3
from termcolor import cprint
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import torch
import sys

class DGADataset(Dataset):

    def __init__(self, params, mode): # train_seqs, val_seqs, test_seqs, data_dir, predata_dir, N, min_train_freq, max_train_freq, v_window,
        super().__init__()
        print('\n# Initialize dataset for %s ...' % mode)

        self.seq_dict = {
            'train': params['dataset']['train_seqs'],
            'val':   params['dataset']['val_seqs'],
            'test':  params['dataset']['test_seqs'],
        }

        # train_seqs, self.sequences = self.get_sequences(train_seqs, val_seqs, test_seqs)
        self.data_dir       = params['dataset']['data_dir']
        self.predata_dir    = params['dataset']['predata_dir']
        self.N              = params['dataset']['N']
        self.min_train_freq = params['dataset']['min_train_freq']
        self.max_train_freq = params['dataset']['max_train_freq']
        self.dv_normed      = params['train']['loss']['dv_normed']
        self.mode = mode
        self.sequences = self.seq_dict[mode]
        self.dt = 0.05

        self.nf_path = os.path.join(self.predata_dir, 'nf.pt') # the path of normalized factor
        self.mean_u, self.std_u = self.init_normalize_factors(self.seq_dict['train'])

        ## Additional
        # noise density
        self.imu_std = torch.Tensor([8e-5, 1e-3]).float()
        # bias repeatability (without in-run bias stability)
        self.imu_b0 = torch.Tensor([1e-3, 1e-3]).float()
        # sequence size during training
        self.uni = torch.distributions.uniform.Uniform(-torch.ones(1), torch.ones(1))
        print('--- init success ---')

    def __getitem__(self, i):
        seq        = self.sequences[i]

        # data       = torch.load(os.path.join(self.predata_dir, seq, 'data.pt'))
        # gt_dict    = torch.load(os.path.join(self.predata_dir, seq, 'gt.pt'))
        # gt_dv_dict = torch.load(os.path.join(self.predata_dir, seq, 'dv.pt'))

        # ts = data['ts'].cuda()
        # us = data['us'].cuda()

        # dw_16   = gt_dict['dw_16'].cuda()
        # gt      = gt_dict['gt_interpolated'].cuda()
        # a_gt    = gt_dict['a_gt'].cuda()

        # gt_dv_normed = gt_dv_dict['dv_normed']

        fname = os.path.join(self.predata_dir, seq, 'imu.csv')
        imu = np.loadtxt(fname, delimiter=',')
        imu = torch.from_numpy(imu).cuda().float()
        ts = imu[:, 0]
        us = imu[:, 1:]

        fname = os.path.join(self.predata_dir, seq, 'q_gt.csv')
        q_gt = np.loadtxt(fname, delimiter=',')
        q_gt = torch.from_numpy(q_gt).cuda().float()

        fname = os.path.join(self.predata_dir, seq, 'w_gt.csv')
        w_gt = np.loadtxt(fname, delimiter=',')
        w_gt = torch.from_numpy(w_gt).cuda()

        fname = os.path.join(self.predata_dir, seq, 'dv_16_gt.csv')
        dv_16_gt = np.loadtxt(fname, delimiter=',')
        dv_16_gt = torch.from_numpy(dv_16_gt).cuda().float()

        fname = os.path.join(self.predata_dir, seq, 'dv_32_gt.csv')
        dv_32_gt = np.loadtxt(fname, delimiter=',')
        dv_32_gt = torch.from_numpy(dv_32_gt).cuda().float()

        fname = os.path.join(self.predata_dir, seq, 'dw_16_gt.csv')
        dw_16_gt = np.loadtxt(fname, delimiter=',')
        dw_16_gt = torch.from_numpy(dw_16_gt).cuda().float()

        fname = os.path.join(self.predata_dir, seq, 'dw_32_gt.csv')
        dw_32_gt = np.loadtxt(fname, delimiter=',')
        dw_32_gt = torch.from_numpy(dw_32_gt).cuda().float()

        dv_normed_windows = [16, 32, 64, 128, 256, 512] #
        dv_normed_dict = {}
        for window in dv_normed_windows:
            fname = os.path.join(self.predata_dir, seq, 'dv_normed_%d_gt.csv' % window)
            val = np.loadtxt(fname, delimiter=',')
            val = torch.from_numpy(val).cuda().float()
            dv_normed_dict[str(window)] = val

        fname = os.path.join(self.predata_dir, seq, 'w_mean.csv')
        w_mean = np.loadtxt(fname, delimiter=',')
        w_mean = torch.from_numpy(w_mean).cuda().float()

        fname = os.path.join(self.predata_dir, seq, 'w_std.csv')
        w_std = np.loadtxt(fname, delimiter=',')
        w_std = torch.from_numpy(w_std).cuda().float()

        fname = os.path.join(self.predata_dir, seq, 'a_mean.csv')
        a_mean = np.loadtxt(fname, delimiter=',')
        a_mean = torch.from_numpy(a_mean).cuda().float()

        fname = os.path.join(self.predata_dir, seq, 'a_std.csv')
        a_std = np.loadtxt(fname, delimiter=',')
        a_std = torch.from_numpy(a_std).cuda().float()

        ## GET Range
        N_max = dv_32_gt.shape[0]
        # random start
        if self.mode == 'train':
            _max = N_max - self.N
            n0 = torch.randint(0, _max, (1, ))
            nend = n0 + self.N
        # full sequence
        else:
            n0 = 0
            nend = N_max - (N_max % self.max_train_freq)

        ## CROP
        ts       = ts[n0: nend]
        us       = us[n0: nend]
        q_gt     = q_gt[n0:nend]
        w_gt     = w_gt[n0:nend]
        dw_16_gt = dw_16_gt[n0:nend]
        dw_32_gt = dw_32_gt[n0:nend]
        dv_16_gt = dv_16_gt[n0: nend-16]
        dv_32_gt = dv_32_gt[n0: nend-32]
        for key, value in dv_normed_dict.items():
            dv_normed_dict[key] = value[n0:nend]

        if self.mode == 'train':
            assert us.shape[0] == self.N

        return \
            seq, us, q_gt, w_gt, \
            dw_16_gt, dw_32_gt, dv_16_gt, dv_32_gt, dv_normed_dict, \
            w_mean, w_std, a_mean, a_std

    def __len__(self):
        return len(self.seq_dict[self.mode])

    def add_noise(self, u):
        """Add Gaussian noise and bias to input"""
        noise = torch.randn_like(u)
        noise[:, :, :3] = noise[:, :, :3] * self.imu_std[0]
        noise[:, :, 3:6] = noise[:, :, 3:6] * self.imu_std[1]

        # bias repeatability (without in-run bias stability)
        b0 = self.uni.sample(u[:, 0].shape).cuda()
        b0[:, :, :3] = b0[:, :, :3] * self.imu_b0[0]
        b0[:, :, 3:6] =  b0[:, :, 3:6] * self.imu_b0[1]
        u = u + noise + b0.transpose(1, 2)
        return u

    def init_normalize_factors(self, seqs):
        if os.path.exists(self.nf_path):
            print('# Load nf.pt ...')
            nf = torch.load(self.nf_path)
            if seqs != nf['seqs']:
                print('[FATAL] normalized factor is derived from different sequences!!!')
                print('FROM:')
                for seq in nf['seqs']:
                    print('- %s' % seq)
                print('NOW SEQS:')
                for seq in seqs:
                    print('- %s' % seq)
                exit(-1)
            print('- mean_u:', nf['mean_u'])
            print('- std_u:', nf['std_u'])
            return nf['mean_u'], nf['std_u']

        print('\n#Compute nf.pt (normalized factors) ...')
        print('- Do this only once on training time -')

        # Compute mean
        n_data = 0
        mean_seqs = torch.zeros(6).cuda()
        for seq in seqs:
            fname = os.path.join(self.predata_dir, seq, 'imu.csv')
            imu = np.loadtxt(fname, delimiter=',')
            imu = torch.from_numpy(imu).cuda()
            us = imu[:, 1:]
            mean_seqs += us.sum(dim=0)
            n_data    += us.shape[0]
        mean_seqs = mean_seqs / n_data
        # print('mean_seqs:', mean_seqs.shape, mean_seqs.dtype, mean_seqs.device)

        # Compute standard deviation
        std_u = torch.zeros(6).cuda()
        for seq in seqs:
            fname = os.path.join(self.predata_dir, seq, 'imu.csv')
            imu = np.loadtxt(fname, delimiter=',')
            imu = torch.from_numpy(imu).cuda()
            us = imu[:, 1:]
            std_u += ((us - mean_seqs) ** 2).sum(dim=0)
        std_u = (std_u / n_data).sqrt()
        # print('std_u:', std_u.shape, std_u.dtype, std_u.device)

        mean_seqs = mean_seqs.cpu().float()
        std_u = std_u.cpu().float()
        nf = {
            'mean_u': mean_seqs,
            'std_u': std_u,
            'seqs': seqs
        }
        torch.save(nf, self.nf_path)
        return mean_seqs, std_u


params = {
    'dataset': {
        'predata_dir': './results/predata',
        'data_dir': './data',
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
    },
    'train': {
        'loss': {
            'min_N': 4, # int(np.log2(dataset_params['min_train_freq'])),
            'max_N': 5, # int(np.log2(dataset_params['max_train_freq'])),
            'w':  1e6,
            'huber': 0.005,
            'dt': 0.005,
            'dv': [16, 32],
            'dv_normed': [32, 64],
        },
    },
    'debug': True,
}

if __name__ == '__main__':
    dataset_train = DGADataset(params, mode='train')
    dataset_test  = DGADataset(params, mode='test')

    print('\n Train Set')
    for seq, us, gt, dw_16, dv, dv_normed in dataset_train:
        print('%s:'%seq)
        print('  us:', us.shape, us.dtype, us.device)
        print('  gt:', gt.shape, gt.dtype, gt.device)
        print('  dw_16:', dw_16.shape, dw_16.dtype, dw_16.device)
        for key, value in dv.items():
            print('  dv[%s]:'%key, value.shape, value.dtype, value.device)
        for key, value in dv_normed.items():
            print('  dv_normed[%s]:'%key, value.shape, value.dtype, value.device)

    print('\n Test Set')
    for seq, us, gt, dw_16, dv, dv_normed in dataset_test:
        print('%s:'%seq)
        print('  us:', us.shape, us.dtype, us.device)
        print('  gt:', gt.shape, gt.dtype, gt.device)
        print('  dw_16:', dw_16.shape, dw_16.dtype, dw_16.device)
        for key, value in dv.items():
            print('  dv[%s]:'%key, value.shape, value.dtype, value.device)
        for key, value in dv_normed.items():
            print('  dv_normed[%s]:'%key, value.shape, value.dtype, value.device)
