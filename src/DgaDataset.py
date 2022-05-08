import sys
sys.path.append('/root/project')
from src.utils import pdump, pload, bmtv, bmtm
from src.lie_algebra import SO3
from termcolor import cprint
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import torch
import sys
import matplotlib.pyplot as plt

class DgaDataset(Dataset):

    def __init__(self, params, mode): # train_seqs, val_seqs, test_seqs, data_dir, predata_dir, N, min_train_freq, max_train_freq, v_window,
        super().__init__()
        print('\n# Initialize dataset for %s ...' % mode)

        self.params = params
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
        self.dt = 0.005

        ## Additional
        # noise density
        self.imu_std = torch.Tensor([8e-5, 1e-3]).float()
        # bias repeatability (without in-run bias stability)
        self.imu_b0 = torch.Tensor([1e-3, 1e-3]).float()
        # sequence size during training
        self.uni = torch.distributions.uniform.Uniform(-torch.ones(1), torch.ones(1))
        print('--- init success ---')

        print('--- preprocess ...')
        for seq in self.sequences:
            self.preprocess(seq)
        print('--- success')

    def preprocess(self, seq):
        # print('Sequence %s pre-process' % seq)

        _dest_path = os.path.join(self.predata_dir, seq)
        if not os.path.exists(_dest_path):
            os.makedirs(_dest_path)

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
            qs = SO3.qinterp(qs, t, t_int)
            qs = SO3.qnorm(qs)
            x_int[:, 4:8] = qs.numpy()
            return x_int

        path_imu = os.path.join(self.params['dataset']['data_dir'], seq, "mav0", "imu0", "data.csv")
        imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1)
        path_gt = os.path.join(self.params['dataset']['data_dir'], seq, "mav0", "state_groundtruth_estimate0", "data.csv")
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

        gt_interpolated = interpolate(gt, gt[:, 0]/1e9, ts)
        gt_interpolated[:, 0] = imu[:, 0]

        # take ground truth position
        p_gt = gt_interpolated[:, 1:4]
        p_gt = p_gt - p_gt[0]

        # take ground true quaternion pose
        q_gt = torch.Tensor(gt_interpolated[:, 4:8]).double()
        q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
        Rot_gt = SO3.from_quaternion(q_gt.cuda(), ordering='wxyz').cpu()
        Rot_gt = SO3.dnormalize(Rot_gt.cuda())

        w_gt_path = os.path.join(self.predata_dir, seq, 'w_gt.csv')
        if not os.path.exists(w_gt_path):
            w_gt = bmtm(Rot_gt[:-1], Rot_gt[1:])
            w_gt = SO3.dnormalize(w_gt.double())
            w_gt = SO3.log(w_gt)
            w_gt = w_gt.cpu().numpy() / self.dt
            print('w_gt:',w_gt.shape, w_gt.dtype)
            np.savetxt(w_gt_path, w_gt, delimiter=',')

        imu_path = os.path.join(self.predata_dir, seq, 'imu.csv')
        if not os.path.exists(imu_path):
            print('imu:',imu.shape, imu.dtype)
            np.savetxt(imu_path, imu, delimiter=',')

        dw_16_gt_path = os.path.join(self.predata_dir, seq, 'dw_16_gt.csv')
        if not os.path.exists(dw_16_gt_path):
            dw_16_gt = bmtm(Rot_gt[:-16], Rot_gt[16:])
            dw_16_gt = SO3.dnormalize(dw_16_gt.double())
            dw_16_gt = SO3.log(dw_16_gt)
            dw_16_gt = dw_16_gt.cpu().numpy()
            print('dw_16_gt:',dw_16_gt.shape, dw_16_gt.dtype)
            np.savetxt(dw_16_gt_path, dw_16_gt, delimiter=',')

        dw_32_gt_path = os.path.join(self.predata_dir, seq, 'dw_32_gt.csv')
        if not os.path.exists(dw_32_gt_path):
            dw_32_gt = bmtm(Rot_gt[:-32], Rot_gt[32:])
            dw_32_gt = SO3.dnormalize(dw_32_gt.double())
            dw_32_gt = SO3.log(dw_32_gt)
            dw_32_gt = dw_32_gt.cpu().numpy()
            print('dw_32_gt:',dw_32_gt.shape, dw_32_gt.dtype)
            np.savetxt(dw_32_gt_path, dw_32_gt, delimiter=',')

        q_gt_path = os.path.join(self.predata_dir, seq, 'q_gt.csv')
        if not os.path.exists(q_gt_path):
            q_gt = q_gt.cpu().numpy()
            print('q_gt:',q_gt.shape, q_gt.dtype)
            np.savetxt(q_gt_path, q_gt, delimiter=',')

        if False: # gt quat 확인
            fname = os.path.join(self.predata_dir, seq, 'q_gt.csv')
            if os.path.exists(fname):
                # print('- %s is already exists.' % fname)
                return

            rpy_gt = SO3.to_rpy(Rot_gt).cpu().numpy()
            def rad2deg(x):
                return x * (180. / np.pi)
            self.plot_orientation(seq, rad2deg(rpy_gt), rad2deg(rpy_gt))

            ## SAVE
            np.savetxt(fname, q_gt, delimiter=',')
            # _save_dict = {
            #     'ts': imu[:, 0].float().cpu(),
            #     'us': imu[:, 1:].float().cpu(),
            #     'gt_interpolated': gt_interpolated.float().cpu()
            # }; torch.save(_save_dict, _save_path)
            print('- save to  %s' % fname)


        if False: # Data
            path_imu = os.path.join(self.params['dataset']['data_dir'], seq, "mav0", "imu0", "data.csv")
            imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1)
            path_gt = os.path.join(self.params['dataset']['data_dir'], seq, "mav0", "state_groundtruth_estimate0", "data.csv")
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

            gt_interpolated = interpolate(gt, gt[:, 0]/1e9, ts)
            gt_interpolated[:, 0] = imu[:, 0]

            # take ground truth position
            p_gt = gt_interpolated[:, 1:4]
            p_gt = p_gt - p_gt[0]

            # take ground true quaternion pose
            q_gt = torch.Tensor(gt_interpolated[:, 4:8]).double()
            q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
            Rot_gt = SO3.from_quaternion(q_gt.cuda(), ordering='wxyz').cpu()

            # convert from numpy
            p_gt = torch.Tensor(p_gt).double()
            v_gt = torch.tensor(gt_interpolated[:, 8:11]).double()
            imu = torch.Tensor(imu).double()
            gt_interpolated = torch.Tensor(gt_interpolated)

            ## SAVE
            _save_path = os.path.join(self.predata_dir, seq, name)
            _save_dict = {
                'ts': imu[:, 0].float().cpu(),
                'us': imu[:, 1:].float().cpu(),
                'gt_interpolated': gt_interpolated.float().cpu()
            }; torch.save(_save_dict, _save_path)
            print('- save %s' % _save_path)


    def plot_orientation(self, seq, rpy_hat:np.ndarray, rpy_gt:np.ndarray):
        id = 'dataset q_gt test'
        figure_dir = os.path.join('/root/project/results/LTC/figures', seq, 'rpy')
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 9), dpi=250)
        fig.suptitle('$SO(3)$ Orientation Estimation / %s / %s' % (seq, id), fontsize=20)

        axs[0].set(ylabel='roll (deg)', title='Orientation estimation')
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        N = rpy_hat.shape[0]
        ts = list(range(N))
        ts = np.array(ts) * 0.005
        rpy_gt = rpy_gt[:N]
        for i in range(3):
            axs[i].plot(ts, rpy_gt[:, i]%360, color='black', label=r'ground truth')
            axs[i].set_xlim(ts[0], ts[-1])

        _path = os.path.join(figure_dir, id + '.png')
        self.savefig(axs, fig, _path)
        plt.close(fig)

    def savefig(self, axs, fig, path):
        if isinstance(axs, np.ndarray):
            for i in range(len(axs)):
                axs[i].grid()
                axs[i].legend()
        else:
            axs.grid()
            axs.legend()
        fig.savefig(path)

    def __getitem__(self, i):
        seq = self.sequences[i]

        fname = os.path.join(self.predata_dir, seq, 'imu.csv')
        imu = np.loadtxt(fname, delimiter=',')
        imu = torch.from_numpy(imu).cuda()
        ts = imu[:, 0]
        us = imu[:, 1:]

        fname = os.path.join(self.predata_dir, seq, 'q_gt.csv')
        q_gt = np.loadtxt(fname, delimiter=',')
        q_gt = torch.from_numpy(q_gt).cuda()

        # fname = os.path.join(self.predata_dir, seq, 'w_gt.csv')
        # w_gt = np.loadtxt(fname, delimiter=',')
        # w_gt = torch.from_numpy(w_gt).cuda()

        fname = os.path.join(self.predata_dir, seq, 'dw_16_gt.csv')
        dw_16_gt = np.loadtxt(fname, delimiter=',')
        dw_16_gt = torch.from_numpy(dw_16_gt).cuda()

        fname = os.path.join(self.predata_dir, seq, 'dw_32_gt.csv')
        dw_32_gt = np.loadtxt(fname, delimiter=',')
        dw_32_gt = torch.from_numpy(dw_32_gt).cuda()

        # # Check
        # print('[%s Dataload]' % seq)
        # print('- ts:', ts.shape, ts.device)
        # print('- us:', us.shape, us.device)
        # print('- q_gt:', q_gt.shape, q_gt.device)
        # print('- w_gt:', w_gt.shape, w_gt.device)


        ## GET Range
        N_max = dw_32_gt.shape[0]
        if self.mode == 'train': # random start
            _max = N_max - self.N
            n0 = torch.randint(0, _max, (1, ))
            nend = n0 + self.N
            # print('n0: %d, nend: %d' % (n0, nend))
        else:  # full sequence
            n0 = 0
            nend = N_max - (N_max % self.max_train_freq)

        # print('%s mode -- crop [%d, %d)' % (self.mode, n0, nend))

        ## CROP
        ts       = ts[n0: nend]
        us       = us[n0: nend]
        # w_gt     = w_gt[n0: nend]
        dw_16_gt = dw_16_gt[n0: nend]
        dw_32_gt = dw_32_gt[n0: nend]
        q_gt     = q_gt[n0: nend]

        if self.mode == 'train':
            us = self.add_noise(us.unsqueeze(0)).squeeze(0)
            return us, dw_16_gt, dw_32_gt
        else:
            return seq, ts, us, dw_16_gt, dw_32_gt, q_gt

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

    def get_dataset_nf(self):
        """Compute train or test datasets' normalize factors (mean, std)"""
        nf_path = os.path.join(self.predata_dir, 'nf_%s.csv' % self.mode)

        if os.path.exists(nf_path):
            print('\nLoad %s datasets\' nf ...' % self.mode)
            nf = np.loadtxt(nf_path, delimiter=',')
            _mean = torch.from_numpy(nf[:, 0].squeeze()).float()
            _std  = torch.from_numpy(nf[:, 1].squeeze()).float()
            return _mean, _std

        else:
            print('\nCompute %s datasets\' nf ...' % self.mode)

            ## Compute mean
            n_data = 0
            mean_us = torch.zeros(6)
            for seq in self.sequences:
                fname = os.path.join(self.predata_dir, seq, 'imu.csv')
                imu = np.loadtxt(fname, delimiter=',')
                imu = torch.from_numpy(imu).float()
                # print('%s imu:' % seq, imu.shape)
                us = imu[:, 1:]
                mean_us += us.sum(dim=0)
                n_data    += us.shape[0]
            mean_us = mean_us / n_data
            # print('mean_us:', mean_us.shape, mean_us.dtype, mean_us.device)

            ## Compute standard deviation
            std_us = torch.zeros(6)
            # print('std_us:', std_us.shape, std_us.dtype, std_us.device)
            for seq in self.sequences:
                fname = os.path.join(self.predata_dir, seq, 'imu.csv')
                imu = np.loadtxt(fname, delimiter=',')
                imu = torch.from_numpy(imu).float()
                us = imu[:, 1:]
                std_us += ((us - mean_us) ** 2).sum(dim=0)
            std_us = (std_us / n_data).sqrt()
            # print('std_us:', std_us.shape, std_us.dtype, std_us.device)

            nf = torch.stack([mean_us, std_us], dim=1).numpy()
            # print('nf :', nf, nf.shape) # Shape (6, 2)
            np.savetxt(nf_path, nf, delimiter=',', header="# mean, std")
            return mean_us, std_us

euroc_data_dir = os.path.join('/root', 'project', 'datasets', 'EUROC')
ltc_results_dir = os.path.join('/root', 'project', 'results', 'LTC')


params = {
    'dataset': {
        'train_seqs': [
            'MH_01_easy',
            'MH_03_medium',
            'MH_05_difficult',
            'V1_02_medium',
            'V2_01_easy',
            'V2_03_difficult'
        ],
        'val_seqs': [
            'MH_02_easy',
            'MH_04_difficult',
            'V2_02_medium',
            'V1_03_difficult',
            'V1_01_easy',
        ],
        'test_seqs': [
            'MH_02_easy',
            'MH_04_difficult',
            'V2_02_medium',
            'V1_03_difficult',
            'V1_01_easy',
        ],

        # size of trajectory during training
        'data_dir':     euroc_data_dir,
        'predata_dir':  os.path.join(ltc_results_dir, 'predata'),
        'N': 32 * 100, # should be integer * 'max_train_freq'
        'min_train_freq': 16,
        'max_train_freq': 32,
        'v_window': 16,
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
    dataset_train = DgaDataset(params, mode='train')
    dataset_test  = DgaDataset(params, mode='test')

    print('\n Train Set')
    for us, w_gt in dataset_train:
        print('- us:', us.shape, us.dtype, us.device)
        print('- w_gt:', w_gt.shape, w_gt.dtype, w_gt.device)

    print('\n Test Set')
    for seq, ts, us, w_gt, q_gt in dataset_test:
        print('%s:'%seq)
        print('  ts:', ts.shape, ts.dtype, ts.device)
        print('  us:', us.shape, us.dtype, us.device)
        print('  w_gt:', w_gt.shape, w_gt.dtype, w_gt.device)
        print('  q_gt:', q_gt.shape, q_gt.dtype, q_gt.device)
