import sys
sys.path.append('/root/denoise')

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os

from src.lie_algebra import SO3
from src.utils import bmtm, bmtv, bmmt, bbmv

euroc_seqs = [
    'MH_01_easy',
    'MH_02_easy',
    'MH_03_medium',
    'MH_04_difficult',
    'MH_05_difficult',
    'V1_01_easy',
    'V1_02_medium',
    'V1_03_difficult',
    'V2_01_easy',
    'V2_02_medium',
    'V2_03_difficult'
]

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

class Data:
    def __init__(self, seq):
        super().__init__()
        # if not (seq in euroc_seqs):
        #     print('[FATAL] -- Not existing dataset name.')
        #     exit(1)
        # else:
        #     print('[Init] -- Data instance is created for %s' % seq)

        self.root = 'datasets/EUROC'
        self.path_seq = os.path.join(self.root, seq, 'mav0')
        self.path_dir_cam0 = os.path.join(self.path_seq, 'cam0')
        self.path_dir_imu  = os.path.join(self.path_seq, 'imu0')
        self.path_dir_gt   = os.path.join(self.path_seq, 'state_groundtruth_estimate0')
        self.path_dir_result   = os.path.join('./results', 'Figures', seq)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.seq = seq

        if not os.path.exists(self.path_dir_result):
            os.makedirs(self.path_dir_result)

        cam0 = np.genfromtxt(os.path.join(self.path_dir_cam0, 'data.csv'), delimiter=",", skip_header=1)
        imu  = np.genfromtxt(os.path.join(self.path_dir_imu, 'data.csv'), delimiter=",", skip_header=1)
        gt   = np.genfromtxt(os.path.join(self.path_dir_gt, 'data.csv'), delimiter=",", skip_header=1)

        ## Time synchronization between IMU and GT
        t0 = np.max([gt[0, 0], imu[0, 0], cam0[0, 0]])
        tN = np.min([gt[-1, 0], imu[-1, 0], cam0[-1, 0]])
        idx0_cam0 = np.searchsorted(cam0[:, 0], t0)
        idx0_imu  = np.searchsorted(imu[:, 0], t0)
        idx0_gt   = np.searchsorted(gt[:, 0], t0)
        idx_end_cam0 = np.searchsorted(cam0[:, 0], tN, 'right')
        idx_end_imu  = np.searchsorted(imu[:, 0], tN, 'right')
        idx_end_gt   = np.searchsorted(gt[:, 0], tN, 'right')
        self.cam0 = cam0[idx0_cam0: idx_end_cam0]
        self.imu = imu[idx0_imu: idx_end_imu]
        self.gt = gt[idx0_gt: idx_end_gt]

        self.gt_int = interpolate(self.gt, self.gt[:, 0]/1e9, self.imu[:, 0]/1e9)
        self.gt_int[:, 0] = self.imu[:, 0]

        self.g = torch.tensor([0, 0, 9.81], device=self.device, requires_grad=False)

    def get_gap_distribution(self, start=None, end=None, plot=False):
        print('[%s] Gap Distribution' % self.seq)
        q_gt = torch.tensor(self.gt_int[:, 4:8], device=self.device)
        q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
        rot_gt = SO3.from_quaternion(q_gt, ordering='wxyz')
        N = rot_gt.shape[0]

        rot_tmp = bmtm(rot_gt[:-1], rot_gt[1:])
        rot_tmp = SO3.dnormalize(rot_tmp.double())
        q_tmp = SO3.to_quaternion(rot_tmp)
        q_tmp = SO3.qnorm(q_tmp).cpu()
        _t = torch.from_numpy(np.linspace(1.0, float(N-1), N-1)).cpu().double()
        _t_int = _t[:-1] + 0.5
        q_int_tmp = SO3.qinterp(q_tmp.cpu(), _t, _t_int)
        q_int_tmp = SO3.qnorm(q_int_tmp)
        q_tmp = torch.cat([q_tmp[0].unsqueeze(0), q_int_tmp, q_tmp[-1].unsqueeze(0)])
        rot_tmp = SO3.from_quaternion(q_tmp.cuda(), ordering='wxyz')
        w_gt = SO3.log(rot_tmp) / 0.005
        assert w_gt.shape[0] == q_gt.shape[0]

        v_gt = torch.tensor(self.gt_int[:, 8:11], device=self.device)
        a_gt = (v_gt[1:] - v_gt[:-1]) / 0.005
        a_gt = torch.cat([a_gt[0].unsqueeze(0), (a_gt[1:] + a_gt[:-1]) / 2.0, a_gt[-1].unsqueeze(0)])

        ts = torch.tensor(self.gt_int[:, 0], device=self.device)

        a_raw = torch.tensor(self.imu[:, 4:7], device=self.device)
        a_raw = torch.einsum('bij, bj -> bi', rot_gt, a_raw) - self.g
        w_raw = torch.tensor(self.imu[:, 1:4], device=self.device)

        ## When you want to see analysis values by specific section
        if start is not None and end is not None:
            i0 = np.searchsorted(ts, start)
            iN = np.searchsorted(ts, end, side='right')
            print('Clip %d to %d' % (i0, iN))

            a_raw = a_raw[i0:iN]
            w_raw = w_raw[i0:iN]
            v_gt = v_gt[i0:iN]
            a_gt = a_gt[i0:iN]
            w_gt = w_gt[i0:iN]
            q_gt = q_gt[i0:iN]
            ts = ts[i0:iN]

        ## Compute a_gap
        a_gap = a_gt - a_raw
        a_gap_std, a_gap_mean = torch.std_mean(a_gap, dim=0)
        a_gap = a_gap.cpu().numpy()
        a_gap_std = a_gap_std.cpu().numpy()
        a_gap_mean = a_gap_mean.cpu().numpy()
        print('%12s'%'a_gap_std:', a_gap_std, a_gap_std.shape)
        print('%12s'%'a_gap_mean:', a_gap_mean, a_gap_mean.shape)

        ## Compute w_gap
        w_gap = w_gt - w_raw
        w_gap_std, w_gap_mean = torch.std_mean(w_gap, dim=0)
        w_gap = w_gap.cpu().numpy()
        w_gap_std = w_gap_std.cpu().numpy()
        w_gap_mean = w_gap_mean.cpu().numpy()
        print('%12s'%'w_gap_std:', w_gap_std, w_gap_std.shape)
        print('%12s'%'w_gap_mean:', w_gap_mean, w_gap_mean.shape)

        ## Compute Gaussian Negative Log Likelihood (NLL) Loss
        def gaussian_nll_loss(gap, mu, sigma, eps=1e-6):
            var = sigma**2
            var[var<eps] = eps
            _first = np.log(var)
            _second = (gap-mu)**2 / var
            _loss = (_first + _second) / 2.0
            _loss = np.mean(_loss, axis=0)
            return _loss
        a_gap_nll_loss = gaussian_nll_loss(a_gap, a_gap_mean, a_gap_std)
        w_gap_nll_loss = gaussian_nll_loss(w_gap, w_gap_mean, w_gap_std)

        ##
        equal_to_torch_gaussian_nll_loss = False
        if equal_to_torch_gaussian_nll_loss:
            Loss = torch.nn.GaussianNLLLoss()
            t_a_gap = torch.from_numpy(a_gap)
            output = Loss(
                t_a_gap,
                torch.from_numpy(a_gap_mean).expand_as(t_a_gap),
                torch.from_numpy(a_gap_std**2).expand_as(t_a_gap)).numpy()
            assert np.abs(output.item() - a_gap_nll_loss.sum().item()) < 1e-8

        if plot:
            ### Visualize Accel Gap
            fig, ax = plt.subplots(3, 1)
            fig.suptitle('%s / Accel Gap histogram' % self.seq)

            N = a_gap.shape[0]
            n_bins = 500

            def gaussian(x, mu, sig):
                return (1. / (sig * np.sqrt(2 * np.pi))) * np.exp(- (x - mu)**2 / (2 * sig**2))

            count, bins, _ = ax[0].hist(a_gap[:, 0], bins=n_bins, density=True, histtype='step', color='k', label='a_gap x')
            ax[0].plot(bins, gaussian(bins, a_gap_mean[0], a_gap_std[0]), color='r', label='gaussian_x')
            ax[0].set_ylabel("P(a_gap_x)")
            ax[0].set_xlabel("NLL Loss: %1.4f" % a_gap_nll_loss[0].item())
            ax[0].legend()

            count, bins, _ = ax[1].hist(a_gap[:, 1], bins=n_bins, density=True, histtype='step', color='k', label='a_gap y')
            ax[1].plot(bins, gaussian(bins, a_gap_mean[1], a_gap_std[1]), color='r', label='gaussian_y')
            ax[1].set_ylabel("P(a_gap_y)")
            ax[1].set_xlabel("NLL Loss: %1.4f" % a_gap_nll_loss[1].item())
            ax[1].legend()

            count, bins, _ = ax[2].hist(a_gap[:, 2], bins=n_bins, density=True, histtype='step', color='k', label='a_gap z')
            ax[2].plot(bins, gaussian(bins, a_gap_mean[2], a_gap_std[2]), color='r', label='gaussian_z')
            ax[2].set_ylabel("P(a_gap_z)")
            ax[2].set_xlabel("NLL Loss: %1.4f" % a_gap_nll_loss[2].item())
            ax[2].legend()

            pth = os.path.join(self.path_dir_result, 'accel_gap_distribution.png')
            plt.tight_layout()
            fig.savefig(pth)
            plt.close(fig)

            ### Visualize Angular Velocity Gap
            fig, ax = plt.subplots(3, 1)
            fig.suptitle('%s / Angular Gap histogram' % self.seq)

            N = w_gap.shape[0]
            n_bins = 500

            count, bins, _ = ax[0].hist(w_gap[:, 0], bins=n_bins, density=True, histtype='step', color='k', label='w_gap x')
            ax[0].plot(bins, gaussian(bins, w_gap_mean[0], w_gap_std[0]), color='r', label='gaussian_x')
            ax[0].set_ylabel("P(w_gap_x)")
            ax[0].set_xlabel("NLL Loss: %1.4f" % w_gap_nll_loss[0].item())
            ax[0].legend()

            count, bins, _ = ax[1].hist(w_gap[:, 1], bins=n_bins, density=True, histtype='step', color='k', label='w_gap y')
            ax[1].plot(bins, gaussian(bins, w_gap_mean[1], w_gap_std[1]), color='r', label='gaussian_y')
            ax[1].set_ylabel("P(w_gap_y)")
            ax[1].set_xlabel("NLL Loss: %1.4f" % w_gap_nll_loss[1].item())
            ax[1].legend()

            count, bins, _ = ax[2].hist(w_gap[:, 2], bins=n_bins, density=True, histtype='step', color='k', label='w_gap z')
            ax[2].plot(bins, gaussian(bins, w_gap_mean[2], w_gap_std[2]), color='r', label='gaussian_z')
            ax[2].set_ylabel("P(w_gap_z)")
            ax[2].set_xlabel("NLL Loss: %1.4f" % w_gap_nll_loss[2].item())
            ax[2].legend()

            pth = os.path.join(self.path_dir_result, 'angular_gap_distribution.png')
            plt.tight_layout()
            fig.savefig(pth)
            plt.close(fig)

        return {
            'a_mean': torch.from_numpy(a_gap_mean),
            'a_std':  torch.from_numpy(a_gap_std),
            'w_mean': torch.from_numpy(w_gap_mean),
            'w_std':  torch.from_numpy(w_gap_std),
        }


    def plot_accel(self, start=None, end=None, avg_window=1):

        v_gt = torch.tensor(self.gt_int[:, 8:11], device=self.device)
        a_gt = (v_gt[1:] - v_gt[:-1]) / 0.005
        a_gt = torch.cat([a_gt[0].unsqueeze(0), (a_gt[1:] + a_gt[:-1]) / 2.0, a_gt[-1].unsqueeze(0)])
        ts = torch.tensor(self.gt_int[:, 0], device=self.device)


        a_raw = torch.tensor(self.imu[:, 4:7], device=self.device)
        q_gt = torch.tensor(self.gt_int[:, 4:8], device=self.device)
        q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
        rot_gt = SO3.from_quaternion(q_gt, ordering='wxyz')
        a_raw = torch.einsum('bij, bj -> bi', rot_gt, a_raw)

        if avg_window > 1:
            assert avg_window % 2 == 1 # recommend odd num for window_size
            pad = avg_window // 2

            def avg(arr):
                N = arr.shape[0]
                M = arr.shape[1]
                pad0 = arr[0, :].unsqueeze(0).expand(pad, M)
                padN = arr[-1, :].unsqueeze(0).expand(pad, M)
                tmp_arr = torch.cat([pad0, arr, padN], dim=0)
                avg_arr = torch.zeros_like(arr)
                for i in range(avg_window):
                    avg_arr += tmp_arr[i: i+N]
                avg_arr /= avg_window
                return avg_arr

            a_raw = avg(a_raw)
            a_gt = avg(a_gt)


        a_raw = a_raw.cpu()
        v_gt = v_gt.cpu()
        a_gt = a_gt.cpu()
        ts = ts.cpu()

        if start is not None and end is not None:
            i0 = np.searchsorted(ts, start)
            iN = np.searchsorted(ts, end, side='right')
            print('Clip %d to %d' % (i0, iN))

            a_raw = a_raw[i0:iN]
            v_gt = v_gt[i0:iN]
            a_gt = a_gt[i0:iN]
            ts = ts[i0:iN]

        ## Visualize

        if avg_window == 1:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(40, 16), dpi=200)
            fig.suptitle('Acceleration / %s' % self.seq, fontsize=20)

            axs[0][0].plot(ts, a_raw[:, 0], 'k', linewidth=1, label="Raw. acc(x)")
            axs[0][0].plot(ts, a_gt[:, 0],  'r', linewidth=1, label="GT.  acc(x)")
            axs[0][0].set_title('Accel - X')
            axs[0][0].legend(loc='best')
            axs[0][1].plot(ts, a_raw[:, 1], 'k', linewidth=1, label="Raw. acc(y)")
            axs[0][1].plot(ts, a_gt[:, 1],  'r', linewidth=1, label="GT.  acc(y)")
            axs[0][1].set_title('Accel - Y')
            axs[0][1].legend(loc='best')
            axs[1][0].plot(ts, a_raw[:, 2], 'k', linewidth=1, label="Raw. acc(z)")
            axs[1][0].plot(ts, a_gt[:, 2],  'r', linewidth=1, label="GT.  acc(z)")
            axs[1][0].set_title('Accel - Z')
            axs[1][0].legend(loc='best')

            axs[1][1].plot(ts, a_raw[:, 2] - a_gt[:, 2], 'k', linewidth=1, label="Gravity")
            axs[1][1].set_title('Gravity')
            axs[1][1].legend(loc='best')

            pth = os.path.join(self.path_dir_result, 'accel.png')
            fig.savefig(pth)
            plt.tight_layout()
            plt.close(fig)
        elif avg_window > 1:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(40, 16), dpi=200)
            fig.suptitle('Average accel / %s' % self.seq, fontsize=20)

            fig = plt.figure(constrained_layout=True, figsize=(32, 16), dpi=200)
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(2, 2, figure=fig)

            axs = [
                [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                fig.add_subplot(gs[1, :])
            ]

            axs[0][0].plot(ts, a_raw[:, 0], 'k', linewidth=1, label="Raw. acc(x)")
            axs[0][0].plot(ts, a_gt[:, 0],  'r', linewidth=1, label="GT.  acc(x)")
            axs[0][0].set_title('Accel - X')
            axs[0][0].legend(loc='best')

            axs[0][1].plot(ts, a_raw[:, 1], 'k', linewidth=1, label="Raw. acc(y)")
            axs[0][1].plot(ts, a_gt[:, 1],  'r', linewidth=1, label="GT.  acc(y)")
            axs[0][1].set_title('Accel - Y')
            axs[0][1].legend(loc='best')

            # axs[1][0].plot(ts, a_raw[:, 2], 'k', linewidth=1, label="Raw. acc(z)")
            # axs[1][0].plot(ts, a_gt[:, 2],  'r', linewidth=1, label="GT.  acc(z)")
            # axs[1][0].set_title('Accel - Z')
            # axs[1][0].legend(loc='best')

            axs[1].plot(ts, a_raw[:, 2] - a_gt[:, 2], 'k', linewidth=1, label="Gravity")
            axs[1].set_title('Gravity')
            axs[1].legend(loc='best')


            pth = os.path.join(self.path_dir_result, 'avg_accel_W%d.png' % avg_window)
            fig.savefig(pth)
            plt.tight_layout()
            plt.close(fig)

    def plot_accel_gap(self, start=None, end=None, avg_window=1):

        q_gt = torch.tensor(self.gt_int[:, 4:8], device=self.device)
        q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
        rot_gt = SO3.from_quaternion(q_gt, ordering='wxyz')

        v_gt = torch.tensor(self.gt_int[:, 8:11], device=self.device)
        a_gt = (v_gt[1:] - v_gt[:-1]) / 0.005
        a_gt = torch.cat([a_gt[0].unsqueeze(0), (a_gt[1:] + a_gt[:-1]) / 2.0, a_gt[-1].unsqueeze(0)])

        ts = torch.tensor(self.gt_int[:, 0], device=self.device)

        a_raw = torch.tensor(self.imu[:, 4:7], device=self.device)
        a_raw = torch.einsum('bij, bj -> bi', rot_gt, a_raw)

        if start is not None and end is not None:
            i0 = np.searchsorted(ts, start)
            iN = np.searchsorted(ts, end, side='right')
            print('Clip %d to %d' % (i0, iN))

            a_raw = a_raw[i0:iN]
            v_gt = v_gt[i0:iN]
            a_gt = a_gt[i0:iN]
            ts = ts[i0:iN]

        assert avg_window % 2 == 1 # recommend odd num for window_size
        pad = avg_window // 2
        def avg(arr):
            N = arr.shape[0]
            M = arr.shape[1]
            pad0 = arr[0, :].unsqueeze(0).expand(pad, M)
            padN = arr[-1, :].unsqueeze(0).expand(pad, M)
            tmp_arr = torch.cat([pad0, arr, padN], dim=0)
            avg_arr = torch.zeros_like(arr)
            for i in range(avg_window):
                avg_arr += tmp_arr[i: i+N]
            avg_arr /= avg_window
            return avg_arr
        smoothed_a_raw = avg(a_raw)
        smoothed_a_gt = avg(a_gt)

        acc_gap_raw = a_raw - smoothed_a_raw
        acc_gap_raw_std = (acc_gap_raw**2).mean(dim=0).sqrt()
        # print("acc_gap_raw_std:", acc_gap_raw_std.shape, acc_gap_raw_std.device)

        acc_gap_gt = a_gt - smoothed_a_gt
        acc_gap_gt_std = (acc_gap_gt**2).mean(dim=0).sqrt()
        # print("acc_gap_gt_std:", acc_gap_gt_std.shape, acc_gap_gt_std.device)

        a_raw = a_raw.cpu()
        v_gt = v_gt.cpu()
        a_gt = a_gt.cpu()
        acc_gap_raw = acc_gap_raw.cpu()
        acc_gap_raw_std = acc_gap_raw_std.cpu()
        acc_gap_gt = acc_gap_gt.cpu()
        acc_gap_gt_std = acc_gap_gt_std.cpu()
        ts = ts.cpu()

        ## Visualize accel
        if False:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(40, 16), dpi=200)
            fig.suptitle('Acceleration / %s' % self.seq, fontsize=20)

            axs[0][0].plot(ts, a_raw[:, 0], 'k', linewidth=1, label="Raw. acc(x)")
            axs[0][0].plot(ts, a_gt[:, 0],  'r', linewidth=1, label="GT.  acc(x)")
            axs[0][0].set_title('Accel - X')
            axs[0][0].legend(loc='best')
            axs[0][1].plot(ts, a_raw[:, 1], 'k', linewidth=1, label="Raw. acc(y)")
            axs[0][1].plot(ts, a_gt[:, 1],  'r', linewidth=1, label="GT.  acc(y)")
            axs[0][1].set_title('Accel - Y')
            axs[0][1].legend(loc='best')
            axs[1][0].plot(ts, a_raw[:, 2], 'k', linewidth=1, label="Raw. acc(z)")
            axs[1][0].plot(ts, a_gt[:, 2],  'r', linewidth=1, label="GT.  acc(z)")
            axs[1][0].set_title('Accel - Z')
            axs[1][0].legend(loc='best')

            axs[1][1].plot(ts, a_raw[:, 2] - a_gt[:, 2], 'k', linewidth=1, label="Gravity")
            axs[1][1].set_title('Gravity')
            axs[1][1].legend(loc='best')

            pth = os.path.join(self.path_dir_result, 'accel.png')
            fig.savefig(pth)
            plt.tight_layout()
            plt.close(fig)

        ## Visualize avg accel
        if False:
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(40, 16), dpi=200)
            fig.suptitle('Average accel / %s' % self.seq, fontsize=20)

            fig = plt.figure(constrained_layout=True, figsize=(32, 16), dpi=200)
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(2, 2, figure=fig)

            axs = [
                [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                fig.add_subplot(gs[1, :])
            ]

            axs[0][0].plot(ts, a_raw[:, 0], 'k', linewidth=1, label="Raw. acc(x)")
            axs[0][0].plot(ts, a_gt[:, 0],  'r', linewidth=1, label="GT.  acc(x)")
            axs[0][0].set_title('Accel - X')
            axs[0][0].legend(loc='best')

            axs[0][1].plot(ts, a_raw[:, 1], 'k', linewidth=1, label="Raw. acc(y)")
            axs[0][1].plot(ts, a_gt[:, 1],  'r', linewidth=1, label="GT.  acc(y)")
            axs[0][1].set_title('Accel - Y')
            axs[0][1].legend(loc='best')

            # axs[1][0].plot(ts, a_raw[:, 2], 'k', linewidth=1, label="Raw. acc(z)")
            # axs[1][0].plot(ts, a_gt[:, 2],  'r', linewidth=1, label="GT.  acc(z)")
            # axs[1][0].set_title('Accel - Z')
            # axs[1][0].legend(loc='best')

            axs[1].plot(ts, a_raw[:, 2] - a_gt[:, 2], 'k', linewidth=1, label="Gravity")
            axs[1].set_title('Gravity')
            axs[1].legend(loc='best')

            pth = os.path.join(self.path_dir_result, 'avg_accel_W%d.png' % avg_window)
            fig.savefig(pth)
            plt.tight_layout()
            plt.close(fig)

        ## Visualize acc gap
        tmp[self.seq] = [self.seq]

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(40, 20), dpi=200)
        fig.suptitle('Average gap / %s' % self.seq, fontsize=20)
        axs[0][0].plot(ts, acc_gap_raw[:, 0], 'k', linewidth=1, label="Raw. acc_gap(x)")
        axs[0][0].set_title('Accel gap of x == %1.4f' % acc_gap_raw_std[0].item())
        axs[0][0].legend(loc='best')
        axs[0][1].plot(ts, acc_gap_gt[:, 0], 'r', linewidth=1, label="GT. acc_gap(x)")
        axs[0][1].set_title('Accel gt gap of x == %1.4f' % acc_gap_gt_std[0].item())
        axs[0][1].legend(loc='best')
        tmp[self.seq].append(acc_gap_raw_std[0].item())
        tmp[self.seq].append( acc_gap_gt_std[0].item())

        axs[1][0].plot(ts, acc_gap_raw[:, 1], 'k', linewidth=1, label="Raw. acc_gap(y)")
        axs[1][0].set_title('Accel gap of y == %1.4f' % acc_gap_raw_std[1].item())
        axs[1][0].legend(loc='best')
        axs[1][1].plot(ts, acc_gap_gt[:, 1], 'r', linewidth=1, label="GT. acc_gap(y)")
        axs[1][1].set_title('Accel gt gap of y == %1.4f' % acc_gap_gt_std[1].item())
        axs[1][1].legend(loc='best')
        tmp[self.seq].append(acc_gap_raw_std[1].item())
        tmp[self.seq].append( acc_gap_gt_std[1].item())

        axs[2][0].plot(ts, acc_gap_raw[:, 2], 'k', linewidth=1, label="Raw. acc_gap(z)")
        axs[2][0].set_title('Accel gap of z == %1.4f' % acc_gap_raw_std[2].item())
        axs[2][0].legend(loc='best')
        axs[2][1].plot(ts, acc_gap_gt[:, 2], 'r', linewidth=1, label="GT. acc_gap(z)")
        axs[2][1].set_title('Accel gt gap of z == %1.4f' % acc_gap_gt_std[2].item())
        axs[2][1].legend(loc='best')
        tmp[self.seq].append(acc_gap_raw_std[2].item())
        tmp[self.seq].append( acc_gap_gt_std[2].item())

        l = tmp[self.seq]
        print(l)

        pth = os.path.join(self.path_dir_result, 'accel_gap.png')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        fig.savefig(pth)
        plt.close(fig)

    def plot_img_imu(self, img_file, window_size):
        img = Image.open(img_file)
        img = np.array(img)

        fig, axs = plt.subplots()

# MH_01_easy 데이터셋에서 드론이 멈춰있는 시간과, 그 시간 안에서 살짝 진동하는 시간을 제외하고,
# World frame에서의 a^{IMU} 값이 어떻게 진동하고, 얼마나 Bias가 생기는지 알아본다.
# 1403636601713555456 : stop start
# 1403636611163555584 : vibe start
# 1403636613613555456 : vibe end
# 1403636623613555456 : stop end

euroc_seqs = [
    'MH_01_easy',
    'MH_02_easy',
    'MH_03_medium',
    'MH_04_difficult',
    'MH_05_difficult',
    'V1_01_easy',
    'V1_02_medium',
    'V1_03_difficult',
    'V2_01_easy',
    'V2_02_medium',
    'V2_03_difficult'
]

if (False):
    mh_01_easy = Data('MH_01_easy')
    # mh_01_easy.plot_accel(1403636601713555456, 1403636623613555456, avg_window=25)
    mh_01_easy.plot_accel(1403636601713555456, 1403636623613555456, avg_window=53)
    mh_01_easy.plot_accel(1403636601713555456, 1403636623613555456, avg_window=53)

    # mh_01_easy = Data('MH_02_easy')
    # mh_01_easy.plot_accel()

    # mh_01_easy = Data('MH_03_medium')
    # mh_01_easy.plot_accel()

def main():

    dist_dict = {}
    for seq in euroc_seqs:
        dataset = Data(seq)
        # dataset.plot_accel(avg_window=1)
        # dataset.plot_accel_gap(avg_window=51)
        _dict = dataset.get_gap_distribution(plot=True)
        dist_dict[seq] = _dict
    fname = os.path.join('/home/leecw/project/results/Figures', 'gap_dist.pt')
    torch.save(dist_dict, fname)

if __name__ == '__main__':
    main()
