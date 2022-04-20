from cgi import parse_multipart
import sys
sys.path.append('/root/denoise')

import torch
import time
import matplotlib.pyplot as plt
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.labelsize'] = 'x-large'
plt.rcParams['legend.fontsize'] = 'x-large'
plt.rcParams['xtick.labelsize'] = 'x-large'
plt.rcParams['ytick.labelsize'] = 'x-large'
from termcolor import cprint
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from src.utils import pload, pdump, yload, ydump, mkdir, bmv
from src.utils import bmtm, bmtv, bmmt, bbmv


from src.lie_algebra import SO3
from src.DGADataset import DGADataset

class LearningProcess:
    """
        Manage all training and test process.
    """

    def __init__(self, params, mode): # , net_class, net_params, address, dt
        """
            - Make sure that a model's in gpu after initialization.
        """
        self.params = params
        self.weight_path = os.path.join(self.params['result_dir'], 'weights.pt')
        self.dict_test_result = {}
        self.figsize = (20, 12)
        self.dt = 0.005 # (s)

        self.preprocess()

        if mode == 'train':
            if not os.path.exists(self.params['result_dir']):
                os.makedirs(self.params['result_dir'])
            ydump(self.params, params['result_dir'], 'params.yaml')
            self.net = params['net_class'](params)
        elif mode == 'test':
            self.params = yload(params['result_dir'], 'params.yaml')
            self.net = params['net_class'](params)
            weights = torch.load(self.weight_path)
            self.net.load_state_dict(weights)
        else:
            self.params = yload(params['result_dir'], 'params.yaml')
            cprint('  No need to initialize a model', 'yellow')
            return

        self.net = self.net.cuda()

    def preprocess(self):
        print('\n# Preprocess ... ')
        all_seqs = [*self.params['dataset']['train_seqs'], *self.params['dataset']['test_seqs']]
        all_seqs.sort()
        dv_windows = [16, 32, 64] #
        dv_normed_windows = [32, 64, 128, 256, 512] #

        for seq in all_seqs:
            _seq_dir = os.path.join(self.params['dataset']['predata_dir'], seq)
            if os.path.exists(_seq_dir):
                print('  %s is already pre-processed.' % seq)
                continue
            else:
                print('  %s is being pre-processed' % seq)
                os.makedirs(_seq_dir)

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

            # compute pre-integration factors for all training
            mtf = self.params['dataset']['min_train_freq']
            dRot_ij = bmtm(Rot_gt[:-mtf], Rot_gt[mtf:])
            dRot_ij = SO3.dnormalize(dRot_ij.cuda())
            dxi_ij = SO3.log(dRot_ij).cpu()
            # print("\tdxi_ij -- ", dxi_ij.shape, dxi_ij.dtype) # [29976, 3]

            ## COMPUTE gt_dv, gt_dv_normed
            v_gt = v_gt.cuda() # torch.Size([N, 3]) torch.float64 cuda:0
            _gt_dv = {}
            for window_size in dv_windows:
                _gt_dv[str(window_size)] = v_gt[window_size:] - v_gt[:-window_size]

            _gt_dv_normed = {}
            for window_size in dv_normed_windows:
                N = v_gt.shape[0]

                bread = torch.ones(window_size, N+window_size-1, 3, dtype=v_gt.dtype).cuda()
                bread *= v_gt[0, :].expand_as(bread)
                for i in range(window_size):
                    bread[i, window_size-1-i:N+window_size-1-i] = v_gt
                bread = bread[:, 0:N]

                ## Debug
                # for i in range(window_size):
                #     li = bread[:, i, 0].tolist()
                #     for e in li:
                #         print('%.6f' % e, end=',')
                #     print()
                ##

                _mean = bread.mean(dim=0)
                _normalized = v_gt - _mean

                ## Debug
                # print('v_gt')
                # print(v_gt[0:8, 0].tolist())
                # print('_normalized:', _normalized.shape, _normalized.dtype)
                # print(_normalized[0:8, 0].tolist())
                ##

                _gt_dv_normed[str(window_size)] = _normalized


            ## SAVE
            _data_path = os.path.join(_seq_dir, 'data.pt')
            _data_dict = {
                'ts': imu[:, 0].float(),
                'us': imu[:, 1:].float() }
            torch.save(_data_dict, _data_path)

            _gt_path = os.path.join(_seq_dir, 'gt.pt')
            _gt_dict = {
                'gt_interpolated': gt_interpolated.float(),
                'dw_16': dxi_ij.float()} # the 16-size window's euler angle difference
            torch.save(_gt_dict, _gt_path)

            _gt_dv_path = os.path.join(_seq_dir, 'dv.pt')
            _gt_dv_dict = {
                'dv': {
                    '16': _gt_dv['16'].float(),
                    '32': _gt_dv['32'].float(),
                    '64': _gt_dv['64'].float(),
                },
                'dv_normed': {
                    '32':   _gt_dv_normed['32'].float(),
                    '64':   _gt_dv_normed['64'].float(),
                    '128':  _gt_dv_normed['128'].float(),
                    '256':  _gt_dv_normed['256'].float(),
                    '512':  _gt_dv_normed['512'].float(),
                }
            }
            torch.save(_gt_dv_dict, _gt_dv_path)

            ## CHECK
            if self.params['debug']:
                def print_dict(d:dict):
                    for k, v in d.items():
                        if type(v) is dict:
                            print('  %s:' % k)
                            print_dict(v)
                        else:
                            print('    %s:' % k, type(v), v.shape, v.dtype)
                print_dict(_data_dict)
                print_dict(_gt_dict)
                print_dict(_gt_dv_dict)

        print("--- success ---")
        """
        _mean: torch.Size([36381, 3]) torch.float64
        _normalized: torch.Size([36381, 3]) torch.float64
            ts: <class 'torch.Tensor'> torch.Size([36381]) torch.float32
            us: <class 'torch.Tensor'> torch.Size([36381, 6]) torch.float32
            dw_16: <class 'torch.Tensor'> torch.Size([36365, 3]) torch.float32
            gt_interpolated: <class 'torch.Tensor'> torch.Size([36381, 17]) torch.float32
        dv:
            16: <class 'torch.Tensor'> torch.Size([36365, 3]) torch.float32
            32: <class 'torch.Tensor'> torch.Size([36349, 3]) torch.float32
            64: <class 'torch.Tensor'> torch.Size([36317, 3]) torch.float32
        dv_normed:
            32: <class 'torch.Tensor'> torch.Size([36381, 3]) torch.float32
            64: <class 'torch.Tensor'> torch.Size([36381, 3]) torch.float32
            512: <class 'torch.Tensor'> torch.Size([36381, 3]) torch.float32
            1024: <class 'torch.Tensor'> torch.Size([36381, 3]) torch.float32
            128: <class 'torch.Tensor'> torch.Size([36381, 3]) torch.float32
            256: <class 'torch.Tensor'> torch.Size([36381, 3]) torch.float32
        """

    def train(self):
        """train the neural network. GPU is assumed"""
        ydump(self.params, self.params['result_dir'], 'params.yaml')

        # define datasets
        dataset_train   = DGADataset(self.params, 'train')
        dataset_val     = DGADataset(self.params, 'val')

        # Define class
        Optimizer = self.params['train']['optimizer_class']
        Scheduler = self.params['train']['scheduler_class']
        Loss = self.params['train']['loss_class']

        # Instantiate optimizer, scheduler and loss.
        optimizer = Optimizer(self.net.parameters(), **self.params['train']['optimizer'])
        scheduler = Scheduler(optimizer, **self.params['train']['scheduler'])
        dataloader = DataLoader(dataset_train, **self.params['train']['dataloader'])
        criterion = Loss(self.params)

        # remaining training parameters
        freq_val = self.params['train']['freq_val']
        n_epochs = self.params['train']['n_epochs']

        # init net w.r.t dataset
        self.net.set_normalized_factors(torch.Tensor(dataset_train.mean_u), torch.Tensor(dataset_train.std_u))

        # start tensorboard writer
        writer = SummaryWriter(self.params['result_dir'])
        start_time = time.time()
        best_loss = torch.Tensor([float('Inf')])

        # define some function for seeing evolution of training
        # def write(epoch, loss_epoch):
        #     scheduler.step(epoch)

        # Training Loop
        loss, best_loss = torch.Tensor([10000.0]), torch.Tensor([10000.0])
        for epoch in range(1, n_epochs + 1):
            print('\n# Epoch %d' % epoch)
            loss_epoch = self.loop_train(dataloader, optimizer, criterion)
            writer.add_scalar('loss/train', loss_epoch.item(), epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            scheduler.step(epoch)

            # Validate
            if epoch % freq_val == 0:
                loss = self.loop_val(dataset_val, criterion)
                dt = time.time() - start_time

                if loss <= best_loss:
                    cprint('Epoch %4d Loss(val) Decrease - %.2fs' % (epoch, dt), 'blue')
                    print('  - current: %.4f' % loss.item())
                    print('  - best   : %.4f' % best_loss.item())
                    best_loss = loss
                    self.save_net(epoch, 'best')
                else:
                    cprint('Epoch %4d Loss(val) Increase - %.2fs' % (epoch, dt), 'yellow')
                    print('  - current: %.4f' % loss.item())
                    print('  - best   : %.4f' % best_loss.item())
                    self.save_net(epoch, 'log')

                writer.add_scalar('loss/val', loss.item(), epoch)
                start_time = time.time()
            elif epoch % (freq_val//5) == 0:
                print('Epoch %4d Loss(train) %.4f' % (epoch, loss_epoch))


        cprint('\n  Train is over  \n', 'cyan', attrs=['bold'])

        cprint('Testing ... ', 'green')
        dataset_test = DGADataset(self.params, 'test')
        weights = torch.load(self.weight_path)
        self.net.load_state_dict(weights)
        self.net.cuda()
        test_loss = self.loop_val(dataset_test, criterion)
        dict_loss = {
            'final_loss/val': best_loss.item(),
            'final_loss/test': test_loss.item()
        }
        for key, value in dict_loss.items():
            print('  %s: ' % key, value)
        ydump(dict_loss, self.params['result_dir'], 'final_loss.yaml')

        writer.close()

    def test(self):
        """test a network once training is over"""
        Loss = self.params['train']['loss_class']
        criterion = Loss(self.params)
        dataset_test = DGADataset(self.params, 'test')

        if not os.path.exists(self.params['test_dir']):
            os.makedirs(self.params['test_dir'])

        cprint('Test ... ', 'green')
        self.loop_test(dataset_test, criterion)
        print('  --success')

    def analyze(self):
        dataset_test = DGADataset(self.params, 'test')

        for i, seq in enumerate(dataset_test.sequences):
            self.seq = seq
            if not os.path.exists(os.path.join(self.params['result_dir'], seq)):
                os.mkdir(os.path.join(self.params['result_dir'], seq))
            cprint('On %s ... ' % seq, 'green', end='')

            ## LOAD DATA
            seq, us, gt, dw_16, dv_normed = dataset_test[i]
            us = us.cpu()
            gt = gt.cpu()
            dw_16 = dw_16.cpu()
            for w in dv_normed:
                dv_normed[w] = dv_normed[w].cpu()

            pos_gt = gt[:, 1:4]
            quat_gt = gt[:, 4:8]
            vel_gt = gt[:, 8:11]

            mondict = pload(self.params['result_dir'], 'tests', 'results_%s.p'%seq)
            w_hat = mondict['w_hat']
            a_hat = mondict['a_hat']
            loss = mondict['loss'] # float
            self.ts = torch.linspace(0, N * self.dt, N)

            ## Analyze Orientation
            N = us.shape[0]
            rot_gt = SO3.from_quaternion(quat_gt.cuda()).cpu()
            rpy_gt = SO3.to_rpy(rot_gt.cuda()).cpu()

            def rad2deg(x):
                return x * (180. / np.pi)

            quat_hat, rot_imu, rot_hat = self.integrate_with_quaternions_superfast(dw_16.shape[0], us, w_hat, quat_gt)
            rpy_imu = SO3.to_rpy(rot_imu).cpu()
            rpy_hat = SO3.to_rpy(rot_hat).cpu()
            self.plot_orientation(N, rad2deg(rpy_imu), rad2deg(rpy_hat), rad2deg(rpy_gt))
            self.plot_orientation_error(N, rot_imu, rot_hat, rot_gt)

            gyro_corrections  =  (us[:, :3]  - w_hat[:N, :])
            self.plot_gyro_correction(gyro_corrections)

            print('- rot_gt:', rot_gt.shape)
            rot_gt = rot_gt.reshape(us.shape[0], 3, 3)
            a_w = bmv(rot_gt, us[:, 3:6]) - torch.Tensor([0., 0., 9.81])
            accel_corrections =  (a_w - a_hat)
            self.plot_accel_correction(accel_corrections)

            self.plot_accel(a_hat, gt)

            cprint('[ok]\n', 'blue')

    def loop_train(self, dataloader, optimizer, criterion):
        """Forward-backward loop over training data"""
        loss_epoch = 0
        optimizer.zero_grad()
        for seq, us, gt, dw_16, dv_normed in dataloader:
            us = dataloader.dataset.add_noise(us)
            # print('  us:', us.shape, us.dtype, us.device)
            q_gt = gt[:, :, 4:8].reshape(-1, 4)
            rot_gt = SO3.from_quaternion(q_gt.cuda())
            rot_gt = rot_gt.reshape(us.shape[0], us.shape[1], 3, 3)

            w_hat, a_hat = self.net(us, rot_gt)
            loss = criterion(w_hat, dw_16, a_hat, dv_normed)/len(dataloader)
            loss.backward()

            # for name, param in self.net.named_parameters():
            #     if param.grad is not None:
            #         print(name, torch.isfinite(param.grad).all())

            loss_epoch += loss.detach().cpu()
            # print('train_loss:', loss.detach().cpu().item())
        optimizer.step()
        return loss_epoch

    def loop_val(self, dataset, criterion):
        """Forward loop over validation data"""
        loss_epoch = 0
        self.net.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                seq, us, gt, dw_16, dv_normed = dataset[i]
                q_gt = gt[:, 4:8]
                rot_gt = SO3.from_quaternion(q_gt.cuda())
                rot_gt = rot_gt.reshape(us.shape[0], 3, 3)

                w_hat, a_hat = self.net(us.unsqueeze(0), rot_gt.unsqueeze(0))
                for key in dv_normed:
                    dv_normed[key] = dv_normed[key].unsqueeze(0)
                loss = criterion(w_hat, dw_16.unsqueeze(0), a_hat, dv_normed)/len(dataset)
                loss_epoch += loss.cpu()
        self.net.train()
        return loss_epoch

    def loop_test(self, dataset, criterion):
        """Forward loop over test data"""
        self.net.eval()

        for i in range(len(dataset)):
            seq, us, gt, dw_16, dv_normed = dataset[i]
            q_gt = gt[:, 4:8]
            rot_gt = SO3.from_quaternion(q_gt.cuda())
            rot_gt = rot_gt.reshape(us.shape[0], 3, 3)
            print('  - %s  ' % seq, end='')
            # print('  us:', us.shape)
            # print('  xs:', xs.shape)
            # print('  dv:', dv.shape)

            with torch.no_grad():
                w_hat, a_hat = self.net(us.unsqueeze(0), rot_gt.unsqueeze(0))
                for key in dv_normed:
                    dv_normed[key] = dv_normed[key].unsqueeze(0)
                loss = criterion(w_hat, dw_16.unsqueeze(0), a_hat, dv_normed)/len(dataset)

                self.dict_test_result[seq] = {
                    'w_hat': w_hat[0].cpu(),
                    'a_hat': a_hat[0].cpu(),
                    'loss': loss.cpu().item(),
                }
                for key, value in self.dict_test_result[seq].items():
                    if key == 'loss':
                        continue
                    print('    %s:'%key, type(value), value.shape, value.dtype)

                path_results = os.path.join(self.params['test_dir'], 'results_%s.p'%seq)
                if not os.path.exists(path_results):
                    pdump(self.dict_test_result[seq], path_results)
            print('[ok]')

    def save_net(self, epoch=None, state='log'):
        """save the weights on the net in CPU"""
        self.net.eval().cpu()
        if state == 'log':
            save_path = os.path.join(self.params['result_dir'], 'ep_%04d.pt' % epoch)
            torch.save(self.net.state_dict(), save_path)
        elif state == 'best':
            save_path = os.path.join(self.params['result_dir'], 'ep_%04d_best.pt' % epoch)
            torch.save(self.net.state_dict(), save_path)
            torch.save(self.net.state_dict(), self.weight_path)
        self.net.train().cuda()


    def save_gyro_estimate(self, seq):
        net_us = pload(self.params['result_dir'], seq, 'results.p')['hat_xs']
        N = net_us.shape[0]
        path = os.path.join("/home/leecw/Data/Result/DenoiseIMU/estimate", seq, seq + '_net_us.csv')
        header = "time(s),wx,wy,wz"
        x = np.zeros(N, 4)
        x[:, 0]

    def to_open_vins(self, dataset):
        """
        Export results to Open-VINS format. Use them eval toolbox available
        at https://github.com/rpng/open_vins/
        """
        print("open_vins()")

        for i, seq in enumerate(dataset.sequences):
            self.seq = seq
            # get ground truth
            self.gt = dataset.load_gt(i)
            raw_us, _ = dataset[i]
            net_us = pload(self.params['result_dir'], seq, 'results.p')['hat_xs']
            N = net_us.shape[0]

            net_qs, imu_Rots, net_Rots = self.integrate_with_quaternions_superfast(N, raw_us, net_us)

            path = os.path.join(self.params['result_dir'], seq + '.csv')
            header = "time(s),tx,ty,tz,qx,qy,qz,qw"
            x = np.zeros((net_qs.shape[0], 8))
            x[:, 0] = self.gt['ts'][:net_qs.shape[0]]

            x[:, [7, 4, 5, 6]] = net_qs
            np.savetxt(path, x[::1], header=header, delimiter=",", fmt='%1.9f')

            ### Save wx, wy, wz csv file
            # if seq in ['MH_02_easy', 'MH_04_difficult']:
            #     print("\t", seq)
            #     N = net_qs.shape[0]
            #     print("\t\tnet_qs: ", net_qs.shape)
            #     print("\t\tnet_us: ", net_us.shape)
            #     print("\t\timu_Rots: ", imu_Rots.shape)
            #     print("\t\tnet_Rots: ", net_Rots.shape)
            #     print("\t\tself.gt['ts']:", self.gt['ts'].shape)

            #     gt_processed_path = os.path.join("/root/Data/Result/DenoiseIMU/gt", seq, "gt_processed.csv")
            #     gt_processed = np.loadtxt(gt_processed_path, delimiter=',')
            #     t_ns = gt_processed[:, 0]
            #     print("\t\tt_ns:", t_ns.shape)      # (29992,) float64
            #     print("\t\traw_us:", raw_us.shape)  # [29952, 6]

            #     imu_processed_path = os.path.join("/root/Data/Result/DenoiseIMU/estimate", seq, "imu_processed.csv")
            #     imu_processed = np.loadtxt(imu_processed_path, delimiter=',')


            #     denoised_path = os.path.join("/root/Data/Result/DenoiseIMU/estimate", seq, "denoised.csv")
            #     header = "timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
            #     denoised = np.zeros((N, 7))
            #     denoised[:, 0] = imu_processed[:N, 0]
            #     denoised[:, 1:4] = net_us
            #     denoised[:, 4:7] = imu_processed[:N, 4:7]
            #     np.savetxt(denoised_path, denoised, header=header, fmt='%d,%1.9f,%1.9f,%1.9f,%1.9f,%1.9f,%1.9f')
            #     print("\t\tdenoised is saved in \'%s\'" % denoised_path)
            ###

            ### DENOISED IMU DATA
            if seq in ['MH_02_easy', 'MH_04_difficult']:
                raw_interpolated_imu_path = os.path.join("/root/Data/Result/DenoiseIMU", seq + '_raw_imu_interpolated.csv')
                raw_interpolated_imu = np.loadtxt(raw_interpolated_imu_path, dtype=np.float64, delimiter=',')
                denoised_interpolated_imu = raw_interpolated_imu[:net_us.shape[0]]
                denoised_interpolated_imu_path = os.path.join(self.params['result_dir'], seq, 'denoised_imu.csv')
                denoised_interpolated_imu[:,1:4] = np.squeeze(net_us)
                header = "time[ns],wx,wy,wz,ax,ay,az"
                np.savetxt(denoised_interpolated_imu_path, denoised_interpolated_imu, fmt="%d,%1.9f,%1.9f,%1.9f,%1.9f,%1.9f,%1.9f", header=header)
                print("denoied imu data is saved in \'%s\'" % denoised_interpolated_imu_path)


            # Save net_us on csv file
            # net_us_csv = np.zeros((net_us))
            # np.savetxt("/root/Data/Result/DenoiseIMU/estimate/MH_02_easy/net_us.csv", )

            imu_rpys_path = os.path.join(self.params['result_dir'], seq, 'raw_rpy.csv')
            net_rpys_path = os.path.join(self.params['result_dir'], seq, 'net_rpy.csv')
            imu_rpys = SO3.to_rpy(imu_Rots).cpu()
            net_rpys = SO3.to_rpy(net_Rots).cpu()
            imu_t = self.gt['ts'][:imu_rpys.shape[0]]
            imu_t = np.expand_dims(imu_t, axis=1)
            net_t = self.gt['ts'][:net_rpys.shape[0]]
            net_t = np.expand_dims(net_t, axis=1)
            imu_rpys = np.hstack((imu_t, imu_rpys))
            net_rpys = np.hstack((net_t, net_rpys))
            header = "timestamp(s),roll,pitch,yaw"
            np.savetxt(imu_rpys_path, imu_rpys, header=header, delimiter=",", fmt='%1.9f')
            np.savetxt(net_rpys_path, net_rpys, header=header, delimiter=",", fmt='%1.9f')
            print("raw imu rpy is saved in \'%s\'" % imu_rpys_path)
            print("net imu rpy is saved in \'%s\'" % net_rpys_path)

    def integrate_with_quaternions_superfast(self, N, raw_us, net_us, quat_gt):
        imu_qs = SO3.qnorm(SO3.qexp(raw_us[:, :3].cuda().double()*self.dt))
        net_qs = SO3.qnorm(SO3.qexp(net_us[:, :3].cuda().double()*self.dt))
        Rot0 = SO3.qnorm(quat_gt[:2].cuda().double())
        imu_qs[0] = Rot0[0]
        net_qs[0] = Rot0[0]

        N = np.log2(imu_qs.shape[0])
        for i in range(int(N)):
            k = 2**i
            imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:-k], imu_qs[k:]))
            net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:-k], net_qs[k:]))

        if int(N) < N:
            k = 2**int(N)
            k2 = imu_qs[k:].shape[0]
            # print("imu_qs: ", imu_qs.shape)
            # print("k: %d, k2: %d"%(k, k2))
            # print("imu_qs: qmul with %d x %d" % (imu_qs[:k2].shape[0], imu_qs[k:].shape[0]))
            # print("net_qs: qmul with %d x %d" % (net_qs[:k2].shape[0], net_qs[k:].shape[0]))
            imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:k2], imu_qs[k:]))
            net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:k2], net_qs[k:]))

        imu_Rots = SO3.from_quaternion(imu_qs).float()
        net_Rots = SO3.from_quaternion(net_qs).float()
        return net_qs.cpu(), imu_Rots, net_Rots

    def plot_gyro(self):
        N = self.raw_us.shape[0]
        raw_us = self.raw_us[:, :3]
        net_us = self.w_hat[:, :3]

        net_qs, imu_Rots, net_Rots = self.integrate_with_quaternions_superfast(N, raw_us, net_us)
        imu_rpys = (180/np.pi)*SO3.to_rpy(imu_Rots).cpu()
        net_rpys = (180/np.pi)*SO3.to_rpy(net_Rots).cpu()
        self.plot_orientation(imu_rpys, net_rpys, N)
        self.plot_orientation_error(imu_Rots, net_Rots, N)

    def plot_accel(self, a_hat, gt_interpolated):

        pos_gt = gt_interpolated[:, 1:4]
        vel_gt = gt_interpolated[:, 8:11]

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].plot(self.ts, vel_gt[:, 0], color='red', label='$v_x$ [m/s]')
        axs[1].plot(self.ts, vel_gt[:, 1], color='blue', label='$v_y$ [m/s]')
        axs[2].plot(self.ts, vel_gt[:, 2], color='black', label='$v_z$ [m/s]')
        self.savefig(axs, fig, 'velocity_gt')
        plt.close(fig)

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].plot(self.ts, pos_gt[:, 0], color='red', label='$p_x$ [m]')
        axs[1].plot(self.ts, pos_gt[:, 1], color='blue', label='$p_y$ [m]')
        axs[2].plot(self.ts, pos_gt[:, 2], color='black', label='$p_z$ [m]')
        self.savefig(axs, fig, 'position_gt')
        plt.close(fig)

        # 1 (2와 똑같이 나옴, 2 구현이 더 직관적)
        # dv = self.dt * (a_hat[1:] + a_hat[:-1]) / 2.0
        # v_hat = torch.ones_like(vel_gt, dtype=torch.float32) * vel_gt[0, :]
        # print('v_hat:', v_hat.shape)
        # print(v_hat)
        # print('a_hat:', a_hat.shape)
        # print('vel_gt:', vel_gt.shape)
        # print('dv:', dv.shape)
        # for i in range(dv.shape[0]):
        #     v_hat[i+1:] += dv[:dv.shape[0]-i]
        # fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        # axs[0].plot(self.ts, v_hat[:, 0], color='red', label='$v_x$ [m/s]')
        # axs[1].plot(self.ts, v_hat[:, 1], color='blue', label='$v_y$ [m/s]')
        # axs[2].plot(self.ts, v_hat[:, 2], color='black', label='$v_z$ [m/s]')
        # self.savefig(axs, fig, 'vel_estimate_1')
        # plt.close(fig)

        # 2
        dv = self.dt * (a_hat[1:] + a_hat[:-1]) / 2.0
        v_hat = torch.ones_like(vel_gt, dtype=torch.float32)
        v_hat[0] = vel_gt[0]
        v_hat[1:] = dv
        for i in range(1, v_hat.shape[0]):
            v_hat[i] += v_hat[i-1]
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].plot(self.ts, v_hat[:, 0], color='red', label='$v_x$ [m/s]')
        axs[1].plot(self.ts, v_hat[:, 1], color='blue', label='$v_y$ [m/s]')
        axs[2].plot(self.ts, v_hat[:, 2], color='black', label='$v_z$ [m/s]')
        self.savefig(axs, fig, 'vel_estimate')
        plt.close(fig)

    def plot_orientation(self, N, rpy_imu, rpy_hat, rpy_gt):
        title = "Orientation estimation"

        # print("rpy_imu:", rpy_imu.shape, rpy_imu.dtype)
        # print("rpy_hat:", rpy_hat.shape, rpy_hat.dtype)
        # print("rpy_gt:", rpy_gt.shape, rpy_gt.dtype)

        rpy_gt = rpy_gt[:N]

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        for i in range(3):
            # axs[i].plot(self.ts, rpy_imu[:, i]%360, color='red', label=r'raw IMU')
            axs[i].plot(self.ts, rpy_hat[:, i]%360, color='blue', label=r'net IMU')
            axs[i].plot(self.ts, rpy_gt[:, i]%360, color='black', label=r'ground truth')
            axs[i].set_xlim(self.ts[0], self.ts[-1])

        self.savefig(axs, fig, 'orientation')
        plt.close(fig)

    def plot_orientation_error(self, N, rot_imu, rot_hat, rot_gt):
        rot_gt = rot_gt[:N].cuda()
        err_imu = (180/np.pi)*SO3.log(bmtm(rot_imu, rot_gt)).cpu()
        err_hat = (180/np.pi)*SO3.log(bmtm(rot_hat, rot_gt)).cpu()
        title = "$SO(3)$ orientation error"
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        for i in range(3):
            axs[i].plot(self.ts, err_imu[:, i] % 360, color='red', label=r'raw IMU')
            axs[i].plot(self.ts, err_hat[:, i] % 360, color='blue', label=r'net IMU')
            axs[i].set_xlim(self.ts[0], self.ts[-1])

        self.savefig(axs, fig, 'orientation_error')
        plt.close(fig)


    # def plot_accel(self, a_hat, dv_gt, dp_gt):
    #     N = self.raw_us.shape[0]
    #     raw_acc = self.raw_us[:, 3:6]
    #     net_acc = self.a_hat

    #     raw_dv = self.dt * (raw_acc[1:] + raw_acc[:-1]) / 2.0
    #     net_dv = self.dt * (net_acc[1:] + net_acc[:-1]) / 2.0

    #     v0 = 0.0 # 임시


    #     net_qs, imu_Rots, net_Rots = self.integrate_with_quaternions_superfast(N, raw_us, net_us)
    #     imu_rpys = 180/np.pi*SO3.to_rpy(imu_Rots).cpu()
    #     net_rpys = 180/np.pi*SO3.to_rpy(net_Rots).cpu()
    #     self.plot_orientation(imu_rpys, net_rpys, N)
    #     self.plot_orientation_error(imu_Rots, net_Rots, N)


    def plot_gyro_correction(self, gyro_corrections):
        title = "Gyro correction" + self.end_title
        ylabel = 'gyro correction (deg/s)'
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set(xlabel='$t$ (min)', ylabel=ylabel, title=title)
        plt.plot(self.ts, gyro_corrections[:, 0], label='correction of $w_x$')
        plt.plot(self.ts, gyro_corrections[:, 1], label='correction of $w_y$')
        plt.plot(self.ts, gyro_corrections[:, 2], label='correction of $w_z$')

        self.savefig(ax, fig, 'gyro_correction')
        plt.close(fig)

    def plot_accel_correction(self, accel_corrections):
        title = "Accel correction" + self.end_title
        ylabel = 'accel correction ($m/s^2$)'
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set(xlabel='$t$ (min)', ylabel=ylabel, title=title)
        plt.plot(self.ts, accel_corrections[:, 0], label='correction of $a_x$')
        plt.plot(self.ts, accel_corrections[:, 1], label='correction of $a_y$')
        plt.plot(self.ts, accel_corrections[:, 2], label='correction of $a_z$')

        self.savefig(ax, fig, 'accel_correction')
        plt.close(fig)

    @property
    def end_title(self):
        return " for sequence " + self.seq.replace("_", " ")

    def savefig(self, axs, fig, name):
        if isinstance(axs, np.ndarray):
            for i in range(len(axs)):
                axs[i].grid()
                axs[i].legend()
        else:
            axs.grid()
            axs.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.params['result_dir'], self.seq, name + '.png'))

