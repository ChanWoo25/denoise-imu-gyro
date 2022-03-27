
from email import header
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
from src.utils import bmtm, bmtv, bmmt
from datetime import datetime
from src.lie_algebra import SO3, CPUSO3

# RESULT_DIR = "/root/Data/Result"

class LearningBasedProcessing:
    def __init__(self, params, net_class, net_params, address, dt):

        self.params = params
        self.net_class = net_class
        self.net_params = net_params

        self._ready = False
        self.figsize = (20, 12)
        self.dt = dt # (s)

        self.address = address
        self.weight_path = os.path.join(self.address, 'weights.pt')

        if self.params['is_train']:
            pdump(self.net_params, self.address, 'net_params.p')
            ydump(self.net_params, self.address, 'net_params.yaml')
            self.net = self.net_class(**self.net_params)
        else:
            self.net_params = pload(self.address, 'net_params.p')
            self.params = pload(self.address, 'train_params.p')
            self.net = self.net_class(**self.net_params)
            weights = torch.load(self.weight_path)
            self.net.load_state_dict(weights)

        self.net.cuda()

    # def find_address(self, address):
    #     """return path where net and training info are saved"""
    #     if address == 'last':
    #         addresses = sorted(os.listdir(self.res_dir))
    #         address = os.path.join(self.res_dir, addresses[-1])
    #     elif address is None:
    #         now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    #         address = os.path.join(self.res_dir, now)
    #         mkdir(address)
    #     return address

    def train(self, dataset_class, dataset_params):
        """train the neural network. GPU is assumed"""
        pdump(self.params, self.address, 'train_params.p')
        ydump(self.params, self.address, 'train_params.yaml')

        hparams = self.get_hparams(dataset_class, dataset_params)
        ydump(hparams, self.address, 'hparams.yaml')

        # define datasets
        dataset_train = dataset_class(**dataset_params, mode='train')
        dataset_train.init_train()
        dataset_val = dataset_class(**dataset_params, mode='val')
        dataset_val.init_val()

        # get class
        Optimizer = self.params['optimizer_class']
        Scheduler = self.params['scheduler_class']
        Loss = self.params['loss_class']

        # get parameters
        dataloader_params = self.params['dataloader']
        optimizer_params = self.params['optimizer']
        scheduler_params = self.params['scheduler']
        loss_params = self.params['loss']

        # define optimizer, scheduler and loss
        dataloader = DataLoader(dataset_train, **dataloader_params)
        optimizer = Optimizer(self.net.parameters(), **optimizer_params)
        scheduler = Scheduler(optimizer, **scheduler_params)
        criterion = Loss(**loss_params)

        # remaining training parameters
        freq_val = self.params['freq_val']
        n_epochs = self.params['n_epochs']

        # init net w.r.t dataset
        self.net = self.net.cuda()
        mean_u, std_u = dataset_train.mean_u, dataset_train.std_u
        self.net.set_normalized_factors(mean_u, std_u)

        # start tensorboard writer
        writer = SummaryWriter(self.address)
        start_time = time.time()
        best_loss = torch.Tensor([float('Inf')])

        # define some function for seeing evolution of training
        def write(epoch, loss_epoch):
            writer.add_scalar('loss/train', loss_epoch.item(), epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            if epoch % 200 == 0:
                print('Train Epoch: {:2d} \tLoss: {:.4f}'.format(epoch, loss_epoch.item()))
            scheduler.step(epoch)

        def write_time(epoch, start_time):
            delta_t = time.time() - start_time
            print("Amount of time spent for epochs " +
                "{}-{}: {:.1f}s\n".format(epoch - freq_val, epoch, delta_t))
            writer.add_scalar('time_spend', delta_t, epoch)

        def write_val(loss, best_loss):
            if 0.5*loss <= best_loss:
                msg = 'validation loss decreases! :) '
                msg += '(curr/prev loss {:.4f}/{:.4f})'.format(loss.item(),
                    best_loss.item())
                cprint(msg, 'green')
                best_loss = loss
                self.save_net()
            else:
                msg = 'validation loss increases! :( '
                msg += '(curr/prev loss {:.4f}/{:.4f})'.format(loss.item(),
                    best_loss.item())
                cprint(msg, 'yellow')
            writer.add_scalar('loss/val', loss.item(), epoch)
            return best_loss

        # training loop !
        for epoch in range(1, n_epochs + 1):
            loss_epoch = self.loop_train(dataloader, optimizer, criterion)
            write(epoch, loss_epoch)
            scheduler.step(epoch)
            if epoch % freq_val == 0:
                loss = self.loop_val(dataset_val, criterion)
                write_time(epoch, start_time)
                best_loss = write_val(loss, best_loss)
                start_time = time.time()
        # training is over !

        # test on new data
        dataset_test = dataset_class(**dataset_params, mode='test')

        weights = torch.load(self.weight_path)
        self.net.load_state_dict(weights)
        self.net.cuda()

        test_loss = self.loop_val(dataset_test, criterion)
        dict_loss = {
            'final_loss/val': best_loss.item(),
            'final_loss/test': test_loss.item()
            }
        writer.add_hparams(hparams, dict_loss)
        ydump(dict_loss, self.address, 'final_loss.yaml')
        writer.close()

    def loop_train(self, dataloader, optimizer, criterion):
        """Forward-backward loop over training data"""
        loss_epoch = 0
        optimizer.zero_grad()
        for us, xs in dataloader:
            us = dataloader.dataset.add_noise(us.cuda())
            hat_xs = self.net(us)
            loss = criterion(xs.cuda(), hat_xs)/len(dataloader)
            loss.backward()
            loss_epoch += loss.detach().cpu()
        optimizer.step()
        return loss_epoch

    def loop_val(self, dataset, criterion):
        """Forward loop over validation data"""
        loss_epoch = 0
        self.net.eval()
        with torch.no_grad():
            for i in range(len(dataset)):
                us, xs = dataset[i]
                hat_xs = self.net(us.cuda().unsqueeze(0))
                loss = criterion(xs.cuda().unsqueeze(0), hat_xs)/len(dataset)
                loss_epoch += loss.cpu()
        self.net.train()
        return loss_epoch

    def save_net(self):
        """save the weights on the net in CPU"""
        self.net.eval().cpu()
        torch.save(self.net.state_dict(), self.weight_path)
        self.net.train().cuda()

    def get_hparams(self, dataset_class, dataset_params):
        """return all training hyperparameters in a dict"""
        Optimizer = self.params['optimizer_class']
        Scheduler = self.params['scheduler_class']
        Loss = self.params['loss_class']

        # get training class parameters
        dataloader_params = self.params['dataloader']
        optimizer_params = self.params['optimizer']
        scheduler_params = self.params['scheduler']
        loss_params = self.params['loss']

        # remaining training parameters
        freq_val = self.params['freq_val']
        n_epochs = self.params['n_epochs']

        dict_class = {
            'Optimizer': str(Optimizer),
            'Scheduler': str(Scheduler),
            'Loss': str(Loss)
        }

        return {**dict_class, **dataloader_params, **optimizer_params,
                **loss_params, **scheduler_params,
                'n_epochs': n_epochs, 'freq_val': freq_val}

    def test(self, dataset_class, dataset_params):
        """test a network once training is over"""

        # get loss function
        Loss = self.params['loss_class']
        loss_params = self.params['loss']
        criterion = Loss(**loss_params)

        self.dataset = dataset_class(mode='test',**dataset_params)
        self.loop_test(criterion)
        self.display_test()

    def loop_test(self, criterion):
        """Forward loop over test data"""
        self.net.eval()
        for i in range(len(self.dataset)):
            seq = self.dataset.sequences[i]
            us, xs = self.dataset[i]
            with torch.no_grad():
                hat_xs = self.net(us.cuda().unsqueeze(0))


            ### DEBUG
            # print("hat_xs:", hat_xs.shape) # [1, 29952, 3]
            # if seq in ['MH_02_easy', 'MH_04_difficult']:
            #     print("Test -- MH_02_easy")
            #     print("\tus:", us.shape)
            #     print("\txs:", xs.shape)
            #     print("\that_xs:", hat_xs.shape)
            #     gt_processed_path = os.path.join("/root/Data/Result/DenoiseIMU/gt", seq, "gt_processed.csv")
            #     gt_processed = np.loadtxt(gt_processed_path, delimiter=',')
            #     print("\tgt_processed from {} -- shape({})".format(seq, gt_processed.shape))
            ###

            loss = criterion(xs.cuda().unsqueeze(0), hat_xs)
            mkdir(self.address, seq)
            mondict = {
                'hat_xs': hat_xs[0].cpu(),
                'loss': loss.cpu().item(),
            }
            pdump(mondict, self.address, seq, 'results.p')

    def display_test(self):
        raise NotImplementedError


class GyroLearningBasedProcessing(LearningBasedProcessing):
    def __init__(self, params, net_class, net_params, address, dt):
        super().__init__(params, net_class, net_params, address, dt)
        self.roe_dist = [7, 14, 21, 28, 35] # m
        self.freq = 100 # subsampling frequency for RTE computation

    def display_test(self):

        self.to_open_vins()
        for i, seq in enumerate(self.dataset.sequences):
            # print('\n', 'Results for sequence ' + seq )
            self.seq = seq
            # get ground truth
            self.gt = self.dataset.load_gt(i)
            Rots = SO3.from_quaternion(self.gt['qs'].cuda())
            self.gt['Rots'] = Rots.cpu()
            self.gt['rpys'] = SO3.to_rpy(Rots).cpu()
            # get data and estimate
            self.net_us = pload(self.address, seq, 'results.p')['hat_xs']
            self.raw_us, _ = self.dataset[i]

            ###
            # print("\tself.net_us:", self.net_us.shape)
            # print("\t\t", self.net_us[:5])
            # print("\tself.raw_us:", self.raw_us.shape)
            # print("\t\t", self.raw_us[:5])
            ###

            N = self.net_us.shape[0]
            self.gyro_corrections =  (self.raw_us[:, :3] - self.net_us[:N, :3])
            self.ts = torch.linspace(0, N*self.dt, N)

            self.convert()
            self.plot_gyro()
            self.plot_gyro_correction()
            # plt.show()

    def save_gyro_estimate(self, seq):
        net_us = pload(self.address, seq, 'results.p')['hat_xs']
        N = net_us.shape[0]
        path = os.path.join("/home/leecw/Data/Result/DenoiseIMU/estimate", seq, seq + '_net_us.csv')
        header = "time(s),wx,wy,wz"
        x = np.zeros(N, 4)
        x[:, 0]


    def to_open_vins(self):
        """
        Export results to Open-VINS format. Use them eval toolbox available
        at https://github.com/rpng/open_vins/
        """
        print("open_vins()")

        for i, seq in enumerate(self.dataset.sequences):
            self.seq = seq
            # get ground truth
            self.gt = self.dataset.load_gt(i)
            raw_us, _ = self.dataset[i]
            net_us = pload(self.address, seq, 'results.p')['hat_xs']
            N = net_us.shape[0]

            net_qs, imu_Rots, net_Rots = self.integrate_with_quaternions_superfast(N, raw_us, net_us)

            path = os.path.join(self.address, seq + '.csv')
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
                denoised_interpolated_imu_path = os.path.join(self.address, seq, 'denoised_imu.csv')
                denoised_interpolated_imu[:,1:4] = np.squeeze(net_us)
                header = "time[ns],wx,wy,wz,ax,ay,az"
                np.savetxt(denoised_interpolated_imu_path, denoised_interpolated_imu, fmt="%d,%1.9f,%1.9f,%1.9f,%1.9f,%1.9f,%1.9f", header=header)
                print("denoied imu data is saved in \'%s\'" % denoised_interpolated_imu_path)


            # Save net_us on csv file
            # net_us_csv = np.zeros((net_us))
            # np.savetxt("/root/Data/Result/DenoiseIMU/estimate/MH_02_easy/net_us.csv", )

            imu_rpys_path = os.path.join(self.address, seq, 'raw_rpy.csv')
            net_rpys_path = os.path.join(self.address, seq, 'net_rpy.csv')
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


    def convert(self):
        # s -> min
        l = 1/60
        self.ts *= l

        # rad -> deg
        l = 180/np.pi
        self.gyro_corrections *= l
        self.gt['rpys'] *= l

    def integrate_with_quaternions_superfast(self, N, raw_us, net_us):
        imu_qs = SO3.qnorm(SO3.qexp(raw_us[:, :3].cuda().double()*self.dt))
        net_qs = SO3.qnorm(SO3.qexp(net_us[:, :3].cuda().double()*self.dt))
        Rot0 = SO3.qnorm(self.gt['qs'][:2].cuda().double())
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
            print("[integrate_with_quaternions_superfast()]")
            print("imu_qs: ", imu_qs.shape)
            print("k: %d, k2: %d"%(k, k2))
            print("qmul with %d x %d" % (imu_qs[:k2].shape[0], imu_qs[k:].shape[0]))
            imu_qs[k:] = SO3.qnorm(SO3.qmul(imu_qs[:k2], imu_qs[k:]))
            net_qs[k:] = SO3.qnorm(SO3.qmul(net_qs[:k2], net_qs[k:]))

        imu_Rots = SO3.from_quaternion(imu_qs).float()
        net_Rots = SO3.from_quaternion(net_qs).float()
        return net_qs.cpu(), imu_Rots, net_Rots

    def plot_gyro(self):
        N = self.raw_us.shape[0]
        raw_us = self.raw_us[:, :3]
        net_us = self.net_us[:, :3]

        net_qs, imu_Rots, net_Rots = self.integrate_with_quaternions_superfast(N, raw_us, net_us)
        imu_rpys = 180/np.pi*SO3.to_rpy(imu_Rots).cpu()
        net_rpys = 180/np.pi*SO3.to_rpy(net_Rots).cpu()
        self.plot_orientation(imu_rpys, net_rpys, N)
        self.plot_orientation_error(imu_Rots, net_Rots, N)

    def plot_orientation(self, imu_rpys, net_rpys, N):
        title = "Orientation estimation"
        gt = self.gt['rpys'][:N]
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        for i in range(3):
            axs[i].plot(self.ts, gt[:, i]%360, color='black', label=r'ground truth')
            axs[i].plot(self.ts, imu_rpys[:, i]%360, color='red', label=r'raw IMU')
            axs[i].plot(self.ts, net_rpys[:, i]%360, color='blue', label=r'net IMU')
            axs[i].set_xlim(self.ts[0], self.ts[-1])
            ### Save csv file
            # header = 'time,roll,pitch,yaw'
            # np.savetxt(os.path.join(self.address, self.dataset.sequences[i] + '_gt_rpys.csv'), np.stack((self.tx, gt[:,i]), axis=1), fmt='%1.9f', delimiter=',', header=header)
            # np.savetxt(os.path.join(self.address, self.dataset.sequences[i] + '_imu_rpys.csv'), np.stack((self.tx, imu_rpys[:,i]), axis=1), fmt='%1.9f', delimiter=',', header=header)
            # np.savetxt(os.path.join(self.address, self.dataset.sequences[i] + '_net_rpys.csv'), np.stack((self.tx, net_rpys[:,i]), axis=1), fmt='%1.9f', delimiter=',', header=header)
            ###
        self.savefig(axs, fig, 'orientation')

    def plot_orientation_error(self, imu_Rots, net_Rots, N):
        gt = self.gt['Rots'][:N].cuda()
        raw_err = 180/np.pi*SO3.log(bmtm(imu_Rots, gt)).cpu()
        net_err = 180/np.pi*SO3.log(bmtm(net_Rots, gt)).cpu()
        title = "$SO(3)$ orientation error"
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=self.figsize)
        axs[0].set(ylabel='roll (deg)', title=title)
        axs[1].set(ylabel='pitch (deg)')
        axs[2].set(xlabel='$t$ (min)', ylabel='yaw (deg)')

        for i in range(3):
            axs[i].plot(self.ts, raw_err[:, i], color='red', label=r'raw IMU')
            axs[i].plot(self.ts, net_err[:, i], color='blue', label=r'net IMU')
            axs[i].set_ylim(-10, 10)
            axs[i].set_xlim(self.ts[0], self.ts[-1])
            ### Save csv file
            # header = 'time,roll,pitch,yaw'
            # np.savetxt(os.path.join(self.address, self.dataset.sequences[i] + '_raw_err.csv'), np.stack((self.ts, raw_err[:,i]), axis=1), fmt='%1.9f', delimiter=',', header=header)
            # np.savetxt(os.path.join(self.address, self.dataset.sequences[i] + '_net_err.csv'), np.stack((self.ts, net_err[:,i]), axis=1), fmt='%1.9f', delimiter=',', header=header)
            ###
        self.savefig(axs, fig, 'orientation_error')

    def plot_gyro_correction(self):
        title = "Gyro correction" + self.end_title
        ylabel = 'gyro correction (deg/s)'
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set(xlabel='$t$ (min)', ylabel=ylabel, title=title)
        plt.plot(self.ts, self.gyro_corrections, label=r'net IMU')
        ax.set_xlim(self.ts[0], self.ts[-1])
        self.savefig(ax, fig, 'gyro_correction')

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
        fig.savefig(os.path.join(self.address, self.seq, name + '.png'))

