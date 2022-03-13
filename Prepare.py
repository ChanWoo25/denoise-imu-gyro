import numpy as np
import torch
import os
from src.lie_algebra import SO3
from src.utils import pdump, pload, bmtv, bmtm

EUROC_DATA_DIR = "/root/Data/EUROC"
RESULT_DIR = "/root/Data/Result/DenoiseIMU/estimate"

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


def read_(sequence:str):
    """sequence[str] -- one of euroc_datasets"""
    path_imu = os.path.join(EUROC_DATA_DIR, sequence, "mav0", "imu0", "data.csv")
    path_gt = os.path.join(EUROC_DATA_DIR, sequence, "mav0", "state_groundtruth_estimate0", "data.csv")

    data_imu = np.loadtxt(path_imu, delimiter=',', comments='#')
    data_gt = np.loadtxt(path_gt, delimiter=',', comments='#')
    imu = data_imu
    print("imu shape: ", imu.shape)
    gt = data_gt
    print("gt shape: ", gt.shape)

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
    time_ns = imu[:, 0]

    # interpolate
    gt = interpolate(gt, gt[:, 0]/1e9, ts)
    print("new_gt shape: ", gt.shape)
    print("times shape: ", ts)
    print("time_ns:", time_ns.shape, time_ns.dtype)

    # take ground truth position
    p_gt = gt[:, 1:4]
    p_gt = p_gt - p_gt[0]

    # take ground true quaternion pose
    q_gt = torch.Tensor(gt[:, 4:8]).double()
    q_gt = q_gt / q_gt.norm(dim=1, keepdim=True)
    Rot_gt = SO3.from_quaternion(q_gt.cuda(), ordering='wxyz').cpu()
    print("Rot_gt shape: ", Rot_gt.shape)

    # convert from numpy
    p_gt = torch.Tensor(p_gt).double()
    v_gt = torch.tensor(gt[:, 8:11]).double()
    imu = torch.Tensor(imu[:, 1:]).double()

    # compute pre-integration factors for all training
    mtf = 16 # min_train_freq
    dRot_ij = bmtm(Rot_gt[:-mtf], Rot_gt[mtf:])
    dRot_ij = SO3.dnormalize(dRot_ij.cuda())
    print("dRot_ij:", dRot_ij.shape)
    dxi_ij = SO3.log(dRot_ij).cpu()

    header = "start_time[ns],end_time[ns],r11~r33"
    path_dRot = os.path.join(RESULT_DIR, sequence, "gt_dRot.csv")
    dRot_ij = dRot_ij.cpu()
    N = dRot_ij.shape[0]
    gt_dRot = np.zeros(shape=(N,11))
    gt_dRot[:, 0] = time_ns[:-mtf]
    gt_dRot[:, 1] = time_ns[mtf:]
    gt_dRot[:, 2:11] = dRot_ij.reshape(N, 9)
    np.savetxt(path_dRot, gt_dRot, header=header,
               fmt="%d,%d,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f,%1.6f")

    ### DEBUG
    # pre = dRot_ij[0:2]
    # print("FROM")
    # print(pre)
    # print("TO")
    # print(pre.reshape(2, 9))
    # print(type(imu[0, 0]), imu[0, 0])
        # FROM
        # tensor([[[ 0.9996,  0.0102,  0.0248],
        #         [-0.0106,  0.9998,  0.0167],
        #         [-0.0247, -0.0170,  0.9996]],

        #         [[ 0.9996,  0.0098,  0.0248],
        #         [-0.0102,  0.9998,  0.0162],
        #         [-0.0247, -0.0164,  0.9996]]], dtype=torch.float64)
        # TO
        # tensor([[ 0.9996,  0.0102,  0.0248, -0.0106,  0.9998,  0.0167, -0.0247, -0.0170,
        #         0.9996],
        #         [ 0.9996,  0.0098,  0.0248, -0.0102,  0.9998,  0.0162, -0.0247, -0.0164,
        #         0.9996]], dtype=torch.float64)
        # <class 'torch.Tensor'> tensor(-0.2150, dtype=torch.float64)
    ###

    # # save for all training
    # mondict = {
    #     'xs': dxi_ij.float(),
    #     'us': imu.float(),
    # }
    # pdump(mondict, self.predata_dir, sequence + ".p")
    # # save ground truth
    # mondict = {
    #     'ts': ts,
    #     'qs': q_gt.float(),
    #     'vs': v_gt.float(),
    #     'ps': p_gt.float(),
    # }
    # pdump(mondict, self.predata_dir, sequence + "_gt.p")

euroc_datasets = [
    "MH_01_easy",
    "MH_02_easy",
    "MH_03_medium",
    "MH_04_difficult",
    "MH_05_difficult"
]

read_("MH_02_easy")
