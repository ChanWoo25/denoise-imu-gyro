import torch
import os
import pickle
import yaml
import numpy as np

def vnorm(velocity, window_size):
    """
        Normalize velocity with latest window data.
        - Note that std is not divided. Only subtract mean
        - data should have dimension 3.
    """
    v = velocity
    N = v.shape[1]

    if v.dim() is not 3:
        print("velocity's dim must be 3 for batch operation")
        exit(-1)

    on_gpu = v.is_cuda
    if not on_gpu and torch.cuda.is_available():
        v = v.cuda()

    padding_size = window_size - 1
    batch_size = v.shape[0]

    pad = v[:, 0, :].reshape(batch_size, 1, 3).expand(batch_size, padding_size, 3)
    v_normed = torch.cat([pad, v], dim=1)

    for i in range(window_size-1, 0, -1):
        v_normed[:, i-1:i-1+N, :] += v

    v_normed = v_normed[:, :N, :] / float(window_size)
    v_normed = v - v_normed

    if not on_gpu:
        v_normed = v_normed.cpu()

    return v_normed

def fast_acc_integration(a_hat, dt=0.005):
    """
        Integrate acceleration fast.
        - data should have dimension 3.
    """

    if a_hat.dim() is not 3:
        print("a_hat's dimension must be 3")
        exit(-1)

    on_gpu = a_hat.is_cuda
    if not on_gpu and torch.cuda.is_available():
        a_hat = a_hat.cuda()

    pad = torch.zeros_like(a_hat[:, 0, :], device=a_hat.device).unsqueeze(1)
    if not on_gpu and torch.cuda.is_available():
        pad = pad.cuda()

    N = a_hat.shape[1]
    n = np.log2(N)
    dv_hat = (( (a_hat[:, 1:] + a_hat[:, :-1]) / 2.0) * dt)
    dv_hat = torch.cat([pad, dv_hat], dim=1)
    for i in range(int(n)):
        k = 2**i
        dv_hat[:, k:] = dv_hat[:, :-k] + dv_hat[:, k:]
    if int(n) < n:
        k = 2**int(n)
        k2 = N - k
        dv_hat[:, k:] = dv_hat[:, :k2] + dv_hat[:, k:]

    if not on_gpu:
        dv_hat = dv_hat.cpu()

    return dv_hat

def pload(*f_names):
    """Pickle load"""
    f_name = os.path.join(*f_names)
    with open(f_name, "rb") as f:
        pickle_dict = pickle.load(f)
    return pickle_dict

def pdump(pickle_dict, *f_names):
    """Pickle dump"""
    f_name = os.path.join(*f_names)
    with open(f_name, "wb") as f:
        pickle.dump(pickle_dict, f)

def mkdir(*paths):
    '''Create a directory if not existing.'''
    path = os.path.join(*paths)
    if not os.path.exists(path):
        os.mkdir(path)

def yload(*f_names):
    """YAML load"""
    f_name = os.path.join(*f_names)
    with open(f_name, 'r') as f:
        yaml_dict = yaml.load(f)
    return yaml_dict

def ydump(yaml_dict, *f_names):
    """YAML dump"""
    f_name = os.path.join(*f_names)
    with open(f_name, 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)

def bmv(mat, vec):
    """batch matrix vector product"""
    return torch.einsum('bij, bj -> bi', mat, vec)

def bbmv(mat, vec):
    """double batch matrix vector product"""
    return torch.einsum('baij, baj -> bai', mat, vec)

def bmtv(mat, vec):
    """batch matrix transpose vector product"""
    return torch.einsum('bji, bj -> bi', mat, vec)

def bmtm(mat1, mat2):
    """batch matrix transpose matrix product"""
    return torch.einsum("bji, bjk -> bik", mat1, mat2)

def bmmt(mat1, mat2):
    """batch matrix matrix transpose product"""
    return torch.einsum("bij, bkj -> bik", mat1, mat2)
