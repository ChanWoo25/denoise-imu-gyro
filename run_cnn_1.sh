#! /bin/bash

# [220524 20:24] Gaussian NLL Loss 적용 Test
CUDA_VISIBLE_DEVICES=5 python3 main_cnn_v0.py --mode train \
    --c0 16 --dv_normed 16 32 --net_version='ori_ver1' \
    --id 220524-gyro-c16-dvn_16_32-00

# # [220511 03:32 - Noise Loss 굉장히 안 좋은 생각 같다... 윈도우 사이즈라도 줄여보자]
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode train \
#     --c0 32 --dv_normed 16 32 --net_version='ver2' \
#     --id 220518-c32-dv_16_32-dvn_16_32
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode test \
#     --c0 32 --dv_normed 16 32 --net_version='ver2' \
#     --id 220518-c32-dv_16_32-dvn_16_32 \
#     --load_weight_path='/home/leecw/project/results/DenoiseIMU/220518-c32-dv_16_32-dvn_16_32/ep_0720_best.pt'
# CUDA_VISIBLE_DEVICES=1 python3 main_cnn_v0.py --mode anal \
#     --c0 32 --dv_normed 16 32 --net_version='ver2' \
#     --id 220518-c32-dv_16_32-dvn_16_32


# python3 main_cnn_v0.py --mode test  --c0 32 --dv_normed 16 32 --id 220511_test
# python3 main_cnn_v0.py --mode anal  --c0 32 --dv_normed 16 32 --id 220511_test

