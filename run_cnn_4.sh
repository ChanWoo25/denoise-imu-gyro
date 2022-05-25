#! /bin/bash

# [220525 14:30] Gaussian NLL Loss 적용 Test
CUDA_VISIBLE_DEVICES=5 python3 main_cnn_v0.py --mode test \
    --c0 16 --dv_normed 16 32 --net_version='ori_ver2' \
    --ori_gnll_ratio=0.1 \
    --id 220524-ori_ver2-c16-dvn_16_32-ratio01

# [220525 14:30] Gaussian NLL Loss 적용 Test
CUDA_VISIBLE_DEVICES=5 python3 main_cnn_v0.py --mode test \
    --c0 16 --dv_normed 16 32 --net_version='ori_ver2' \
    --ori_gnll_ratio=0.02 \
    --id 220524-ori_ver2-c16-dvn_16_32-ratio002

# # [220524 20:54] Gaussian NLL Loss 적용 Test
# CUDA_VISIBLE_DEVICES=5 python3 main_cnn_v0.py --mode train \
#     --c0 16 --dv_normed 16 32 --net_version='ori_ver2' \
#     --ori_gnll_ratio=0.1 \
#     --id 220524-ori_ver2-c16-dvn_16_32-ratio01

# # [220524 22:00] Gaussian NLL Loss 적용 Test
# CUDA_VISIBLE_DEVICES=5 python3 main_cnn_v0.py --mode train \
#     --c0 16 --dv_normed 16 32 --net_version='ori_ver2' \
#     --ori_gnll_ratio=0.02 \
#     --id 220524-ori_ver2-c16-dvn_16_32-ratio002

# [220519 01:33] -- Loss 동일하게 주고, world frame 보정만 하는 걸로 실험
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode train \
#     --c0 32 --dv_normed 16 32 --net_version='ver3' \
#     --id 220518-ver3-c32-dv_16_32-dvn_16_32
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode test \
#     --c0 32 --dv_normed 16 32 --net_version='ver3' \
#     --id 220518-ver3-c32-dv_16_32-dvn_16_32
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode anal \
#     --c0 32 --dv_normed 16 32 --net_version='ver3' \
#     --id 220518-ver3-c32-dv_16_32-dvn_16_32
