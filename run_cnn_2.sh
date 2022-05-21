#! /bin/bash

# [220519 23:20]
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode train \
#     --c0 16 --dv_normed 16 32 --net_version='ver2' \
#     --id CompareTilde-ver2-c16-dv_16_32-dvn_16_32
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode test \
#     --c0 16 --dv_normed 16 32 --net_version='ver2' \
#     --id CompareTilde-ver2-c16-dv_16_32-dvn_16_32
CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode anal \
    --c0 16 --dv_normed 16 32 --net_version='ver2' \
    --id CompareTilde-ver2-c16-dv_16_32-dvn_16_32

# # [220519 15:26]
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode train \
#     --c0 32 --dv_normed 16 32 --net_version='ver1' \
#     --id CompareTilde-ver1-c32-dv_16_32-dvn_16_32
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode test \
#     --c0 32 --dv_normed 16 32 --net_version='ver1' \
#     --id CompareTilde-ver1-c32-dv_16_32-dvn_16_32

# # [220519 00:37] -- Loss 동일하게 주고, body frame 보정만 하는 걸로 실험
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode train \
#     --c0 32 --dv_normed 16 32 --net_version='ver1' \
#     --id 220518-ver1-c32-dv_16_32-dvn_16_32
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode test \
#     --c0 32 --dv_normed 16 32 --net_version='ver1' \
#     --id 220518-ver1-c32-dv_16_32-dvn_16_32
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode anal \
#     --c0 32 --dv_normed 16 32 --net_version='ver1' \
#     --id 220518-ver1-c32-dv_16_32-dvn_16_32
