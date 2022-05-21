#! /bin/bash

# [220519 23:20]
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode train \
#     --c0 16 --dv_normed 16 32 --net_version='ver1' \
#     --id CompareTilde-ver1-c16-dv_16_32-dvn_16_32
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode test \
#     --c0 16 --dv_normed 16 32 --net_version='ver1' \
#     --id CompareTilde-ver1-c16-dv_16_32-dvn_16_32
CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode anal \
    --c0 16 --dv_normed 16 32 --net_version='ver1' \
    --id CompareTilde-ver1-c16-dv_16_32-dvn_16_32

# # [220519 15:26]
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode train \
#     --c0 32 --dv_normed 16 32 --net_version='ver2' \
#     --id CompareTilde-ver2-c32-dv_16_32-dvn_16_32
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode test \
#     --c0 32 --dv_normed 16 32 --net_version='ver2' \
#     --id CompareTilde-ver2-c32-dv_16_32-dvn_16_32

# # [220518 03:32]
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode train \
#     --c0 32 --dv_normed 16 32 --net_version='ver2' \
#     --id 220518-c32-dv_16_32-dvn_16_32
# CUDA_VISIBLE_DEVICES=7 python3 main_cnn_v0.py --mode test \
#     --c0 32 --dv_normed 16 32 --net_version='ver2' \
#     --id 220518-c32-dv_16_32-dvn_16_32
# CUDA_VISIBLE_DEVICES=1 python3 main_cnn_v0.py --mode anal \
#     --c0 32 --dv_normed 16 32 --net_version='ver2' \
#     --id 220518-c32-dv_16_32-dvn_16_32


# python3 main_cnn_v0.py --mode test  --c0 32 --dv_normed 16 32 --id 220511_test
# python3 main_cnn_v0.py --mode anal  --c0 32 --dv_normed 16 32 --id 220511_test

