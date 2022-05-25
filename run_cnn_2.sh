#! /bin/bash

# [220525 14:30] Gaussian NLL Loss 적용 Test
CUDA_VISIBLE_DEVICES=5 python3 main_cnn_v0.py --mode test \
    --c0 16 --dv_normed 16 32 --net_version='ori_ver2' \
    --id 220524-ori_ver2-c16-dvn_16_32-00

# # [220524 20:54] Gaussian NLL Loss 적용 Test
# CUDA_VISIBLE_DEVICES=5 python3 main_cnn_v0.py --mode train \
#     --c0 16 --dv_normed 16 32 --net_version='ori_ver2' \
#     --id 220524-ori_ver2-c16-dvn_16_32-00

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

# # [220511 03:14 Noise Loss log scale로 학습: <1일때, -쪽으로 확, >=1일 때 적당히 경사]
# # python3 main_cnn_v0.py --mode train --c0 32 --dv_normed 16 32 --id 220510_c32_v16_v32_loggap
# python3 main_cnn_v0.py --mode test  --c0 32 --dv_normed 16 32 --id 220510_c32_v16_v32_loggap
# python3 main_cnn_v0.py --mode anal  --c0 32 --dv_normed 16 32 --id 220510_c32_v16_v32_loggap
# # python3 main_cnn_v0.py --mode train --c0 64 --dv_normed 16 32 --id 220510_c64_v16_v32_loggap
# python3 main_cnn_v0.py --mode test  --c0 64 --dv_normed 16 32 --id 220510_c64_v16_v32_loggap
# python3 main_cnn_v0.py --mode anal  --c0 64 --dv_normed 16 32 --id 220510_c64_v16_v32_loggap
