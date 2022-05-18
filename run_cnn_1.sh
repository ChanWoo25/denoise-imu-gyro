#! /bin/bash

# [220511 01:39 - 첫 노이즈 Loss 도입]
# python3 main_cnn_v0.py --mode train --c0 32 --dv_normed 16 32 --id 220510_c32_v16_v32_gap
# python3 main_cnn_v0.py --mode test  --c0 32 --dv_normed 16 32 --id 220510_c32_v16_v32_gap
# python3 main_cnn_v0.py --mode anal  --c0 32 --dv_normed 16 32 --id 220510_c32_v16_v32_gap
# python3 main_cnn_v0.py --mode train --c0 64 --dv_normed 16 32 --id 220510_c64_v16_v32_gap
# python3 main_cnn_v0.py --mode test  --c0 64 --dv_normed 16 32 --id 220510_c64_v16_v32_gap
# python3 main_cnn_v0.py --mode anal  --c0 64 --dv_normed 16 32 --id 220510_c64_v16_v32_gap

# [220511 03:32 - Noise Loss 굉장히 안 좋은 생각 같다... 윈도우 사이즈라도 줄여보자]
CUDA_VISIBLE_DEVICES=1 python3 main_cnn_v0.py --mode train \
    --c0 32 --dv_normed 16 32 --net_version='ver2' \
    --id 220518-c32-dv_16_32-dvn_16_32
CUDA_VISIBLE_DEVICES=1 python3 main_cnn_v0.py --mode test \
    --c0 32 --dv_normed 16 32 --net_version='ver2' \
    --id 220518-c32-dv_16_32-dvn_16_32
CUDA_VISIBLE_DEVICES=1 python3 main_cnn_v0.py --mode anal \
    --c0 32 --dv_normed 16 32 --net_version='ver2' \
    --id 220518-c32-dv_16_32-dvn_16_32


# python3 main_cnn_v0.py --mode test  --c0 32 --dv_normed 16 32 --id 220511_test
# python3 main_cnn_v0.py --mode anal  --c0 32 --dv_normed 16 32 --id 220511_test

