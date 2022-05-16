#! /bin/bash

# [220511 03:14 Noise Loss log scale로 학습: <1일때, -쪽으로 확, >=1일 때 적당히 경사]
# python3 main_cnn_v0.py --mode train --c0 32 --dv_normed 16 32 --id 220510_c32_v16_v32_loggap
python3 main_cnn_v0.py --mode test  --c0 32 --dv_normed 16 32 --id 220510_c32_v16_v32_loggap
python3 main_cnn_v0.py --mode anal  --c0 32 --dv_normed 16 32 --id 220510_c32_v16_v32_loggap
# python3 main_cnn_v0.py --mode train --c0 64 --dv_normed 16 32 --id 220510_c64_v16_v32_loggap
python3 main_cnn_v0.py --mode test  --c0 64 --dv_normed 16 32 --id 220510_c64_v16_v32_loggap
python3 main_cnn_v0.py --mode anal  --c0 64 --dv_normed 16 32 --id 220510_c64_v16_v32_loggap
