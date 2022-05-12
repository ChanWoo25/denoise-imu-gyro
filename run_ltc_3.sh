#! /bin/bash

# [220512 15:00] 비교실험 (https://www.notion.so/LTC_CELL-ode_unfolds-d108546ac2224eb09f50d7258e02dd5d)
CUDA_VISIBLE_DEVICES=1 python3 main_ltc_v0.py --mode='train' \
  --machine='server' --input_type 'raw' \
  --seq_len 16000 --train_batch_size 1 --goal_epoch=500 \
  --id='220512_Raw_Unfold_3' \
  --n_inter=36 --n_command=20 --out_sensory=12 \
  --out_inter=10 --rec_command=10 --in_motor=10 \
  --ode_unfolds=3

# # [220511]
# CUDA_VISIBLE_DEVICES=1 python3 main_ltc_v0.py --mode='train' \
#   --resume_path='/home/leecw/project/log/220510_Raw_V1/last.ckpt' \
#   --goal_epoch=600 \
#   --machine='server' --input_type 'raw' \
#   --seq_len 16000 --train_batch_size 6 \
#   --id='220510_Raw_V1' \
#   --n_inter=32 --n_command=16 --out_sensory=8 \
#   --out_inter=10 --rec_command=10 --in_motor=8
