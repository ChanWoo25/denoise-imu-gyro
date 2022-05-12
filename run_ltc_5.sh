#! /bin/bash

# [220512 15:00] 비교실험 (https://www.notion.so/LTC_CELL-ode_unfolds-d108546ac2224eb09f50d7258e02dd5d)
CUDA_VISIBLE_DEVICES=1 python3 main_ltc_v0.py --mode='train' \
  --machine='server' --input_type 'raw' \
  --seq_len 16000 --train_batch_size 1 --goal_epoch=500 \
  --id='220512_Raw_Unfold_1' \
  --n_inter=36 --n_command=20 --out_sensory=12 \
  --out_inter=10 --rec_command=10 --in_motor=10 \
  --ode_unfolds=1

# [220510]
# CUDA_VISIBLE_DEVICES=3 python3 main_ltc_v0.py --mode='train' \
#   --machine='server' --input_type 'raw' \
#   --seq_len 16000 --train_batch_size 3 \
#   --id='220510_Raw_V2' \
#   --n_inter=48 --n_command=24 --out_sensory=12 \
#   --out_inter=15 --rec_command=15 --in_motor=12

# # [220511 13:19 - 400 Epoch 돌리고 테스트]
# CUDA_VISIBLE_DEVICES=3 python3 main_ltc_v0.py --mode='test' \
#   --test_path='/home/leecw/project/log/220510_Raw_V2/epoch=369-val_loss=0.63.ckpt' \
#   --machine='server' --input_type 'raw' \
#   --seq_len 16000 --train_batch_size 3 \
#   --id='220510_Raw_V2' \
#   --n_inter=48 --n_command=24 --out_sensory=12 \
#   --out_inter=15 --rec_command=15 --in_motor=12
