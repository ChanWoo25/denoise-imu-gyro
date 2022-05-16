#! /bin/bash

# [220512 15:00] 비교실험 (https://www.notion.so/LTC_CELL-ode_unfolds-d108546ac2224eb09f50d7258e02dd5d)
CUDA_VISIBLE_DEVICES=1 python3 main_ltc_v0.py --mode='train' \
  --resume_path='/home/leecw/project/log/220512_Raw_Unfold_6/epoch=459-val_loss=0.50.ckpt' \
  --machine='server' --input_type 'raw' \
  --seq_len 16000 --train_batch_size 1 --goal_epoch=700 \
  --id='220512_Raw_Unfold_6' \
  --n_inter=36 --n_command=20 --out_sensory=12 \
  --out_inter=10 --rec_command=10 --in_motor=10 \
  --ode_unfolds=6 --lr=0.0005

# # [220510]
# CUDA_VISIBLE_DEVICES=0 python3 main_ltc_v0.py --mode='train' \
#   --machine='server' --input_type 'raw' \
#   --seq_len 16000 --train_batch_size 6 \
#   --id='220510_Raw_V0'

# # [220511] # 아직 실험 못함, 근데 파라미터가 적은 놈이라 더 해봤자 굳이?
# CUDA_VISIBLE_DEVICES=0 python3 main_ltc_v0.py --mode='train' \
#   --resume_path='/home/leecw/project/log/220510_Raw_V0/epoch=399-val_loss=2.75.ckpt' \
#   --machine='server' --input_type 'raw' \
#   --seq_len 16000 --train_batch_size 6 \
#   --id='220510_Raw_V0'
