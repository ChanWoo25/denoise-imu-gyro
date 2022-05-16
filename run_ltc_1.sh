#! /bin/bash

# [220516 12:00] 비교실험 (https://www.notion.so/LTC_CELL-ode_unfolds-d108546ac2224eb09f50d7258e02dd5d)
# 0.37부터 시작했는데, 학습 결과가 영 이상해서, Learning rate 조정의 영향인가 싶어서 원래 lr로 다시 학습
CUDA_VISIBLE_DEVICES=1 python3 main_ltc_v0.py --mode='train' \
  --resume_path='/home/leecw/project/log/220512_Raw_Unfold_3/epoch=489-val_loss=0.37.ckpt' \
  --machine='server' --input_type 'raw' \
  --seq_len 16000 --train_batch_size 1 --goal_epoch=1000 \
  --id='220512_Raw_Unfold_3' \
  --n_inter=36 --n_command=20 --out_sensory=12 \
  --out_inter=10 --rec_command=10 --in_motor=10 \
  --ode_unfolds=3 --lr=0.001

# # [220515 17:00] 비교실험 (https://www.notion.so/LTC_CELL-ode_unfolds-d108546ac2224eb09f50d7258e02dd5d)
# CUDA_VISIBLE_DEVICES=1 python3 main_ltc_v0.py --mode='train' \
#   --resume_path='/home/leecw/project/log/220512_Raw_Unfold_3/epoch=489-val_loss=0.37.ckpt' \
#   --machine='server' --input_type 'raw' \
#   --seq_len 16000 --train_batch_size 1 --goal_epoch=1000 \
#   --id='220512_Raw_Unfold_3' \
#   --n_inter=36 --n_command=20 --out_sensory=12 \
#   --out_inter=10 --rec_command=10 --in_motor=10 \
#   --ode_unfolds=3 --lr=0.001

# # [220511 00:56 Training]
# CUDA_VISIBLE_DEVICES=0 \
# python3 main_ltc_v0.py --mode='train' \
#   --machine='server' --input_type 'raw' \
#   --seq_len 16000 --train_batch_size 1 \
#   --id='220510_Raw_V3' \
#   --n_inter=64 --n_command=32 --out_sensory=16 \
#   --out_inter=20 --rec_command=20 --in_motor=16

# python3 main_ltc_v0.py --mode='test' --test_path='/root/project/log/lightning_logs/version_85/checkpoints/epoch=399-step=1200.ckpt'\
#   --id='220509_Raw_00' --input_type 'raw' --seq_len 16000 --train_batch_size 2 \
#   --n_inter=32 --n_command=16 --out_sensory=8 \
#   --out_inter=10 --rec_command=10 --in_motor=8
