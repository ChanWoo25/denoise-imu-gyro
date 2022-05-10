#! /bin/bash
CUDA_VISIBLE_DEVICES=3 python3 main_ltc_v0.py --mode='train' \
  --machine='server' --input_type 'raw' \
  --seq_len 16000 --train_batch_size 3 \
  --id='220510_Raw_V2' \
  --n_inter=48 --n_command=24 --out_sensory=12 \
  --out_inter=15 --rec_command=15 --in_motor=12
