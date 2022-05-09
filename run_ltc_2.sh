#! /bin/bash


python3 main_ltc_v0.py \
  --input_type 'raw' --seq_len 16000 --train_batch_size 2 \
  --n_inter=32 --n_command=16 --out_sensory=8 \
  --out_inter=10 --rec_command=10 --in_motor=8
