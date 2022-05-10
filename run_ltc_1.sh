#! /bin/bash


python3 main_ltc_v0.py --mode='train' \
  --input_type 'window' --seq_len 6400 --train_batch_size 6 \
  --id='220509_Window_V0'
  # \
  # --n_inter=32 --n_command=16 --out_sensory=8 \
  # --out_inter=10 --rec_command=10 --in_motor=8


# python3 main_ltc_v0.py --mode='test' --test_path='/root/project/log/lightning_logs/version_85/checkpoints/epoch=399-step=1200.ckpt'\
#   --id='220509_Raw_00' --input_type 'raw' --seq_len 16000 --train_batch_size 2 \
#   --n_inter=32 --n_command=16 --out_sensory=8 \
#   --out_inter=10 --rec_command=10 --in_motor=8
