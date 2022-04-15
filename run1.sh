#! /bin/bash

# Not adjust
python3 main_ver01.py --mode train  -c 32 --v_window 16  --id 220414_c32_window16
python3 main_ver01.py --mode test   -c 32 --v_window 16  --id 220414_c32_window16
python3 main_ver01.py --mode anal   -c 32 --v_window 16  --id 220414_c32_window16

python3 main_ver01.py --mode train  -c 32 --v_window 32  --id 220414_c32_window32
python3 main_ver01.py --mode test   -c 32 --v_window 32  --id 220414_c32_window32
python3 main_ver01.py --mode anal   -c 32 --v_window 32  --id 220414_c32_window32

python3 main_ver01.py --mode train  -c 32 --v_window 64  --id 220414_c32_window64
python3 main_ver01.py --mode test   -c 32 --v_window 64  --id 220414_c32_window64
python3 main_ver01.py --mode anal   -c 32 --v_window 64  --id 220414_c32_window64

python3 main_ver01.py --mode train  -c 32 --v_window 128 --id 220414_c32_window128
python3 main_ver01.py --mode test   -c 32 --v_window 128 --id 220414_c32_window128
python3 main_ver01.py --mode anal   -c 32 --v_window 128 --id 220414_c32_window128

python3 main_ver01.py --mode train  -c 32 --v_window 256 --id 220414_c32_window256
python3 main_ver01.py --mode test   -c 32 --v_window 256 --id 220414_c32_window256
python3 main_ver01.py --mode anal   -c 32 --v_window 256 --id 220414_c32_window256
