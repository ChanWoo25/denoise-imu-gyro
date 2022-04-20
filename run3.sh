#! /bin/bash

python3 main_ver01.py --mode train --c0 32 --dv_normed 256 --id 220419_c32_v256_a
python3 main_ver01.py --mode test  --c0 32 --dv_normed 256 --id 220419_c32_v256_a
python3 main_ver01.py --mode anal  --c0 32 --dv_normed 256 --id 220419_c32_v256_a
# python3 main_ver01.py --mode train --c0 32 --dv_normed 256 --id 220419_c32_v256_b
# python3 main_ver01.py --mode test  --c0 32 --dv_normed 256 --id 220419_c32_v256_b
# python3 main_ver01.py --mode anal  --c0 32 --dv_normed 256 --id 220419_c32_v256_b
# python3 main_ver01.py --mode train --c0 32 --dv_normed 256 --id 220419_c32_v256_c
# python3 main_ver01.py --mode test  --c0 32 --dv_normed 256 --id 220419_c32_v256_c
# python3 main_ver01.py --mode anal  --c0 32 --dv_normed 256 --id 220419_c32_v256_c
