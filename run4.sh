#! /bin/bash

# python3 main_ver01.py --mode train --c0 16 --dv_normed 32 128 512 --id 220419_c16_v32_128_512
python3 main_ver01.py --mode test  --c0 16 --dv_normed 32 128 512 --id 220419_c16_v32_128_512
python3 main_ver01.py --mode anal  --c0 16 --dv_normed 32 128 512 --id 220419_c16_v32_128_512

python3 main_ver01.py --mode train --c0 32 --dv_normed 32 128 512 --id 220419_c32_v32_128_512
python3 main_ver01.py --mode test  --c0 32 --dv_normed 32 128 512 --id 220419_c32_v32_128_512
python3 main_ver01.py --mode anal  --c0 32 --dv_normed 32 128 512 --id 220419_c32_v32_128_512

python3 main_ver01.py --mode train --c0 64 --dv_normed 32 128 512 --id 220419_c64_v32_128_512
python3 main_ver01.py --mode test  --c0 64 --dv_normed 32 128 512 --id 220419_c64_v32_128_512
python3 main_ver01.py --mode anal  --c0 64 --dv_normed 32 128 512 --id 220419_c64_v32_128_512
