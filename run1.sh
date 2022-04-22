#! /bin/bash

python3 main_ver01.py --mode train --c0 16 --dv_normed 16 32 --id 220422_c16_v16_v32
python3 main_ver01.py --mode test  --c0 16 --dv_normed 16 32 --id 220422_c16_v16_v32
python3 main_ver01.py --mode anal  --c0 16 --dv_normed 16 32 --id 220422_c16_v16_v32
python3 main_ver01.py --mode train --c0 32 --dv_normed 16 32 --id 220422_c32_v16_v32
python3 main_ver01.py --mode test  --c0 32 --dv_normed 16 32 --id 220422_c32_v16_v32
python3 main_ver01.py --mode anal  --c0 32 --dv_normed 16 32 --id 220422_c32_v16_v32
