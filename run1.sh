#! /bin/bash

python3 main_ver01.py --mode train --c0 32 --dv_normed 16 32 --id 220426_c32_vall
python3 main_ver01.py --mode test  --c0 32 --dv_normed 16 32 --id 220426_c32_vall
python3 main_ver01.py --mode anal  --c0 32 --dv_normed 16 32 --id 220426_c32_vall
python3 main_ver01.py --mode train --c0 64 --dv_normed 16 32 --id 220426_c64_vall
python3 main_ver01.py --mode test  --c0 64 --dv_normed 16 32 --id 220426_c64_vall
python3 main_ver01.py --mode anal  --c0 64 --dv_normed 16 32 --id 220426_c64_vall
