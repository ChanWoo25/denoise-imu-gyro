#! /bin/bash

# python3 main_ver01.py --mode anal --id 220419_c32_v256_a
# python3 main_ver01.py --mode anal --id 220419_c32_v256_a
# python3 main_ver01.py --mode anal --id 220419_c32_v256_a
# python3 main_ver01.py --mode anal --id 220419_c32_v256_a
# python3 main_ver01.py --mode anal --id 220419_c32_v256_a
# python3 main_ver01.py --mode anal --id 220419_c32_v256_a
# python3 main_ver01.py --mode anal --id 220419_c32_v256_a
# python3 main_ver01.py --mode anal --id 220419_c32_v256_a
# python3 main_ver01.py --mode anal --id 220419_c32_v256_a
# python3 main_ver01.py --mode anal --id 220419_c32_v256_a
# python3 main_ver01.py --mode anal --id 220419_c32_v256_a
# python3 main_ver01.py --mode anal --id 220419_c32_v256_a
# python3 main_ver01.py --mode anal --id 220419_c32_v256_a
# python3 main_ver01.py --mode anal --id 220419_c32_v256_a
# python3 main_ver01.py --mode anal --id 220419_c32_v256_a
# python3 main_ver01.py --mode anal --id 220419_c32_v256_a
# python3 main_ver01.py --mode anal --id 220419_c32_v256_a

search_dir=/root/denoise/results
cd $search_dir

ids=$(find . -maxdepth 1 -name '2204*' -type d -not -path '*/\.*' | sed 's/^\.\///g' | sort)

cd ..
for id in $ids
do
  python3 main_ver01.py --mode anal --id $id
done
