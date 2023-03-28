#!/bin/bash

#SBATCH --job-name=model_test        # name
#SBATCH --time 72:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --cpus-per-task 5

set -x

mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")

isperf=$1
output_json=$2

py_file="perf.py"
if [ $isperf == 0 ]
then
    py_file="main.py"
fi

echo "srun --exclusive python -u ${py_file} --config configs/inception.yaml ${output_json}"

srun --exclusive python -u ${py_file} --config configs/inception.yaml --output ${output_json} \
2>&1 | tee log/train_inceptionv3.log-$now