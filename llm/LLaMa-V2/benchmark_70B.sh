#!/bin/bash
set +x
partitiion=$1

srun -p $partitiion -N 4 -n 32 --ntasks-per-node=8 --gpus-per-task=1 python benchmark.py \
    -c 70b -g -b 2 \
    --max_length 4096 -p gemini -x --tp 8 --pp 1 --zero 2 --mbs 1\
    > train_70B_benchmark.log 2>&1 &
