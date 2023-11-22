#!/bin/bash
set +x
partition=$1

srun -p $partition --gres=gpu:8 colossalai run --nproc_per_node 8 benchmark.py \
    -g -x -b 2 --max_length 4096  --tp 1 --pp 1 --zero 0 \
    > train_7B_benchmark.log 2>&1 &
