#!/bin/bash
ngpu=$1
bin=$2
num=$3

set -x

echo "$ngpu - $nnode"


export MV2_USE_CUDA=1
export LOCAL_RANK=$MV2_COMM_WORLD_LOCAL_RANK

## -m set the minimum and maximum message length to be used in a benchmark. (bytes) min:max: to test the latency for a given data size, set min=max
## -M set per process maximum memory consumption (bytes)
## -f report additional statistics of the benchmark, such as min and max latencies and the number of iterations.
## -i can be used to set the number of iterations to run for each message length.
## -d TYPE accelerator device buffers can be of TYPE `cuda' or `openacc'


#mpirun -np $ngpu $bin/osu_nccl_allgather -m $3:$3  -M 8589934592 -i 1000 -d cuda
mpirun -np $ngpu $bin/osu_nccl_allreduce -m $3:$3  -M 8589934592 -i 1000 -d cuda

