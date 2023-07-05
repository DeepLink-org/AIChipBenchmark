#!/bin/bash

partition=$1
output_json=$2


cwd=`pwd`

function check_dir_exist()
{
    dir=$1
    if [ ! -d $dir ]; then
        echo  "FATAL: $dir does not exist"
        exit 1
    fi
}

function get_repo()
{
    if [ ! $NNUNET_PATH ]; then
        echo "FATAL: NNUNET_PATH is not set"
        exit 1
    fi
    check_dir_exist $NNUNET_PATH
    pushd $NNUNET_PATH
    git checkout a810773
    git apply ${cwd}/patches/nnunet_patch
    popd
}

get_repo

MODEL_DIR=$NNUNET_PATH/PyTorch/Segmentation/nnUNet

echo $MODEL_DIR

pushd $MODEL_DIR


function get_data()
{
    if [ ! -d ${MODEL_DIR}/data/04_3d ]
    then
        echo "data not ready"
        mkdir -p data
        pushd data
        wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C' -O task04.tar
        if [ ! -e task04.tar ]; then
            echo "FATAL: task04.tar does not exist, maybe download failures!"
            exit 1
        fi
        tar -xvf task04.tar
        popd
        python preprocess.py --data ./data --results ./data --task 01 --dim 3
    else
        echo "data exists"
    fi
}

get_data

mkdir -p results

function test_accu()
{
    mkdir -p results/accu
    srun -p ${partition} --cpus-per-task 5 --gres=gpu:8 --exclusive --ntasks-per-node 8 -n 8 --job-name nnunet python main.py --exec_mode train --task 04 --fold 0 --gpus 8 --dim 3 --data `pwd`/data/04_3d --results ./results/accu --epochs 291 --patience 291  > log/accu.log 2>&1

    pushd $cwd
    python nv_nnunet_log_analyse.py --accu --log ${MODEL_DIR}/results/accu/logs.json  --output $output_json
    popd
}

function test_perf()
{
    mkdir -p log
    ngpu=$1
    gpu_per_node=$ngpu
    if [ $ngpu -gt 8 ]; then
        gpu_per_node=8
    fi
    nodes=`expr $ngpu / $gpu_per_node`
    mkdir -p results/perf${ngpu}
    srun -p ${partition} --cpus-per-task 5 --gres=gpu:${gpu_per_node} -n $ngpu --ntasks-per-node ${gpu_per_node} --exclusive --job-name nnunet python scripts/benchmark.py --mode train --gpus ${gpu_per_node} --dim 3 --batch_size 2 --results results/perf${ngpu} --task 04  --nodes $nodes --data `pwd`/data/04_3d > log/perf_$ngpu.log 2>&1

    echo " training finished... parse result ..."
    pushd $cwd
    python nv_nnunet_log_analyse.py --perf --log ${MODEL_DIR}/results/perf$ngpu/perf.json --param ${MODEL_DIR}/results/perf$ngpu/params.json --output $output_json
    popd
}


test_accu
test_perf 1
test_perf 4
test_perf 8
test_perf 16

popd
