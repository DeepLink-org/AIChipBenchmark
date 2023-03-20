cwd=`pwd`

if [ $# -lt 2 ] ; then
echo "USAGE: $0 slurm_partition output_dir"
exit 1;
fi

partition=$1
dst=$2
dst=$(realpath "$dst")

if [ ! -e $dst ]
then
    mkdir -p $dst
fi


function compile_cuda_op()
{
    pushd $cwd/speed_test/cuda_ops
    srun -p $partition -n 1 -N 1 mkdir -p build && cd build && cmake .. && make
    popd
}

function test_accu()
{
    pushd $cwd/accuracy_test
    result_dir=./result
    mkdir -p $result_dir
    CUBLAS_WORKSPACE_CONFIG=:4096:8 srun -p $partition --ntasks-per-node 1 -N 1 --gres=gpu:1 --cpus-per-task 10 --exclusive python cuda_op_validate.py a100_data $result_dir

    mv $result_dir/cuda_val_result.csv $dst
    popd
}

function test_perf_baseline()
{
    pushd $cwd/speed_test

    # set environment variables to avoid automatic TF32
    export NVIDIA_TF32_OVERRIDE=0

    srun -p $partition --gres=gpu:1 --exclusive python test_conv.py conv_f16.csv 16 0

    srun -p $partition --gres=gpu:1 --exclusive python test_conv.py conv_f32.csv 32 0

    srun -p $partition --gres=gpu:1 --exclusive python test_gemm.py gemm_f16.csv 16 0

    srun -p $partition --gres=gpu:1 --exclusive python test_gemm.py gemm_f32.csv 32 0

    popd
}

function test_perf()
{
    pushd $cwd/speed_test

    # set environment variables to avoid automatic TF32
    export NVIDIA_TF32_OVERRIDE=0

    srun -p $partition --gres=gpu:1 --exclusive python test_conv.py conv_f16.csv 16 1

    cp conv_f16.csv $dst

    srun -p $partition --gres=gpu:1 --exclusive python test_conv.py conv_f32.csv 32 1

    cp conv_f32.csv $dst

    srun -p $partition --gres=gpu:1 --exclusive python test_gemm.py gemm_f16.csv 16 1

    cp gemm_f16.csv $dst

    srun -p $partition --gres=gpu:1 --exclusive python test_gemm.py gemm_f32.csv 32 1

    cp gemm_f32.csv $dst

    popd
}

function test_lt()
{
    pushd $cwd/speed_test/LongTail-Bench

    export PYTHONPATH=./long_tail_bench:$PYTHONPATH

    srun -p $partition --gres=gpu:1 --exclusive python ./long_tail_bench/api/api.py -f ../longtail_perf.csv --outcsv $dst/ltperf.csv --validate

    popd
}

compile_cuda_op

test_accu

# used when generating benchmark values, otherwise please comment out
#test_perf_baseline

test_perf

test_lt
