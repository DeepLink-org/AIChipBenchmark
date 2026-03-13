#!/bin/bash

# 多机 NCCL Tests 版本
# 用法: sh test_all.sh <nccl_test_binary> <output_file> <comm_type>
# 示例: sh test_all.sh ./nccl-tests-2.17.9/build/all_reduce_perf_mpi result.json all-reduce

bin=$1      # path to nccl-tests binary (e.g., all_reduce_perf_mpi)
output=$2   # output json file
commtype=$3 # all-reduce or all-gather

mkdir -p logs

# 设置环境变量
export LD_LIBRARY_PATH=/usr/local/openmpi/4.1.8/lib:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export NCCL_DEBUG=""

# 测试配置
HOSTFILE="hostfile.txt"
TOTAL_GPUS=32          # 总 GPU 数 (根据 hostfile 配置)
MAX_GPU_PER_NODE=8     # 每节点最大 GPU 数

# 基础消息大小（每个 rank 的 bytes，对应 osu 的 300000 floats * 4）
BASE_SIZES=(1200000 12390400 26214400 67108864 536870800)

function run_one()
{
    ngpu=$1
    base_bytes=$2      # 每个 rank 的基础 bytes
    actual_bytes=$3    # 实际传递给 nccl 的总 bytes

    echo "Testing ngpu=$ngpu, base_bytes_per_rank=$base_bytes, actual_bytes=$actual_bytes"

    outfile="logs/comm_${ngpu}_${base_bytes}.out"

    # 计算迭代次数
    if [ $actual_bytes -lt 1048576 ]; then
        iter=1000
    elif [ $actual_bytes -lt 16777216 ]; then
        iter=500
    else
        iter=100
    fi

    # 选择对应的测试二进制文件
    if [ "$commtype" == "all-gather" ]; then
        if [[ "$bin" == *"all_gather_perf_mpi"* ]]; then
            test_bin="$bin"
        else
            test_bin="${bin%/all_reduce_perf_mpi}/all_gather_perf_mpi"
        fi
        echo "Running all-gather with $ngpu GPUs, total message size $actual_bytes bytes ($base_bytes per rank)"
    elif [ "$commtype" == "all-reduce" ]; then
        if [[ "$bin" == *"all_reduce_perf_mpi"* ]]; then
            test_bin="$bin"
        else
            test_bin="${bin%/all_gather_perf_mpi}/all_reduce_perf_mpi"
        fi
        echo "Running all-reduce with $ngpu GPUs, message size $actual_bytes bytes"
    else
        echo "Not Supported Comm Ops: $commtype"
        return 1
    fi

    /usr/local/openmpi/4.1.8/bin/mpirun \
        --hostfile $HOSTFILE \
        -np $ngpu \
        --allow-run-as-root \
        --mca pml ucx \
        --mca btl ^openib \
        -x LD_LIBRARY_PATH \
        -x NCCL_SOCKET_IFNAME=bond0 \
        -x NCCL_NVLS_ENABLE=0 \
        -x NCCL_IB_HCA="=mlx5_0,mlx5_1" \
        -x NCCL_IB_GID_INDEX=3 \
        $test_bin \
        -b $actual_bytes \
        -e $actual_bytes \
        -n $iter \
        -w 10 \
        -f 1 \
        -g 1 > $outfile 2>&1

    echo "Result saved to $outfile"
}

function test_iter()
{
    for ngpu in 32
    do
        if [ $ngpu -gt $TOTAL_GPUS ]; then
            echo "Skipping ngpu=$ngpu (exceeds total GPUs=$TOTAL_GPUS)"
            continue
        fi

        for base_bytes in "${BASE_SIZES[@]}"
        do
            # 根据通信类型计算实际字节数
            if [ "$commtype" == "all-gather" ]; then
                # all_gather: 总大小 = 每个 rank 大小 * ngpu
                # 这样每个 rank 的 count 保持 300000, 3097600 等不变
                actual_bytes=$((base_bytes * ngpu))
            else
                # all_reduce: 每个 rank 大小不变
                actual_bytes=$base_bytes
            fi
            
            run_one $ngpu $base_bytes $actual_bytes
        done
    done

    python3 parse_nccl_result.py logs $output $commtype
}

# 检查参数
if [ -z "$bin" ] || [ -z "$output" ] || [ -z "$commtype" ]; then
    echo "Usage: sh test_all.sh <nccl_test_binary> <output_file> <comm_type>"
    echo "Example: sh test_all.sh ./nccl-tests-2.17.9/build/all_reduce_perf_mpi result.json all-reduce"
    exit 1
fi

if [ ! -f "$bin" ]; then
    echo "Error: $bin not found"
    exit 1
fi

if [ ! -f "$HOSTFILE" ]; then
    echo "Error: $HOSTFILE not found"
    exit 1
fi

echo "Starting NCCL communication benchmark..."
echo "Binary: $bin"
echo "Output file: $output"
echo "Comm type: $commtype"
echo "Total GPUs: $TOTAL_GPUS"

test_iter

echo "Benchmark completed!"