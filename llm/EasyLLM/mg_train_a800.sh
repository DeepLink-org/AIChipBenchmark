set -x -e -o pipefail

export ACCELERATOR_BACKEND=CUDA
export NCCL_DEBUG=WARN

ROOT=/mnt/kongbohua/workspace/EasyLLM
export PYTHONPATH=$ROOT/DeepSpeed:$PYTHONPATH
export PYTHONPATH=/mnt/wangxing/transformers/src:$PYTHONPATH
export PYTHONPATH=$ROOT/llm/models:$ROOT:$ROOT/llm/utils/tools:$PYTHONPATH

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-36589}
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

export ModelConfig=${1:-demo}
mkdir -p logs/${ModelConfig}
logfile=logs/${ModelConfig}/`date +%Y-%m-%d-%H-%M`_node_${RANK}-${WORLD_SIZE}.log

echo "---Model Config Start---"  >> $logfile
cat configs/megatron/${ModelConfig}.yaml >> $logfile
echo "---Model Config End---"  >> $logfile

DISTRIBUTED_ARGS="python -m torch.distributed.launch  --nnodes=$WORLD_SIZE  --node_rank=$RANK  --master_addr=$MASTER_ADDR  --nproc_per_node=$GPUS_PER_NODE  --master_port=$MASTER_PORT"

export LAUNCHER="$DISTRIBUTED_ARGS"

export CMD=" \
    $ROOT/llm/runners/base_llm_runner.py \
    --config configs/megatron/${ModelConfig}.yaml \
    --launcher torch 2>&1 | tee -a $logfile
    "

set -x
bash -c "$LAUNCHER $CMD"