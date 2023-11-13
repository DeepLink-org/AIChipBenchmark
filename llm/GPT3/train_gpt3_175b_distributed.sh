#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost  # change to master node addr
MASTER_PORT=6000
NUM_NODES=8
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt

DATASET_1="$DATASET_PATH/wikipedia_20220301_en_text_document"
DATASET_2="$DATASET_PATH/openwebtext_Skylion007_text_document"
DATASET="0.35 ${DATASET_1} 0.65 ${DATASET_2}"
DATASETCACHE="$PATH_TO/gptcache"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

log_interval=10
batch_size=1
global_batch_size=1536

mp_size=8
pp_size=8

## Activation checkpointing saves GPU memory, but reduces training speed
activation_checkpoint="true"
activation_checkpoint="false"


## use distributed optimizer(zero liked method)
distributed_optimizer="true"
distributed_optimizer="false"


GPT_MODEL_ARGS=(
    --num-layers 96 
    --hidden-size 12288 
    --num-attention-heads 96 
    --seq-length 2048 
    --max-position-embeddings 2048
)

TRAINING_ARGS=(
    --micro-batch-size $batch_size
    --global-batch-size $global_batch_size
    --rampup-batch-size 16 16 433870
    --train-samples 10846761
    --lr-decay-samples 9400526 # --lr-decay-iters 430000 
    --lr-warmup-samples 13558 # --lr-warmup-fraction .001 
    --lr 6.0e-5  
    --min-lr 6.0e-6  
    --lr-decay-style cosine  
    --clip-grad 1.0 
    --weight-decay 0.1  
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --fp16
    --use-flash-attn-v2
)

TRAINING_ARGS2=""
if [ "${activation_checkpoint}" = "true" ]; then
TRAINING_ARGS2="${TRAINING_ARGS2} \
    --recompute-activations"
fi

if [ "${distributed_optimizer}" = "true" ]; then
TRAINING_ARGS2="${TRAINING_ARGS2} \
    --use-distributed-optimizer"
fi


MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $mp_size
    --pipeline-model-parallel-size $pp_size
)

DATA_ARGS=(
    --data-path ${DATASET}
    --data-cache-path ${DATASETCACHE}
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 98,2,0
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval $log_interval
    --save-interval 1000
    --eval-interval 100
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 40
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --log-validation-ppl-to-tensorboard
)

torchrun ${DISTRIBUTED_ARGS[@]} ../../pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${TRAINING_ARGS2[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

