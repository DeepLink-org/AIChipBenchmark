# set -x

ROOT=/mnt/huangye/workspace/InternVL/internvl_chat
#PARTITION=${PARTITION:-"INTERN2"}
GPUS=${GPUS:-256}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
#QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
#CPUS_PER_TASK=${CPUS_PER_TASK:-10}
#SRUN_ARGS=${SRUN_ARGS:-""}
BATCH_SIZE=${BATCH_SIZE:-2048}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
#export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/internvl_chat_v1_5_internlm2_20b_dynamic_res_pretrain'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# for torch
export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT "

export CMD="\
    $ROOT/internvl/train/internvl_chat_pretrain.py \
    --vision_path "/mnt/huangye/workspace/pretrained/InternViT-6B-448px-V1-5" \
    --llm_path "/mnt/huangye/workspace/model/internlm2-chat-20b" \
    --conv_style "internlm2-chat" \
    --output_dir ${OUTPUT_DIR} \
    --meta_path "/mnt/huangye/workspace/InternVL/internvl_chat/shell/data/internvl_1_5.json" \
    --overwrite_output_dir True \
    --force_image_size 448 \
    --max_dynamic_patch 12 \
    --down_sample_ratio 0.5 \
    --drop_path_rate 0.2 \
    --pad2square False \
    --freeze_llm True \
    --freeze_mlp False \
    --freeze_backbone False \
    --vision_select_layer -1 \
    --use_data_resampling False \
    --dataloader_num_workers 4 \
    --bf16 True \
    --max_steps 500 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0.05 \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --max_seq_length 8192 \
    --do_train True \
    --grad_checkpoint True \
    --group_by_length False \
    --dynamic_image_size True \
    --use_thumbnail True \
    --ps_version 'v2' \
    --deepspeed 'zero_stage3_config.json' \
    --report_to 'tensorboard' \
    "
echo $CMD
bash -c "$LAUNCHER $CMD"  2>&1 | tee -a main_log.txt
echo "END TIME: $(date)"

