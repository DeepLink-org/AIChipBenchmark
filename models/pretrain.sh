#!/bin/bash
cd ./onedl-mmpretrain

export MMSEG_PATH=./onedl-mmsegmentation
export MMPRE_PATH=./onedl-mmpretrain
export MMDET_PATH=./onedl-mmdetection
export MMCV_PATH=./onedl-mmcv

mkdir -p work_dirs

CONFIG=configs/resnet/resnet50_8xb32_in1k.py
BASE_WORK_DIR=work_dirs/resnet50

# ============ 单卡测试 ============
echo "========== Testing 1 GPU =========="
python tools/train.py $CONFIG \
    --work-dir ${BASE_WORK_DIR}_gpus1 \
    # --cfg-options \
    #     optim_wrapper.type=AmpOptimWrapper

# # ============ 4卡测试 ============
# echo "========== Testing 4 GPUs =========="
# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     tools/train.py $CONFIG \
#     --launcher pytorch \
#     --work-dir ${BASE_WORK_DIR}_gpus4 \
#     # --cfg-options \
#     #     optim_wrapper.type=AmpOptimWrapper \

# ============ 8卡测试 (单机) ============
# echo "========== Testing 8 GPUs (single node) =========="
# python -m torch.distributed.launch \
#     --nproc_per_node=8 \
#     tools/train.py $CONFIG \
#     --launcher pytorch \
#     --work-dir ${BASE_WORK_DIR}_gpus8 \
#     --cfg-options \
#         optim_wrapper.type=AmpOptimWrapper \

# # ============ 16卡测试 (双机) ============
# echo "========== Testing 16 GPUs (2 nodes x 8 GPUs) =========="
# python -m torch.distributed.launch \
#     --nnodes=$NODE_COUNT \
#     --node-rank=$NODE_RANK \
#     --master-addr=$MASTER_ADDR \
#     --nproc-per-node=8 \
#     --master-port=29500 \
#     tools/train.py $CONFIG \
#     --launcher pytorch \
#     --work-dir ${BASE_WORK_DIR}_gpus16 \
#     # --cfg-options \
#     #     optim_wrapper.type=AmpOptimWrapper \


# # ============ FP16 测试示例 ============
# echo "========== Testing 8 GPUs with FP16 =========="
# python -m torch.distributed.launch \
#     --nproc_per_node=8 \
#     tools/train.py $CONFIG \
#     --launcher pytorch \
#     --work-dir ${BASE_WORK_DIR}_gpus8_fp16 \
#     --cfg-options \
#         optim_wrapper.type=AmpOptimWrapper \
#         optim_wrapper.loss_scale=512.0 \
#         optim_wrapper.optimizer.lr=0.1
