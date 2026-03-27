#!/bin/bash
# 批量测试 segmentation 模型（在 rjob 内部运行）
# 每个模型分别测试 FP32 和 FP16，2节点16卡

cd ./models/onedl-mmsegmentation
PYTHON=/usr/bin/python3

export MMSEG_PATH=/mnt/shared-storage-user/ailab-sys/chenyuxiao/h200_Benchmark/AIChipBenchmark/models/onedl-mmsegmentation
export MMCV_PATH=/mnt/shared-storage-user/ailab-sys/chenyuxiao/h200_Benchmark/AIChipBenchmark/models/onedl-mmcv
export SYSTEM_PACKAGES=/usr/local/lib/python3.10/dist-packages
export PYTHONPATH=$MMSEG_PATH:$MMCV_PATH:$SYSTEM_PACKAGES:$PYTHONPATH

export NCCL_NVLS_ENABLE=0

NGPU=16
WEIGHT_PATH=./models/weight/resnet50_v1c-2cccc1ad.pth
mkdir -p work_dirs

# 模型列表: "config_path model_name"
# 4个模型全部使用 ResNetV1c-50 backbone，共用同一个权重
MODELS=(
    "configs/deeplabv3/deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024.py deeplabv3"
    "configs/fcn/fcn_r50-d8_4xb2-40k_cityscapes-512x1024.py fcn"
    "configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet"
    "configs/apcnet/apcnet_r50-d8_4xb2-40k_cityscapes-512x1024.py apcnet"
)

TOTAL=${#MODELS[@]}
COUNT=0

for entry in "${MODELS[@]}"; do
    CONFIG=$(echo "$entry" | awk '{print $1}')
    MODEL_NAME=$(echo "$entry" | awk '{print $2}')
    COUNT=$((COUNT + 1))

    for PRECISION in fp32 fp16; do
        if [ "$PRECISION" == "fp16" ]; then
            AMP_OPT="optim_wrapper.type=AmpOptimWrapper"
        else
            AMP_OPT="optim_wrapper.type=OptimWrapper"
        fi

        echo ""
        echo "============================================================"
        echo "  [${COUNT}/${TOTAL}] ${MODEL_NAME} - ${PRECISION} - ${NGPU} GPUs (2 nodes)"
        echo "============================================================"

        $PYTHON -m torch.distributed.launch \
            --nnodes=$NODE_COUNT \
            --node-rank=$NODE_RANK \
            --master-addr=$MASTER_ADDR \
            --nproc_per_node=8 \
            --master-port=29500 \
            tools/train.py ${CONFIG} \
            --launcher pytorch \
            --work-dir work_dirs/${MODEL_NAME}_gpus${NGPU}_${PRECISION} \
            --cfg-options \
                model.backbone.init_cfg.type=Pretrained \
                model.backbone.init_cfg.checkpoint=${WEIGHT_PATH} \
                model.pretrained=None \
                ${AMP_OPT}

        echo "[${COUNT}/${TOTAL}] ${MODEL_NAME} ${PRECISION} done."
    done
done

echo ""
echo "========== All segmentation tests finished (16 GPUs) =========="
