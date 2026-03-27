#!/bin/bash
# 批量测试 pretrain 模型（在 rjob 内部运行）
# 每个模型分别测试 FP32 和 FP16，2节点16卡

cd /mnt/shared-storage-user/ailab-sys/chenyuxiao/h200_Benchmark/AIChipBenchmark/models/onedl-mmpretrain
PYTHON=/usr/bin/python3

export MMPRE_PATH=/mnt/shared-storage-user/ailab-sys/chenyuxiao/h200_Benchmark/AIChipBenchmark/models/onedl-mmpretrain
export MMCV_PATH=/mnt/shared-storage-user/ailab-sys/chenyuxiao/h200_Benchmark/AIChipBenchmark/models/onedl-mmcv
export SYSTEM_PACKAGES=/usr/local/lib/python3.10/dist-packages
export PYTHONPATH=$MMPRE_PATH:$MMCV_PATH:$SYSTEM_PACKAGES:$PYTHONPATH

export NCCL_NVLS_ENABLE=0

NGPU=16
mkdir -p work_dirs

# 模型列表: "config_path model_name"
MODELS=(
    "configs/resnet/resnet50_8xb32_in1k.py resnet50"
    "configs/inception_v3/inception-v3_8xb32_in1k.py inception_v3"
    "configs/seresnet/seresnet50_8xb32_in1k.py seresnet50"
    "configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py mobilenet_v2"
    "configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py shufflenet_v2"
    "configs/densenet/densenet121_4xb256_in1k.py densenet121"
    "configs/swin_transformer/swin-large_16xb64_in1k.py swin_large"
    "configs/efficientnet/efficientnet-b2_8xb32_in1k.py efficientnet_b2"
)

TOTAL=${#MODELS[@]}
COUNT=0

for entry in "${MODELS[@]}"; do
    CONFIG=$(echo "$entry" | awk '{print $1}')
    MODEL_NAME=$(echo "$entry" | awk '{print $2}')
    COUNT=$((COUNT + 1))

    for PRECISION in fp32 ; do
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
                ${AMP_OPT}

        echo "[${COUNT}/${TOTAL}] ${MODEL_NAME} ${PRECISION} done."
    done
done

echo ""
echo "========== All pretrain tests finished (16 GPUs) =========="
