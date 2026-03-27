#!/bin/bash
# 批量测试 detection 模型（在 rjob 内部运行）
# 每个模型分别测试 FP32 和 FP16，2节点16卡

cd /mnt/shared-storage-user/ailab-sys/chenyuxiao/h200_Benchmark/AIChipBenchmark/models/onedl-mmdetection
PYTHON=/usr/bin/python3

export MMDET_PATH=/mnt/shared-storage-user/ailab-sys/chenyuxiao/h200_Benchmark/AIChipBenchmark/models/onedl-mmdetection
export MMCV_PATH=/mnt/shared-storage-user/ailab-sys/chenyuxiao/h200_Benchmark/AIChipBenchmark/models/onedl-mmcv
export SYSTEM_PACKAGES=/usr/local/lib/python3.10/dist-packages
export PYTHONPATH=$MMDET_PATH:$MMCV_PATH:$SYSTEM_PACKAGES:$PYTHONPATH

export NCCL_NVLS_ENABLE=0

NGPU=16
WEIGHT_DIR=/mnt/shared-storage-user/ailab-sys/chenyuxiao/h200_Benchmark/AIChipBenchmark/models/weight
mkdir -p work_dirs

# 权重映射
RESNET50_W=${WEIGHT_DIR}/resnet50-0676ba61.pth
RESNET50_CAFFE_W=${WEIGHT_DIR}/resnet50_msra-5891d200.pth
DARKNET53_W=${WEIGHT_DIR}/darknet53-a628ea1b.pth
VGG16_W=${WEIGHT_DIR}/vgg16_caffe-292e1171.pth
RESNET18_W=${WEIGHT_DIR}/resnet18-f37072fd.pth
SWIN_TINY_W=${WEIGHT_DIR}/swin_tiny_patch4_window7_224.pth

# 模型列表: "config_path model_name weight_path"
MODELS=(
    "configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py faster_rcnn ${RESNET50_W}"
    "configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py mask_rcnn ${RESNET50_W}"
    "configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py cascade_rcnn ${RESNET50_W}"
    "configs/retinanet/retinanet_r50_fpn_1x_coco.py retinanet ${RESNET50_W}"
    "configs/yolo/yolov3_d53_8xb8-320-273e_coco.py yolov3 ${DARKNET53_W}"
    "configs/fcos/fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco.py fcos ${RESNET50_CAFFE_W}"
    "configs/ssd/ssd300_coco.py ssd300 ${VGG16_W}"
    "configs/centernet/centernet_r18_8xb16-crop512-140e_coco.py centernet ${RESNET18_W}"
    "configs/solo/decoupled-solo_r50_fpn_1x_coco.py solo ${RESNET50_W}"
    "configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py swin_mask_rcnn ${SWIN_TINY_W}"
)

TOTAL=${#MODELS[@]}
COUNT=0

for entry in "${MODELS[@]}"; do
    CONFIG=$(echo "$entry" | awk '{print $1}')
    MODEL_NAME=$(echo "$entry" | awk '{print $2}')
    WEIGHT=$(echo "$entry" | awk '{print $3}')
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
                model.backbone.init_cfg.checkpoint=${WEIGHT} \
                ${AMP_OPT}

        echo "[${COUNT}/${TOTAL}] ${MODEL_NAME} ${PRECISION} done."
    done
done

echo ""
echo "========== All detection tests finished (16 GPUs) =========="
