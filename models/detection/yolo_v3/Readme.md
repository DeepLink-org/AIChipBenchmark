# env
- mmdetection:
    - v2.24.0
- mmcv:
    - mmcv-full 1.3.17

# 精度测试
```
GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev yolo_v3 ./configs/yolo/yolov3_d53_mstrain-416_273e_coco.py cases/yolov3_d53_mstrain-416_273e_coco_acc_gpus8
```

# 性能测试

只跑一个epoch 每个iter log一下

单机1卡
```
GPUS=1 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev yolo_v3 ./configs/yolo/yolov3_d53_mstrain-416_273e_coco.py cases/yolov3_d53_mstrain-416_273e_coco_gpus1
```

单机4卡
```
GPUS=4 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev yolo_v3 ./configs/yolo/yolov3_d53_mstrain-416_273e_coco.py cases/yolov3_d53_mstrain-416_273e_coco_gpus4
```

单机8卡
```
GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev yolo_v3 ./configs/yolo/yolov3_d53_mstrain-416_273e_coco.py cases/yolov3_d53_mstrain-416_273e_coco_gpus8
```

双机16卡
```
GPUS=16 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev yolo_v3 ./configs/yolo/yolov3_d53_mstrain-416_273e_coco.py cases/yolov3_d53_mstrain-416_273e_coco_gpus16
```

计算每个iter的平均时间
```
srun -p caif_dev --gres=gpu:1 python tools/analysis_tools/analyze_logs.py cal_train_time ./cases/yolov3_d53_mstrain-416_273e_coco_gpus1/20220511_104705.log.json
```

# 测试结果
单卡|单机4卡|单机8卡|双机16卡|精度
:---:|:---:|:---:|:---:|:---:
46.06|164.10|328.04|620.16|epoch=273, box_AP 31.1

