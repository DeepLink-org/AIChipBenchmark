# env
- mmdetection:
    - v2.24.0
- mmcv:
    - mmcv-full 1.3.17

# 精度测试
```
GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev ssd_300 ./configs/ssd/ssd300_coco.py cases/ssd300_coco_acc_gpus8
```

# 性能测试

只跑一个epoch 每个iter log一下

单机1卡
```
GPUS=1 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev ssd_300 ./configs/ssd/ssd300_coco.py cases/ssd300_coco_acc_gpus8_gpus1
```

单机4卡
```
GPUS=4 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev ssd_300 ./configs/ssd/ssd300_coco.py cases/ssd300_coco_acc_gpus8_gpus4
```

单机8卡
```
GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev ssd_300 ./configs/ssd/ssd300_coco.py cases/ssd300_coco_acc_gpus8_gpus8
```

双机16卡
```
GPUS=16 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev ssd_300 ./configs/ssd/ssd300_coco.py cases/ssd300_coco_acc_gpus8_gpus16
```

计算每个iter的平均时间
```
srun -p caif_dev --gres=gpu:1 python tools/analysis_tools/analyze_logs.py cal_train_time ./cases/ssd300_coco_acc_gpus8_gpus1/20220510_180632.log.json
```

# 测试结果
单卡|单机4卡|单机8卡|双机16卡|精度
:---:|:---:|:---:|:---:|:---:
65.63|255.39|491.17|851.63|epoch=24，单机8卡, box_AP 25.6
