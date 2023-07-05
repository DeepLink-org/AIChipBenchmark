# env
- mmclassification:
    - v0.23.0
- mmcv:
    - mmcv-full 1.5.0




# 精度测试
```
单机8卡，默认配置

GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive --quotatype=auto"  sh ./tools/slurm_train.sh caif_dev mobilenet_acc ./configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet.py trainout/mobilenet_acc

```

# 测试结果
Acc top1: 71.948%, Acc top5: 90.364%