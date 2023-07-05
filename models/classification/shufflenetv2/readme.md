# env
- mmclassification:
    - v0.23.0
- mmcv:
    - mmcv-full 1.5.0




# 精度测试
```
双机16卡，默认配置

GPUS=16 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive --quotatype=auto"  sh ./tools/slurm_train.sh caif_dev shufflenet_acc ./configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py trainout/shufflenet_acc

```

# 测试结果
Acc top1: 69.78%, Acc top5: 88.898%