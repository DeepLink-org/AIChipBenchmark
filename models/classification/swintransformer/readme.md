# env
- mmclassification:
    - v0.23.0
- mmcv:
    - mmcv-full 1.5.0




# 精度测试
```
双机16卡，默认配置

GPUS=16 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive --quotatype=auto"  sh ./tools/slurm_train.sh caif_dev swin_trans_acc ./configs/swin_transformer/swin-large_16xb64_in1k.py trainout/swin_trans_acc

```

# 测试结果
Acc top 1: 78.724%, Acc top5: 93.434%