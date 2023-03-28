# env
mmcls-v0.23.0
mmcv-v1.5.0

# accu

```
16卡时lr=0.2

GPUS=16 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev se50accu ./configs/seresnet/seresnet50_8xb32_in1k.py se50/accu_16
```