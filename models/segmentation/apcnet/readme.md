
# env
mmcv v1.5.0
mmseg v0.23.0


# train

```
GPUS=4 GPUS_PER_NODE=4 SRUN_ARGS="--exclusive --quotatype=auto" sh ./tools/slurm_train.sh caif_dev apcaccu configs/apcnet/apcnet_r50-d8_512x1024_40k_cityscapes.py --work-dir apcnet/accu4gpu

```