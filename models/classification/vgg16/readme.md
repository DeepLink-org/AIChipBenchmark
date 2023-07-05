# env
mmcls-v0.23.0
mmcv-v1.5.0

# accu

```
GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev vgg16accu ./configs/vgg/vgg16_8xb32_in1k.py vgg16/accu

```

# perf

```
PUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev vgg16perf ./configs/vgg/vgg16_8xb32_in1k.py vgg16/perf_8


```
