# env
mmcls-v0.23.0
mmcv-v1.5.0

# accu

```
8卡schdule配置为 imagenet_bs2048

GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev densenet ./configs/densenet/densenet121_4xb256_in1k.py denseout/accu

```

# perf

```
GPUS=1 GPUS_PER_NODE=1 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev densenet ./configs/densenet/densenet121_4xb256_in1k.py denseout/perf_1
```

ips:
```
1           4           8           16
282.6855124	1072.974971	1584.158416	3179.630492
```