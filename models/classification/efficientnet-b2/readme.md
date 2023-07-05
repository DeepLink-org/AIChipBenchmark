# env
mmcls-v0.23.0
mmcv-v1.5.0


# 修改optimizer

mmclassification/configs/efficientnet/efficientnet-b2_8xb32_in1k.py:
```
# optimizer
optimizer = dict(
    type='RMSprop',
    lr=0.01,
    alpha=0.9,
    momentum=0.9,
    eps=1e-2,
    weight_decay=1e-5)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(_delete_=True,
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=16,
    warmup_ratio=0.0001,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=400)

checkpoint_config = dict(interval=100)

#  "accuracy_top-1": 77.02
```

# accu
```
GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev effiaccu ./configs/efficientnet/efficientnet-b2_8xb32_in1k.py effiout/accu
```

# perf

```
GPUS=16 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev effiaccu ./configs/efficientnet/efficientnet-b2_8xb32_in1k.py effiout/perf_16
```

result:

```
1           4           8           16
264.4628099	971.168437	1437.394722	2838.137472

```