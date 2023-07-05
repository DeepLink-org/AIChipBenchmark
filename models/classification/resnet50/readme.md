# env
- mmclassification:
    - v0.23.0
- mmcv:
    - mmcv-full 1.5.0



# 精度测试
```
#16卡时需要增大batchsize:
optimizer = dict(type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0001)

GPUS=16 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev res50 ./configs/resnet/resnet50_8xb32_in1k.py resnet50/accu

```

# 性能测试

双机16卡：

```
更改：只跑一个epoch 每个iter log一下

GPUS=16 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev res50 ./configs/resnet/resnet50_8xb32_in1k.py perfs/resnet50_perf_16gpus

srun -p caif_dev --gres=gpu:1 python tools/analysis_tools/analyze_logs.py cal_train_time perfs/resnet50_perf_16gpus/20220505_092423.log

average iter time: 0.1689 s/iter
```
单机:
```
GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev res50 ./configs/resnet/resnet50_8xb32_in1k.py perfs/resnet50_perf_8gpus

GPUS=4 GPUS_PER_NODE=4 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev res50 ./configs/resnet/resnet50_8xb32_in1k.py perfs/resnet50_perf_4gpus

GPUS=1 GPUS_PER_NODE=1 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev res50 ./configs/resnet/resnet50_8xb32_in1k.py perfs/resnet50_perf_1gpus


单卡	单机4卡	单机8卡	双机16卡	训练参数	精度指标
282.4360106	1044.897959	1548.699335	3031.379515	"epoch=100	top-1: 76.5
```

