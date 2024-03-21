# InternLM2 预训练

## 准备工作

- 代码库：https://github.com/InternLM/InternEvo
  - 版本：v0.2.3dev20240201
- 安装：https://github.com/InternLM/InternEvo/blob/v0.2.3dev20240201/doc/install.md#%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85
- 数据集：
  - 性能测试：使用dummy
  - 精度测试：使用RedPajama-Data-1T-Sample中的book_sample.jsonl
  - 数据集准备可参考https://github.com/InternLM/InternEvo/blob/develop/doc/en/usage.md#dataset-preparation-pre-training
- bug修复
  - 在迭代数据集时有bug
  - 具体bug和修复方法参考https://github.com/InternLM/InternEvo/issues/77

## 配置文件

### 模型配置

模型配置文件参考：https://github.com/InternLM/InternEvo/blob/v0.2.3dev20240201/configs/7B_sft.py

InternLM2-7B 模型部分超参设置如下
```python
SEQ_LEN = 4096
HIDDEN_SIZE = 4096
NUM_ATTENTION_HEAD = 32
MLP_RATIO = 8 / 3
NUM_LAYER = 32
VOCAB_SIZE = 103168

micro_num=4
micro_bsz=2
```

InternLM-70B 模型部分超参设置如下
```python
SEQ_LEN = 4096
HIDDEN_SIZE = 8192
NUM_ATTENTION_HEAD = 64
MLP_RATIO = 8/3
NUM_LAYER = 80
NUM_KV_ATTENTION_HEAD = 8
VOCAB_SIZE = 103168

micro_num=32
micro_bsz=1
```

### 并行策略

7B和70B的训练并行策略分别如下，其中，7B使用单机8卡，70B使用4机32卡，厂商可根据芯片显存大小调整并行配置以避免OOM，比如设置zero1=1 tensor=8：

```python
parallel = dict(
    zero1=dict(size=4, fsdp=False),
    tensor=dict(size=2, fsdp=False, mode='mtp'),
    pipeline=dict(size=1, interleaved_overlap=False),
    sequence_parallel=False,
)
```

```python
parallel = dict(
    zero1=dict(size=-1),
    tensor=dict(size=4, mode="mtp"),
    pipeline=dict(size=8, interleaved_overlap=True),
    weight=dict(size=1),
)
```
- zero1：zero 并行策略，分如下三种情况，默认值为 -1
  - 当`size <= 0`，则 zero1 进程组的大小等于数据并行进程组的大小，因此优化器状态参数将在数据并行范围内分配
  - 当`size == 1`，则不使用 zero1 ，所有数据并行组保留完整的优化器状态参数
  - 当`size > 1`且`size <= data_parallel_world_size`，则 zero1 进程组是数据并行进程组的子集
- pipeline：流水线并行大小，目前只支持 1，默认值为 1
- tensor：张量并行大小，通常是每个节点的 GPU 数量，默认值为 1

注意：

1. 数据并行大小：`数据并行大小 = 总的 GPU 数目 / 流水线并行大小 / 张量并行大小`

2. 调增并行策略，需要保持 `global_batch_size`不变。 其中`global_batch_size=micro_bsz * micro_num * 数据并行大小`。`micro_bsz`和`micro_num`可以在配置中进行修改。

参考：https://github.com/InternLM/InternEvo/blob/v0.2.3dev20240201/doc/usage.md


## 启动命令

若在 slurm 上启动分布式运行环境，4机 32 卡的运行命令如下所示：

```bash
srun -p internllm -N 4 -n 32 --ntasks-per-node=8 --gpus-per-task=1 python train.py --config ./configs/7B_sft.py
```

单机 8 卡的运行命令如下所示：
```bash
srun -p internllm -N 1 -n 8 --ntasks-per-node=8 --gpus-per-task=1 python train.py --config ./configs/7B_sft.py
```

若在 torch 上启动分布式运行环境，单机 8 卡的运行命令如下所示：
```bash
torchrun --nnodes=1 --nproc_per_node=8 train.py --config ./configs/7B_sft.py --launcher "torch"
```

4机32卡参考：
```bash
torchrun --nnodes=4 --nproc_per_node=8 --node-rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT  train.py --config ../70b_internlm2.py  --launcher torch
```

### 性能指标
根据训练日志，采集其中性能指标TGS、TFlops、Loss数值
```bash
2024-03-21 14:48:31,587      INFO training_internlm.py:599 in record_current_batch_training_metrics -- tflops=186.62241336587493 step=1 loss=11.63248062133789 tgs (tokens/gpu/second)=3917.95 tgs/last_tgs_1=3917.97 tgs/tgs_all=3262.85 tgs/tgs_avg=3356.7 tgs/tgs_SMA=3262.85 tgs/last_tgs_10=0 tgs/last_tgs_50=0 lr=6.000000000000001e-07 loss_scale=65536.0 grad_norm={'0_default': 23.297677981215344, '1_fp32': 0.0} micro_num=4 num_consumed_tokens=262144 inf_nan_skip_batches=0 num_samples_in_batch=8 largest_length=4096 largest_batch=2 smallest_batch=2 adam_beta2=0.95 fwd_bwd_time=4.03 acc=0.0 perplexity=111859.625 acc/en=0.0 tokens/en=131072 loss_from_metric=11.6199 loss/en=11.6199
```

## 训练目标

7B训练step > 20000，Loss的中位数大约是 0.005。
