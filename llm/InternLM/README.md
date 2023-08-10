# InternLM 预训练(微调)

## 准备工作

- 代码下载：https://github.com/InternLM/InternLM
- 安装：参考 https://github.com/InternLM/InternLM/blob/main/doc/install.md, 需根据厂商环境进行适配
- 数据集：
  - 预训练：使用内置dummy数据集
  - 微调：使用Alpaca数据集，数据集处理参考：https://github.com/InternLM/InternLM/blob/main/doc/usage.md


## 配置

InternLM-7B参考配置：https://github.com/InternLM/InternLM/blob/main/configs/7B_sft.py

InternLM-7B 模型部分超参设置如下
```python
SEQ_LEN = 2048
HIDDEN_SIZE = 4096
NUM_ATTENTION_HEAD = 32
MLP_RATIO = 8 / 3
NUM_LAYER = 32
VOCAB_SIZE = 103168
```
InternLM-13B 模型部分超参设置如下
```python
SEQ_LEN = 2048
HIDDEN_SIZE = 5120
NUM_ATTENTION_HEAD = 40
MLP_RATIO = 8 / 3
NUM_LAYER = 40
VOCAB_SIZE = 103168
```
### 并行配置

训练并行配置样例如下，厂商可根据芯片显存大小调整并行配置以避免OOM，比如设置zero1=1 tensor=8：
```python
parallel = dict(
    zero1=8,
    pipeline=1,
    tensor=1,
)
```
- zero1：zero 并行策略，分如下三种情况，默认值为 -1
  - 当`size <= 0`，则 zero1 进程组的大小等于数据并行进程组的大小，因此优化器状态参数将在数据并行范围内分配
  - 当`size == 1`，则不使用 zero1 ，所有数据并行组保留完整的优化器状态参数
  - 当`size > 1`且`size <= data_parallel_world_size`，则 zero1 进程组是数据并行进程组的子集
- pipeline：流水线并行大小，目前只支持 1，默认值为 1
- tensor：张量并行大小，通常是每个节点的 GPU 数量，默认值为 1

注意：`数据并行大小 = 总的 GPU 数目 / 流水线并行大小 / 张量并行大小`

参考：https://github.com/InternLM/InternLM/blob/main/doc/usage.md


## 启动及数据采集

若在 slurm 上启动分布式运行环境，多节点 32 卡的运行命令如下所示：

```bash
srun -p internllm -N 4 -n 32 --ntasks-per-node=8 --gpus-per-task=1 python train.py --config ./configs/7B_sft.py
```

单节点 8 卡的运行命令如下所示：
```bash
srun -p internllm -N 1 -n 8 --ntasks-per-node=8 --gpus-per-task=1 python train.py --config ./configs/7B_sft.py
```

若在 torch 上启动分布式运行环境，单节点 8 卡的运行命令如下所示：
```bash
torchrun --nnodes=1 --nproc_per_node=8 train.py --config ./configs/7B_sft.py --launcher "torch"
```

### 性能指标
根据训练日志，采集其中性能指标TGS、TFlops、Loss数值
```bash
2023-07-27 13:11:56,488 INFO train.py:317 in record_current_batch_training_metrics -- tflops=54.33760564166562,step=0,loss=11.577922821044922,tgs (tokens/gpu/second)=491.55,lr=9.779
754323328192e-05,loss_scale=65536.0,grad_norm=77.8995810422784,micro_num=1,num_consumed_tokens=262144,inf_nan_skip_batches=0,num_samples_in_batch=8,largest_length=2048,largest_batch
=8,smallest_batch=8,adam_beta2=0.95,fwd_bwd_time=13.78
```
### 稳定性指标

对于稳定性指标，记录前100个step的Loss。



## 训练目标
训练step > 100000或者训练时间大于72小时，Loss小于 xxx。