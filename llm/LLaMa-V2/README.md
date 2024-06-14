# LLAMA2 预训练

## 准备工作

- 代码库：https://github.com/InternLM/InternEvo 
  - 性能测试参考版本：v0.2.3dev20240201
  - 精度测试参考版本：v0.3.1dev20240229
- 安装参考：https://github.com/InternLM/InternEvo/blob/v0.2.3dev20240201/doc/install.md#%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85
- 数据集：
  - 性能测试：使用dummy
  - 精度测试：使用RedPajama-Data-1T-Sample https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample (约10G)
  - 数据集准备可参考https://github.com/InternLM/InternEvo/blob/develop/doc/en/usage.md#dataset-preparation-pre-training
- bug修复
  - 在迭代数据集时有bug
  - 具体bug和修复方法参考 https://github.com/InternLM/InternEvo/issues/77

## 配置文件

### 模型配置

模型配置文件配置，可以参考config中文件。
  - 性能测试：使用`7B_llama2.py`（32卡）和`70B_llama2.py`（128卡）
  - 精度测试：使用`train_7B_withdata.py`（8卡）和`train_70B_withdata.py`（64卡）

LLAMA2-7B 模型部分超参设置如下
```python
model_type="LLAMA2"
SEQ_LEN = 4096
HIDDEN_SIZE = 4096
NUM_ATTENTION_HEAD = 32
MLP_RATIO = 3.5
NUM_LAYER = 32
NUM_KV_ATTENTION_HEAD = 8
VOCAB_SIZE = 32000
```

LLAMA2-70B 模型部分超参设置如下
```python
SEQ_LEN = 4096
HIDDEN_SIZE = 8192
NUM_ATTENTION_HEAD = 64
MLP_RATIO = 3.5
NUM_LAYER = 80
NUM_KV_ATTENTION_HEAD = 8
VOCAB_SIZE = 32000
```

### 并行策略

7B和70B的训练并行策略分别如下，其中，功能测试中7B使用单机8卡，70B使用4机32卡，厂商可根据芯片显存大小调整并行配置以避免OOM，比如设置zero1=1 tensor=8：

```python
parallel = dict(
    zero1=dict(size=-1),
    tensor=dict(size=1, mode="mtp"),
    pipeline=dict(size=1, interleaved_overlap=True),
    weight=dict(size=1, overlap=True, memory_pool=True),
)
```

```python
parallel = dict(
    zero1=dict(size=-1),
    tensor=dict(size=8, mode="mtp"),
    pipeline=dict(size=4, interleaved_overlap=True),
    weight=dict(size=1, overlap=True, memory_pool=True),
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

2. 性能测试时调增并行策略，需要保持 `global_batch_size`不变。 其中`global_batch_size=micro_bsz * micro_num * 数据并行大小`。`micro_bsz`和`micro_num`可以在配置中进行修改。


参考：https://github.com/InternLM/InternEvo/blob/v0.2.3dev20240201/doc/usage.md


## 启动命令

4机32卡参考命令：
```bash
torchrun --nnodes=4 --nproc_per_node=8 --node-rank $RANK --master-addr $MASTER_ADDR --master-port $MASTER_PORT  train.py --config ./config/7B_llama2.py  --launcher torch
```

### batch size说明

- LLAMA2-7B 预训练功能配置参考`train_7B_withdata.py`（64卡参考配置），功能和性能指标，采用统一的`batchsize=256`。

- LLAMA2-7B 预训练性能配置参考`7B_llama2.py`（32卡参考配置），功能和性能指标，采用统一的`batchsize=256`。

- LLAMA2-70B 预训练功能配置参考`train_70B_withdata.py`（64卡参考配置），功能和性能指标，采用统一的`batchsize=48`。

- LLAMA2-70B 预训练性能配置参考`70B_llama2.py`（32卡参考配置），功能和性能指标，采用统一的`batchsize=1024`。


说明：

1. 由于InternLM框架并未显式给出`global batchsize`的配置，而是通过`global_batch_size = micro_bsz * micro_num * 数据并行大小`来计算，这一点在进行多卡迁移时需要注意（关注`数据并行大小`项变化）
2. 如果需要计算global batchsize中的tokens，可以用`batchsize` * `sequence length`
3. 如果因为厂商硬件限制，可以减小相应的`batchsize`，但是不可以任意增大`batchsize`。
4. 如果因为厂商硬件限制，可以进行`micro_bsz`和`micro_num`的调整，只需要保持两者乘积和推荐配置保持一致即可。


### 性能指标
根据训练日志，采集其中性能指标TGS、TFlops、Loss数值
```bash
2024-03-27 17:03:42,047 INFO training_internlm.py:599 in record_current_batch_training_metrics -- tflops=180.4572540886346 step=19 loss=0.005358175374567509 tgs (tokens/gpu/second)=3345.72 tgs/last_tgs_1=3345.74 tgs/tgs_all=2958.36 tgs/tgs_avg=3149.53 tgs/tgs_SMA=2958.36 tgs/last_tgs_10=3339.47 tgs/last_tgs_50=0 lr=1e-05 loss_scale=65536.0 grad_norm={'0_default': 0.01974448980463596, '1_fp32': 0.0} micro_num=4 num_consumed_tokens=10485760 inf_nan_skip_batches=0 num_samples_in_batch=12 largest_length=2672 largest_batch=4 smallest_batch=2 adam_beta2=0.95 fwd_bwd_time=4.13 acc=0.9992 perplexity=1.0075 acc/en=0.9992 acc/cn=0.0 acc/code=0.0 tokens/en=489165 tokens/cn=0 tokens/code=0 loss_from_metric=0.0075 loss/en=0.0075 loss/cn=nan loss/code=nan 
```

## 训练目标
1. `llama2-7B`模型根据参考配置训练后，训练Loss小于`1.716`
2. `llama2-70B`模型根据参考配置训练后，训练Loss小于`2.617`
