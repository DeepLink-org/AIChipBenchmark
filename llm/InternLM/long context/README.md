# InternLM-7B 32K 预训练

## 准备工作

- 代码下载：https://github.com/InternLM/InternEvo.git（81fb73506b53d56d03dfe85e5e63d027525a3dbb）
- 安装：参考 https://github.com/InternLM/InternEvo/blob/develop/doc/install.md, 需根据厂商环境进行适配
- 数据集：使用内置的dummy数据集，替换`internlm/data/build_dataloader.py`

## 配置

InternLM-7B参考配置：`7B_internlm2_long.py`

InternLM-7B 模型部分超参设置如下
```python
VOCAB_SIZE = 92544
SEQ_LEN = 32768
HIDDEN_SIZE = 4096
NUM_ATTENTION_HEAD = 32
NUM_KV_ATTENTION_HEAD = 8
MLP_RATIO = 3.5
NUM_LAYER = 32
```

### 并行配置

训练并行配置样例如下，厂商可根据芯片显存大小调整并行配置以避免OOM：

```python
SP = 8
weight = 1
zero1 = 8

parallel = dict(
    zero1=dict(size=zero1),
    tensor=dict(size=SP, mode="isp"),
    pipeline=dict(size=1, interleaved_overlap=True),
    weight=dict(size=weight, overlap=True, memory_pool=True),
)
```
- zero1：zero 并行策略，分如下三种情况，默认值为 -1
  - 当`size <= 0`，则 zero1 进程组的大小等于数据并行进程组的大小，因此优化器状态参数将在数据并行范围内分配
  - 当`size == 1`，则不使用 zero1 ，所有数据并行组保留完整的优化器状态参数
  - 当`size > 1`且`size <= data_parallel_world_size`，则 zero1 进程组是数据并行进程组的子集
- pipeline：流水线并行大小，目前只支持 1，默认值为 1
- tensor：mode="isp"下，序列并行大小
- weight：weight parallel，在mode="isp"下可以开启

注意：

1. 数据并行大小：`数据并行大小 = 总的 GPU 数目 / 流水线并行大小 / 张量并行大小`

2. 调增并行策略，需要保持 `global_batch_size`不变。 其中`global_batch_size=micro_bsz * micro_num * 数据并行大小`。`micro_bsz`和`micro_num`可以在配置中进行修改。

参考：https://github.com/InternLM/InternEvo/blob/develop/doc/usage.md


## 启动及数据采集

若在 slurm 上启动分布式运行环境，多节点 32 卡的运行命令如下所示：

```bash
srun -p Intern5 -N 4 -n 32  --ntasks-per-node=8 --gpus-per-task=1 --quotatype=spot python train.py --config ./configs/7B_internlm2_long.py
```


若在 torch 上启动分布式运行环境，单节点 8 卡的运行命令如下所示：
```bash
torchrun --nnodes=4 --nproc_per_node=8 train.py --config ./configs/7B_sft.py --launcher "torch"
```


### batch size说明
- InternLM-7B 32k 预训练配置功能和性能指标，采用统一的`batchsize=32`。（可以参考`7B_internlm2_long.py`）

说明：
1. 由于InternLM框架并未显式给出`global batchsize`的配置，而是通过`global_batch_size = micro_bsz * micro_num * 数据并行大小`来计算，这一点在进行多卡迁移时需要注意（关注`数据并行大小`项变化）
2. 如果需要计算global batchsize中的tokens，可以用`batchsize` * `sequence length`
3. 如果因为厂商硬件限制，可以减小相应的`batchsize`，但是不可以任意增大`batchsize`。
4. 如果因为厂商硬件限制，可以进行`micro_bsz`和`micro_num`的调整，只需要保持两者乘积和推荐配置保持一致即可。


### 性能指标
根据训练日志，采集其中性能指标TGS、TFlops、Loss数值
```bash
2024-08-23 19:25:04,187 INFO pipeline.py:663 in record_current_batch_training_metrics -- tflops=252.44732125406708 step=199 loss=0.00042401382233947515 real_tgs=2055.59 tgs (tokens/gpu/second)=2511.37 tgs/last_tgs_1=2511.37 tgs/tgs_all=2481.83 tgs/tgs_avg=2508.74 tgs/tgs_SMA=2510.99 tgs/last_tgs_10=2510.67 tgs/last_tgs_50=2510.99 lr=1e-05 loss_scale=65536.0 grad_norm={'0_default': 0.00021934305195273646, '1_fp32': 0.0} micro_num=8 num_consumed_tokens=209715200 inf_nan_skip_batches=0 num_samples_in_batch=8 largest_length=32768 largest_batch=1 smallest_batch=1 adam_beta2=0.95 fwd_bwd_time=12.87 bwd_time=8.69 acc=1.0 perplexity=1.0006 acc/en=1.0 acc/cn=0.0 acc/code=0.0 tokens/en=858249 tokens/cn=0 tokens/code=0 loss_from_metric=0.0006 loss/en=0.0006 loss/cn=nan loss/code=nan 
```


## 训练目标
训练step > 200，Loss小于  `0.0005`。