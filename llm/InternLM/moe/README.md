# InternLM-7B MoE4 预训练

## 准备工作

- 代码下载：https://github.com/blankde/InternEvo/tree/test/new_optim_for_moe（c7129f94c1d45e02d3887f60dc7b56f896e57f5f）
- 安装：参考 https://github.com/blankde/InternEvo/blob/test/new_optim_for_moe/doc/install.md, 需根据厂商环境进行适配
- 数据集：使用内置的dummy数据集

## 配置

InternLM-7B参考配置：`7B_MoE4.py`

InternLM-7B 模型部分超参设置如下
```python
SEQ_LEN = 2048
HIDDEN_SIZE = 4096
NUM_ATTENTION_HEAD = 32
MLP_RATIO = 4 / 3
NUM_LAYER = 32
VOCAB_SIZE = 103168
```

### 并行配置

训练并行配置样例如下，厂商可根据芯片显存大小调整并行配置以避免OOM：

```python
    zero1=dict(size=-1, fsdp=False),
    tensor=dict(size=1, mode="mtp"),
    pipeline=dict(size=1, interleaved_overlap=True),
    weight=dict(size=1, overlap=True, memory_pool=True),
```
- zero1：zero 并行策略，分如下三种情况，默认值为 -1
  - 当`size <= 0`，则 zero1 进程组的大小等于数据并行进程组的大小，因此优化器状态参数将在数据并行范围内分配
  - 当`size == 1`，则不使用 zero1 ，所有数据并行组保留完整的优化器状态参数
  - 当`size > 1`且`size <= data_parallel_world_size`，则 zero1 进程组是数据并行进程组的子集
- pipeline：流水线并行大小，默认值为 1
- tensor：mode="mtp"下，张量并行大小
- weight：weight parallel，在tensor mode="isp"下可以开启

注意：

1. 数据并行大小：`数据并行大小 = 总的 GPU 数目 / 流水线并行大小 / 张量并行大小`

2. 调增并行策略，需要保持 `global_batch_size`不变。 其中`global_batch_size=micro_bsz * micro_num * 数据并行大小`。`micro_bsz`和`micro_num`可以在配置中进行修改。

参考：https://github.com/blankde/InternEvo/blob/test/new_optim_for_moe/doc/usage.md

### MoE配置
```python
model = dict(
   ...
    num_experts=4,
    moe_use_residual=False,
    moe_type="GShard",  # Support: "GShard", "MegaBlock", "MegaBlock-D"
   ...
)
moe = dict(
    top_k=2,
    capacity_factor=1.0,
    eval_capacity_factor=1.0,
    min_capacity=4,
    noisy_gate_policy=None,
    drop_tokens=True,
    use_rts=True,
    use_fused_gating=True,
    enable_token_rearrange_opt = True,
    use_tutel = True,
)
```

- num_experts: 专家个数为4
- moe_type： GShard
- top_k：top-K router
- drop_tokens：开启drop
- capacity_factor: 每个专家的capacity factor
- use_fused_gating： gating fused优化
- enable_token_rearrange_opt: token重排优化
- tutel：moe加速库优化

注意：

1. moe可根据实际支持度，调整优化策略；
2. moe模型相关其他参数，需要保持一致，不要更改。

## 启动及数据采集

若在 slurm 上启动分布式运行环境，多节点 32 卡的运行命令如下所示：

```bash
srun -p Intern5 -N 4 -n 32  --ntasks-per-node=8 --gpus-per-task=1 --quotatype=spot python train.py --config ./configs/7B_MoE4.py
```


若在 torch 上启动分布式运行环境，单节点 8 卡的运行命令如下所示：
```bash
torchrun --nnodes=4 --nproc_per_node=8 train.py --config ./configs/7B_sft.py --launcher "torch"
```


### batch size说明
- InternLM-7B MoE4 预训练配置功能和性能指标，采用统一的`batchsize=512`。（可以参考`7B_MoE4.py`）

说明：
1. 由于InternLM框架并未显式给出`global batchsize`的配置，而是通过`global_batch_size = micro_bsz * micro_num * 数据并行大小`来计算，这一点在进行多卡迁移时需要注意（关注`数据并行大小`项变化）
2. 如果需要计算global batchsize中的tokens，可以用`batchsize` * `sequence length`
3. 如果因为厂商硬件限制，可以减小相应的`batchsize`，但是不可以任意增大`batchsize`。
4. 如果因为厂商硬件限制，可以进行`micro_bsz`和`micro_num`的调整，只需要保持两者乘积和推荐配置保持一致即可。


### 性能指标
根据训练日志，采集其中性能指标TGS、TFlops、Loss数值
```bash
2024-08-28 10:56:01,469	INFO pipeline.py:681 in record_current_batch_training_metrics -- tflops=105.89925496973669 step=49 loss=0.32518792152404785 real_tgs=2956.7 tgs (tokens/gpu/second)=3359.06 tgs/last_tgs_1=3359.06 tgs/tgs_all=3181.84 tgs/tgs_avg=3356.98 tgs/tgs_SMA=3343.04 tgs/last_tgs_10=3362.74 tgs/last_tgs_50=3343.04 lr=8.756803171472813e-05 loss_scale=65536.0 grad_norm={'0_default': 1.4511364692486448, '1_fp32': 5.752340752537247, '2_moe_ep_size_4': 0.13802250075382713} moe_loss=3.296875 micro_num=8 num_consumed_tokens=52428800 inf_nan_skip_batches=0 num_samples_in_batch=16 largest_length=2048 largest_batch=2 smallest_batch=2 adam_beta2=0.95 fwd_bwd_time=8.99 bwd_time=5.91 acc=0.9322 perplexity=1.3148 acc/en=0.9322 acc/cn=0.0 acc/code=0.0 tokens/en=922958 tokens/cn=0 tokens/code=0 loss_from_metric=0.2737 loss/en=0.2737 loss/cn=nan loss/code=nan 
```


## 训练目标
训练step > 50，Loss小于  `0.325`。