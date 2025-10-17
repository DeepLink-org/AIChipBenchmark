# Qwen-LLM 预训练

## 准备工作

拉取NeMo开源镜像：nvcr.io/nvidia/nemo:25.09.00 。
- 数据集：
    - 预训练：使用[arxiv_sample.jsonl](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample/resolve/main/arxiv_sample.jsonl)

## 数据集预处理

```
python scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input=arxiv_sample.jsonl \
    --json-keys=text \
    --tokenizer-library=huggingface \
    --tokenizer-type=models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    --output-prefix=arxiv_sample \
    --workers=48
```
参数说明：
- input：输入json文件
- tokenizer-type：huggingface模型名或者本地位置

如需预处理后的数据集，可联系`Deeplink`团队获取。

## 配置文件
- 模型配置文件配置，可以参考 https://github.com/NVIDIA-NeMo/NeMo/tree/main?tab=readme-ov-file 中 ./NeMo/nemo/collections/llm/recipes 文件
- 性能测试：`nemotron_pretraining_qwen3_8b.py`(8卡)、`nemotron_pretraining_qwen3_30b_a3b.py`(8卡)和`nemotron_pretraining_qwen25_72b.py`(32卡)

Qwen3-8B模型部分超参设置如下：
```
...
tensor_parallelism: int = 4,  
pipeline_parallelism: int = 1,
pipeline_parallelism_type: Optional[torch.dtype] = None,
num_nodes: int = 1,
num_gpus_per_node: int = 8,
global_batch_size=32,
micro_batch_size=2,
seq_length=4096,
...
```
Qwen3-30B-A3B模型部分超参设置如下：
```
...
tensor_parallelism: int = 4, 
pipeline_parallelism: int = 2,
pipeline_parallelism_type: Optional[torch.dtype] = None,
num_nodes: int = 1,
num_gpus_per_node: int = 8,
global_batch_size=32,
micro_batch_size=2,
seq_length=4096,
...
```
Qwen2.5-72B模型部分超参设置如下：
```
...
tensor_parallelism: int = 8,  
pipeline_parallelism: int = 4,
pipeline_parallelism_type: Optional[torch.dtype] = torch.bfloat16,
num_nodes: int = 4,
num_gpus_per_node: int = 8,
global_batch_size=32,
micro_batch_size=2,
seq_length=4096,
...
```
## 启动及数据采集

启动训练脚本：
```Python
set -ex

export MASTER_PORT=29500
export GPUS_PER_NODE=8
export NNODES=${NODE_COUNT}
export NODE_RANK=${NODE_RANK}
export MASTER_ADDR=${MASTER_ADDR}
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))


python ./nemotron_pretraining_qwen3_8b.py
# python ./nemotron_pretraining_qwen3_30b_a3b.py
# python ./nemotron_pretraining_qwen25_72b.py

```

### 性能指标

根据训练日志，采集其中Loss数值和相关性能指标。
```bash
2025-10-17T11:32:59+08:00] Training epoch 0, iteration 527/999 | lr: 0.0001758 | global_batch_size: 32 | global_step: 527 | reduced_train_loss: 2.792 | train_step_timing in s: 2.389 | tokens_per_sec_per_gpu: 6.859e+03 | consumed_samples: 16896 | val_loss: 3.624
[2025-10-17T11:33:02+08:00] Training epoch 0, iteration 528/999 | lr: 0.0001754 | global_batch_size: 32 | global_step: 528 | reduced_train_loss: 3.06 | train_step_timing in s: 2.443 | tokens_per_sec_per_gpu: 6.706e+03 | consumed_samples: 16928 | val_loss: 3.624
[2025-10-17T11:33:04+08:00] Training epoch 0, iteration 529/999 | lr: 0.0001749 | global_batch_size: 32 | global_step: 529 | reduced_train_loss: 3.221 | train_step_timing in s: 2.127 | tokens_per_sec_per_gpu: 7.702e+03 | consumed_samples: 16960 | val_loss: 3.624
```

## 训练目标
根据参考配置训练后，训练到第最后一个step时（即`global_step: 999`），Loss值和基准值loss的差异不超过`5%`。