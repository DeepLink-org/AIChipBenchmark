# LLaMa-V3 预训练

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
    --tokenizer-type=llama3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
    --output-prefix=arxiv_sample \
    --workers=48
```
参数说明：
- input：输入json文件
- tokenizer-type：huggingface模型名或者本地位置

如需预处理后的数据集，可联系`Deeplink`团队获取。

## 配置文件
- 模型配置文件配置，可以参考 https://github.com/NVIDIA-NeMo/NeMo/tree/main?tab=readme-ov-file 中 ./NeMo/nemo/collections/llm/recipes 文件
- 性能测试：`nemotron_pretraining_llama3_8b.py`(8卡)和`nemotron_pretraining_llama3_70b.py`(32卡)


预训练超参说明：
- batch_size：需要保持`global_batch_size`一致
- seq_length：保持和基准`seq_length`一致
- seed：为了保证精度对齐，保持seed一致
- 并行配置：在不低于基准卡数的前提下，可自行调整


| Model     | #GPUs | global_batch_size  | seq_length| 
|-----------|--------|----|----|
|llama3_8b|8|128|8192|
|llama3_70b|32|64|8192|




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


python ./nemotron_pretraining_llama3_8b.py
# python ./nemotron_pretraining_llama3_70b.py

```

### 性能指标

根据训练日志，采集其中Loss数值和相关性能指标。
```bash
[2025-11-20T14:55:23+08:00] retraining/0 Training epoch 0, iteration 97/99 | lr: 1.469e-05 | global_batch_size: 128 | global_step: 97 | reduced_train_loss: 6.265 | train_step_timing in s: 12.22 | tokens_per_sec_per_gpu: 1.073e+04 | consumed_samples: 12544
[2025-11-20T14:55:35+08:00] retraining/0 Training epoch 0, iteration 98/99 | lr: 1.484e-05 | global_batch_size: 128 | global_step: 98 | reduced_train_loss: 6.181 | train_step_timing in s: 12.21 | tokens_per_sec_per_gpu: 1.074e+04 | consumed_samples: 12672
[2025-11-20T14:55:47+08:00] retraining/0 Training epoch 0, iteration 99/99 | lr: 1.499e-05 | global_batch_size: 128 | global_step: 99 | reduced_train_loss: 6.218 | train_step_timing in s: 12.21 | tokens_per_sec_per_gpu: 1.073e+04 | consumed_samples: 12800
```

性能数据：取去除前后各10个step的`tokens_per_sec_per_gpu`平均值


## 训练目标
根据参考配置训练后，训练到第最后一个step时（即`global_step: 99`），Loss值和基准值loss的差异不超过`5%`。