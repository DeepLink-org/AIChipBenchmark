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
 iteration        5/      10 | consumed samples:         7680 | consumed tokens:     15728640 | elapsed time per iteration (ms): 344656.1 | learning rate: 2.270E-05 | global batch size:  1536 | lm loss: 1.100078E+01 | loss scale: 4294967296.0 | grad norm: 0.000 | actual seqlen:  2048 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 4.457 | tokens per gpu per second (tgs): 142.612 | TFLOPs: 153.38 |
 iteration        6/      10 | consumed samples:         9216 | consumed tokens:     18874368 | elapsed time per iteration (ms): 347319.2 | learning rate: 1.392E-05 | global batch size:  1536 | lm loss: 1.100214E+01 | loss scale: 4294967296.0 | grad norm: 0.000 | actual seqlen:  2048 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 4.422 | tokens per gpu per second (tgs): 141.518 | TFLOPs: 152.20 |
 iteration        7/      10 | consumed samples:        10752 | consumed tokens:     22020096 | elapsed time per iteration (ms): 345333.7 | learning rate: 8.059E-06 | global batch size:  1536 | lm loss: 1.100195E+01 | loss scale: 4294967296.0 | grad norm: 0.000 | actual seqlen:  2048 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 4.448 | tokens per gpu per second (tgs): 142.332 | TFLOPs: 153.08 |
 iteration        8/      10 | consumed samples:        12288 | consumed tokens:     25165824 | elapsed time per iteration (ms): 346430.5 | learning rate: 6.000E-06 | global batch size:  1536 | lm loss: 1.100105E+01 | loss scale: 4294967296.0 | grad norm: 0.000 | actual seqlen:  2048 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 4.434 | tokens per gpu per second (tgs): 141.881 | TFLOPs: 152.59 |
 iteration        9/      10 | consumed samples:        13824 | consumed tokens:     28311552 | elapsed time per iteration (ms): 347071.6 | learning rate: 6.000E-06 | global batch size:  1536 | lm loss: 1.100164E+01 | loss scale: 4294967296.0 | grad norm: 0.000 | actual seqlen:  2048 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 4.426 | tokens per gpu per second (tgs): 141.619 | TFLOPs: 152.31 |
 iteration       10/      10 | consumed samples:        15360 | consumed tokens:     31457280 | elapsed time per iteration (ms): 344835.7 | learning rate: 6.000E-06 | global batch size:  1536 | lm loss: 1.099972E+01 | loss scale: 4294967296.0 | grad norm: 0.000 | actual seqlen:  2048 | number of skipped iterations:   0 | number of nan iterations:   0 | samples per second: 4.454 | tokens per gpu per second (tgs): 142.537 | TFLOPs: 153.30 |

```

## 训练目标
根据参考配置训练后，训练到第最后一个step时（即`global_step: 999`），Loss值和基准值loss的差异不超过`5%`。

