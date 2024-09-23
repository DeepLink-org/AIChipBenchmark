# EasyLLM

## 准备工作

- 代码库：https://github.com/ModelTC/EasyLLM.git
  - 版本：dev
- 安装: 参考https://github.com/ModelTC/EasyLLM/blob/dev/README.md
- 镜像：
  nvcr.io/nvidia/pytorch:23.12-py3，如需开启context parallel，该镜像是最低所需版本，包含必要的transformer engine，并支持flash attn2
- 依赖：hjson== 3.1.0，peft== 0.12.0，py-cpuinfo==9.0.0，regex== 2024.7.24，sentencepiece== 0.2.0
  tensorboardx== 2.6.2.2, tiktoken== 0.7.0, tokenizers== 0.19.1, tqdm== 4.66.5, safetensors==0.4.3, huggingface-hub==0.23.1
- 数据集：
  - 微调：使用Alpaca数据集：[alpaca_all](alpaca_all.json)

## 配置文件

### 模型配置

模型配置文件参考：https://github.com/InternLM/InternEvo/blob/v0.2.3dev20240201/configs/7B_sft.py

Internlm2-7B 模型部分超参设置如下
```python
SEQ_LEN = 32768
num_layers: 32
hidden_size: 4096
num_attention_heads: 32
intermediate_size: 14336
num_kv_attention_heads: 8
```

Internlm2-20B 模型部分超参设置如下
```python
SEQ_LEN = 32768
num_layers: 48
hidden_size: 6144
num_attention_heads: 48
intermediate_size: 16384
num_kv_attention_heads: 8
```

Llama3-8B 模型部分超参设置如下
```python
SEQ_LEN = 32768
hidden_size: 4096
num_attention_heads: 32
intermediate_size: 14336
num_kv_attention_heads: 8
```

### 并行策略

Internlm2-7B、Internlm2-20B、Llama3-8B的训练并行策略如下，Internlm2-20B、Llama3-8B使用4机32卡，Internlm2-7B使用单机8卡，
厂商可根据芯片显存大小调整并行配置以避免OOM，比如设置zero1=1 tensor_pipeline=8：

```python
runtime:
  seed: &seed 42
  tensor_model_parallel_size: &tp 4
  pipeline_model_parallel_size: 4
  context_parallel_size: 2
  deepspeed: True
```

- pipeline_model_parallel_size：流水线并行大小，默认值为 1
- tensor_model_parallel_size：张量并行大小，通常是每个节点的 GPU 数量，默认值为 1
- context_parallel_size：序列大小，默认值为 1，

注意：

1. 并行大小：`总的 GPU 数目 = 序列并行大小 x 流水线并行大小 x 张量并行大小`
2. 张量并行大小：如需开启序列并行，张量并行值必须同时大于1

## 启动命令

若在 slurm 上启动分布式运行环境，4机 32 卡的运行命令如下所示：

```bash
Internlm2-8b
srun -p internllm -N 4 -n 32 --ntasks-per-node=8 --gpus-per-task=1 bash mg_train_a800.sh internlm2-7b

Internlm2-20b
srun -p internllm -N 4 -n 32 --ntasks-per-node=8 --gpus-per-task=1 bash mg_train_a800.sh internlm2-20b

Llama3-8b
srun -p internllm -N 4 -n 32 --ntasks-per-node=8 --gpus-per-task=1 bash mg_train_a800.sh llama_8b_sft
```
