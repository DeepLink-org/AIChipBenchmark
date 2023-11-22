# LLaMa2 预训练

## 准备工作

- 代码下载：git clone https://github.com/hpcaitech/ColossalAI.git (fd6482ad8caad818256a9e1f1aaa0af49a1aecca)
- 环境依赖：`torch==1.13.1+cu117`, `flash-attn==2.0.5`, `apex==0.1`, `transformers>=4.31.0`等，详见req_env.txt供参考
- 数据集：
    -  RedPajama-Data-1T-Sample https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample (约10G)
    
## 代码适配修改
1. 性能数据需要替换`megatron/training.py`为本目录下相应文件。相关的PR，可以参考：https://github.com/microsoft/Megatron-DeepSpeed/pull/286
2. 增加gpt3训练脚本`train_gpt3_175b_distributed_tgs.sh`和`train_gpt3_175b_distributed.sh` 到`examples/gpt3`（需要新建相关目录）下。



## 数据下载和预处理

### 数据下载

RedPajama-Data-1T-Sample数据集（~10Gb）
```Python
from datasets import load_dataset
dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")
dataset.save_to_disk("RedPajama-Data-1T-Sample")
#dataset = load_from_disk("data")

```

## 启动训练及数据采集

按需启动对应的训练脚本，示例如下：
```bash
cd ColossalAI/examples/language/llama2/scripts/
bash benchmark_70B.sh
```

预训练超参说明：
- batch_size：batch_size=2，mbs=1，不要修改
- fp16：开启混合精度
- max_length=4096
- 并行配置：32卡下，张量并行tp=8，zero=2


### 性能指标

参数参考benchmark训练脚本，获取前5个step的性能，根据训练日志，采集吞吐量性能指标。

### 预训练指标
参数参考预训练脚本，训练完成后，根据训练日志，采集其中Loss数值。


## 训练目标
根据参考配置训练后，训练Loss小于 ** （待补充），且ppl指标小于 ** （待补充）。
