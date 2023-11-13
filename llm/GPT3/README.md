# GPT3 预训练

## 准备工作

- 代码下载：git clone https://github.com/microsoft/Megatron-DeepSpeed.git (b7529f3505a6248e0ef58af9be80eaa4da4af0ee)
- 环境依赖：`torch==1.13.1+cu11`, `flash-attn==v2.0.4`, `apex==0.1`, `deepspeed==0.12.2`等。
- 数据集：
    - wikipedia https://huggingface.co/datasets/wikipedia (20220301.en)
    - openwebtext https://huggingface.co/datasets/Skylion007/openwebtext

## 代码适配修改
1. 性能数据需要替换`megatron/training.py`为本目录下相应文件。相关的PR，可以参考：https://github.com/microsoft/Megatron-DeepSpeed/pull/286
2. 增加gpt3训练脚本`train_gpt3_175b_distributed_tgs.sh`和`train_gpt3_175b_distributed.sh` 到`examples/gpt3`（需要新建相关目录）下。



## 数据下载和预处理

### 数据下载

wikipedia数据集（~20Gb）
```Python
from datasets import load_dataset                   
wikipedia = load_dataset("wikipedia", "20220301.en", split="train")
wikipedia.to_json("$DATASET_PATH/wikipedia.20220301.en.json")

```
openwebtext数据集（~40Gb）
```Python
from datasets import load_dataset                   
openwebtext = load_dataset("Skylion007/openwebtext", split="train")
openwebtext.to_json("$DATASET_PATH/openwebtext.Skylion007.json") 
```
数据集保存目录：`$DATASET_PATH`。

### 数据预处理

以wikipedia数据集为例，处理命令如下：
```bash
python tools/preprocess_data.py \
       --input $DATASET_PATH/wikipedia.20220301.en.json \
       --output-prefix $DATASET_PATH/wikipedia_20220301_en \
       --vocab-file gpt2-vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod \
       --workers 8
```
其中[vocab-file](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json)和[merge-file](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt)可自行下载。


## 启动及数据采集

启动训练脚本：
```bash
cd examples/gpt3

bash train_gpt3_175b_distributed.sh \
    $CHECKPOINT_PATH \
    $TENSORBOARD_LOGS_PATH  \
    $VOCAB_FILE \
    $MERGE_FILE 
```

脚本中需要适配的配置如下：

```bash
...

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost  # change to master node addr
MASTER_PORT=6000
NUM_NODES=8
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt

DATASET_1="$DATASET_PATH/wikipedia_20220301_en_text_document"
DATASET_2="$DATASET_PATH/openwebtext_Skylion007_text_document"
DATASET="0.35 ${DATASET_1} 0.65 ${DATASET_2}"
DATASETCACHE="$PATH_TO/gptcache"

...
```

需要指定的配置说明如下：

- CHECKPOINT_PATH：表示模型检查点位置
- TENSORBOARD_LOGS_PATH：tensorboard日志位置
- VOCAB_FILE ：gpt2-vocab.json的位置
- MERGE_FILE ：gpt2-merges.txt的位置
- MASTER_ADDR ：分布式训练的master节点地址
- DATASETCACHE ：数据cache位置


预训练超参说明：
- batch_size：global_batch_size=1536，micro-batch-size=1，不要修改
- fp16：开启混合精度
- flash-attn：出于性能考虑，训练使用flash attenstion v2
- 并行配置：64卡下，模型并行tp=8，流水线并行pp=8


### 性能指标
启动性能数据脚本：
```bash
cd examples/gpt3

bash train_gpt3_175b_distributed_tgs.sh \
    $CHECKPOINT_PATH \
    $TENSORBOARD_LOGS_PATH  \
    $VOCAB_FILE \
    $MERGE_FILE 
```
参数同训练脚本，缩小了训练迭代数，在固定`batchsize=1536`的情形下，获取前10个iter的性能。



根据训练日志，采集其中Loss数值和相关性能指标。
```bash

待补充

```

## 训练目标
根据参考配置训练后，训练Loss小于 ** （待补充），且ppl指标小于 ** （待补充）。
