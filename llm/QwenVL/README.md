# Qwen2.5VL-7B 预训练

## 准备工作

- 代码下载：git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git (5da76224eddcfe1b31dbc343543ac0831f31cb9c)
- 环境依赖：拉取官方镜像源：dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:25.01。升级 `tokenizers==0.21.4`、 `multi-storage-client==0.26.0` 及以上。

## 数据下载和预处理

### 数据下载

LLaVA-Pretrain 数据集（~27Gb）
```Python
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
cd LLaVA-Pretrain
unzip images.zip
```
### 数据预处理

处理命令如下：
```bash
#convert to webdataset format:
cd /workspace/Pai-Megatron-Patch/toolkits/multimodal_data_preprocessing
python convert_llava_pretrain_to_wds.py /mnt/llava-datasets/LLaVA-Pretrain/

#convert to megatron-energon format:
cd /mnt/llava-datasets/LLaVA-Pretrain/wds
energon prepare ./

#select the following values for the presented options:
> Please enter a desired train/val/test split like "0.5, 0.2, 0.3" or "8,1,1": 9,1,0
> Do you want to create a dataset.yaml interactively? [Y/n]: Y
> Please enter a number to choose a class: 10 (VQAWebdataset)
> Do you want to set a simple field_map[Y] (or write your own sample_loader [n])? [Y/n]: Y
> Please enter a webdataset field name for 'image' (<class 'torch.Tensor'>): jpg
> Please enter a webdataset field name for 'context' (<class 'str'>): json[0][value]
> Please enter a webdataset field name for 'answers' (typing.Optional[typing.List[str]], default: None): json[1][value]
> Please enter a webdataset field name for 'answer_weights' (typing.Optional[torch.Tensor], default: None):
```
或者直接使用官方处理好的数据集文件
```bash
cd /mnt/llava-datasets/LLaVA-Pretrain/
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/vlm-datasets/wds.tgz
tar -zxf wds.tgz
```
## 模型下载和预处理

模型使用 Huggingface 的 Qwen/Qwen2.5-VL-7B-Instruct ：https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

### 模型预处理
Megatron-Core模型格式转换
```bash
cd ./Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/qwen2_5_vl/run_8xH20.sh \
7B \
./models--Qwen--Qwen2.5-VL-7B-Instruct \
./Qwen2.5VL-7B-Instruct-to-mcore  \
false \
true \
bf16
```
参数说明：
```bash
MODEL_SIZE=$1               # 模型大小，3B, 7B, 32B, 72B
LOAD_DIR=$2                 # 源权重路径
SAVE_DIR=$3                 # 目标权重路径
MG2HF=$4                    # 转换方向 可选: true, false
USE_CUDA=$5                 # 是否使用GPU转换 建议: true
PR=$6                       # 转换精度 可选: fp32 bf16 fp16
HF_DIR=$7                   # HF权重路径(mcore2hf时必须提供)
```
## Megatron-Core预训练

> 关于attention: Qwen2.5-VL调用了varlen attention，若您使用Hopper架构GPU，推荐将FL设为false以使用FusedAttention后端来获得最佳性能； 对于其他NVIDIA GPU，由于FusedAttention不支持varlen，请将FL设置为true。此外，目前观察到Flash-Attention 3会出现不正常的grad norm，不推荐使用。

## 启动及数据采集

启动训练脚本：
```bash
cd ./Pai-Megatron-Patch/examples/qwen2_5_vl
bash run_mcore_qwen.sh  \
dsw  \                               # 运行环境配置开关: dsw单机训练训练，dlc表示多机训练环境
7B   \                               # 模型结构参数量级: 3B/7B/72B
1    \                               # 一次迭代一个数据并行内的样本数
64 \                                 # GLOBAL_BATCH_SIZE
1e-5   \                             # 学习率
1e-6   \                             # 最小学习率
8192  \                              # 序列长度
8192  \                              # Padding后长度
bf16  \                              # 训练精度: fp16, bf16, fp8
1   \                                # TP
1  \                                 # PP
1 \                                  # CP
true \                               # SP
true \                               # 是否使用Megatron版Zero-1降显存优化器: true, false
false   \                             # 是否优先使用Flash Attention: true, false
false \                              # 激活检查点模式: sel, full, offload, false
false \                              # 是否启用Offload optimizer: false, 或输入0～1的小数作为参数offload比例
100000  \                            # 保存ckpt的间隔
./LLaVA-Pretrain/wds   \             # 训练数据集路径
./LLaVA-Pretrain/wds   \             # 验证数据集路径
./Qwen2.5VL-7B-Instruct-to-mcore \   # 预训练模型路径
500  \                               # Iter数
100   \                              # 预热Iter数        
./output_mcore_qwen2_5_vl_pretrain   # 训练输出日志文件路径
```
替换./Pai-Megatron-Patch/backends/megatron/Megatron-LM-250624/megatron/training/training.py目录下文件用于增加 TGS 指标计算。
```bash
...
tokens_per_gpu_per_sec = (batch_size * args.seq_length / elapsed_time_per_iteration / args.world_size)
...
log_string += f' tokens/sec/gpu: {tokens_per_gpu_per_sec:.1f} |'  
```

### 性能指标

根据训练日志，采集其中Loss数值和相关性能指标。
```bash
[2025-10-24 09:52:15] iteration      480/     500 | consumed samples:        30720 | elapsed time per iteration (ms): 5533.0 | throughput per GPU (TFLOP/s/GPU): 560.9 | learning rate: 1.000000E-06 | global batch size:    64 | lm loss: 1.605485E+00 | loss scale: 1.0 | grad norm: 17.924 | tokens/sec/gpu: 11844.5 | number of skipped iterations:   0 | number of nan iterations:   0 |
[2025-10-24 09:52:21] iteration      481/     500 | consumed samples:        30784 | elapsed time per iteration (ms): 5859.3 | throughput per GPU (TFLOP/s/GPU): 529.7 | learning rate: 1.000000E-06 | global batch size:    64 | lm loss: 1.770668E+00 | loss scale: 1.0 | grad norm: 18.098 | tokens/sec/gpu: 11185.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
[2025-10-24 09:52:27] iteration      482/     500 | consumed samples:        30848 | elapsed time per iteration (ms): 5546.0 | throughput per GPU (TFLOP/s/GPU): 559.6 | learning rate: 1.000000E-06 | global batch size:    64 | lm loss: 1.682535E+00 | loss scale: 1.0 | grad norm: 9.632 | tokens/sec/gpu: 11816.9 | number of skipped iterations:   0 | number of nan iterations:   0 |
```
### 训练目标

根据参考配置训练后，训练到第500个Iter时，Loss值和基准值loss的差异不超过5%。

```bash
[2025-10-24 09:54:09] iteration      500/     500 | consumed samples:        32000 | elapsed time per iteration (ms): 5622.0 | throughput per GPU (TFLOP/s/GPU): 552.0 | learning rate: 1.000000E-06 | global batch size:    64 | lm loss: 1.596570E+00 | loss scale: 1.0 | grad norm: 11.199 | tokens/sec/gpu: 11657.0 | number of skipped iterations:   0 | number of nan iterations:   0 |
```