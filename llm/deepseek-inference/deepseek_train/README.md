# Xtuner deepseek lite 16b sft
## 1. Model & Datasets
模型位于huggingface中，model_path:[deepseek-ai/DeepSeek-V2-Lite-Chat](https://hf-mirror.com/deepseek-ai/DeepSeek-V2-Lite-Chat)

数据集使用Huggingface的[tatsu-lab/alpaca](https://hf-mirror.com/datasets/tatsu-lab/alpaca) 

模型配置文件：[deepseek_v2_lite_chat_full_alpaca_e3_32k_varlen.py](https://github.com/InternLM/xtuner/blob/main/xtuner/configs/deepseek/deepseek_v2_lite_chat/deepseek_v2_lite_chat_full_alpaca_e3_32k_varlen.py)

## 2. Train
训练/微调采用Xtuner框架进行，训练脚本及config文件如上所述
### 仓库安装
```
git clone https://github.com/InternLM/xtuner.git
cd xtuner
pip install -e '.[all]'
```

### 启动训练/微调
```
 NPROC_PER_NODE=${GPU_NUM} xtuner train deepseek_v2_lite_chat_full_alpaca_e3_32k_varlen --deepspeed deepspeed_zero2 
 #deepspeed选项中的模型并行策略，可通过修改其参数指定配置文件，配置文件位于xtuner/xtuner/configs/deepspeed下
```
