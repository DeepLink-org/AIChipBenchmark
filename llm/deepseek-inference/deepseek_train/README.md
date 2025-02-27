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
## 3. Log
启动训练后，可在终端中看到如下日志，其中loss为训练损失，tokens_per_sec为性能指标，日志位于xtuner/work_dirs/deepseek_v2_lite_chat_full_alpaca_e3_32k_varlen/{time_you_run_it}下

``` log
2025/02/09 07:47:19 - mmengine - INFO - Iter(train) [ 8851/96000]  lr: 9.7918e-06  eta: 18 days, 16:18:38  time: 115.0463  data_time: 99.0316  memory: 54047  loss: 0.0018  tflops: 4.8280  tokens_per_sec: 142.4122
2025/02/09 07:47:36 - mmengine - INFO - Iter(train) [ 8852/96000]  lr: 9.7917e-06  eta: 18 days, 16:17:59  time: 16.4075  data_time: 0.0452  memory: 54044  loss: 0.0008  tflops: 33.8529  tokens_per_sec: 998.5649
2025/02/09 07:47:53 - mmengine - INFO - Iter(train) [ 8853/96000]  lr: 9.7917e-06  eta: 18 days, 16:17:25  time: 16.9071  data_time: 0.0439  memory: 54048  loss: 0.0009  tflops: 32.8527  tokens_per_sec: 969.0630
2025/02/09 07:48:10 - mmengine - INFO - Iter(train) [ 8854/96000]  lr: 9.7916e-06  eta: 18 days, 16:16:53  time: 17.1550  data_time: 0.0435  memory: 54046  loss: 0.0016  tflops: 32.3779  tokens_per_sec: 955.0578
2025/02/09 07:48:26 - mmengine - INFO - Iter(train) [ 8855/96000]  lr: 9.7916e-06  eta: 18 days, 16:16:13  time: 16.3679  data_time: 0.0433  memory: 54051  loss: 0.0031  tflops: 33.9349  tokens_per_sec: 1000.9857
2025/02/09 07:48:43 - mmengine - INFO - Iter(train) [ 8856/96000]  lr: 9.7915e-06  eta: 18 days, 16:15:37  time: 16.7183  data_time: 0.0442  memory: 54048  loss: 0.0003  tflops: 33.2237  tokens_per_sec: 980.0064
2025/02/09 07:48:59 - mmengine - INFO - Iter(train) [ 8857/96000]  lr: 9.7915e-06  eta: 18 days, 16:14:54  time: 16.0072  data_time: 0.0445  memory: 54048  loss: 0.0007  tflops: 34.6995  tokens_per_sec: 1023.5386
2025/02/09 07:49:16 - mmengine - INFO - Iter(train) [ 8858/96000]  lr: 9.7914e-06  eta: 18 days, 16:14:19  time: 16.9344  data_time: 0.0438  memory: 54045  loss: 0.0013  tflops: 32.7997  tokens_per_sec: 967.4989
2025/02/09 07:49:32 - mmengine - INFO - Iter(train) [ 8859/96000]  lr: 9.7914e-06  eta: 18 days, 16:13:38  time: 16.1764  data_time: 0.0440  memory: 54047  loss: 0.0010  tflops: 34.3366  tokens_per_sec: 1012.8326
2025/02/09 07:49:48 - mmengine - INFO - Iter(train) [ 8860/96000]  lr: 9.7913e-06  eta: 18 days, 16:12:58  time: 16.3557  data_time: 0.0441  memory: 54047  loss: 0.0017  tflops: 33.9602  tokens_per_sec: 1001.7315
```
## 4. 精度
| 模型 | 数据集 | 精度 | 最少所需iter |
| --- | --- | --- | --- |
| deepseek-ai/DeepSeek-V2-Lite-Chat | tatsu-lab/alpaca | 99.87 | 2000 |