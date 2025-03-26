# Deepseek 推理
## step 1 Model
下载模型，DeepSeek-R1	671B 模型位于huggingface中，model_path:[deepseek-ai/DeepSeek-R1](https://hf-mirror.com/deepseek-ai/DeepSeek-R1)

## step 2 Inference
使用vllm等推理框架将推理服务部署起来，部署过程这里不赘述，请参考各框架使用指导。

## step 3 Benchmark
运行单例测试脚本
```
python3 llm_profile.py \
    --url ${URL} \
    --num_clients ${WORKER} \
    --tokenizer_path ${MODEL_PATH} \
    --input_len 2048 \
    --output_len 128 \
    --input_num 200 \
    --trust_remote_code
```
参数解释
```
url: http://localhost:8080/v1/completions # 服务地址
num_clients: 5 # 测试的并发数控制
tokenizer_path: /data_share/model/DeepSeek-R1 # 指定模型的tokenizer路径
input_len: 2048 # 输入长度
input_num: 200 # 测试数据的数量
output_len: 128 # 输出最大长度
```

或者在脚本中指定关键配置后运行测试脚本 `python benchmark_all.py`，得到测试结果`test_result.csv`，每一条测例保存在对应log中。
```
# 测试配置
URL = "http://localhost:8080/v1/completions"
WORKER = 1
MODEL_PATH = "/data_share/model/DeepSeek-R1"
INPUT_NUM = 50

# 要测试的输入和输出长度组合
input_lengths = [256, 512]
output_lengths = [128, 512, 1024, 2048, 4096, 8192]
```
## step 4 Result
从日志中选取 `Output Throughput: 172.6589935410431 token/s` 作为TPS性能结果。

# Deepseek 训练
## 1. Model & Datasets
模型位于huggingface中，model_path:[deepseek-ai/DeepSeek-V2-Lite-Chat](https://hf-mirror.com/deepseek-ai/DeepSeek-V2-Lite-Chat)

数据集使用Huggingface的[tatsu-lab/alpaca](https://hf-mirror.com/datasets/tatsu-lab/alpaca) 

模型配置文件：[deepseek_v2_lite_chat_full_alpaca_e3_32k_varlen.py](./deepseek_v2_lite_chat_full_alpaca_e3_32k_varlen.py)

模型运行参数
```
max_length = 32768
sequence_parallel_size = 2
batch_size = 1  # per_device
accumulative_counts = 1 * sequence_parallel_size
dataloader_num_workers = 4
global batch size = batch_size * gpu_nums / sequence_parallel_size
```
## 2. Train
训练/微调采用Xtuner框架进行，训练脚本及config文件如上所述
### 仓库安装
```
git clone https://github.com/InternLM/xtuner.git
cd xtuner
pip install -e '.[all]'
```

### 启动训练/微调
本节使用了8卡进行训练，并且在前述模型配置文件中开启了sequence parallel，并行度指定为2,global batch size也因此为4
```
 NPROC_PER_NODE=${GPU_NUM} xtuner train deepseek_v2_lite_chat_full_alpaca_e3_32k_varlen --deepspeed deepspeed_zero2 
 #deepspeed选项中的模型并行策略，可通过修改其参数指定配置文件，配置文件位于xtuner/xtuner/configs/deepspeed下
```
## 3. Log
启动训练后，可在终端中看到如下日志，其中loss为训练损失，tokens_per_sec为性能指标，日志位于xtuner/work_dirs/deepseek_v2_lite_chat_full_alpaca_e3_32k_varlen/{time_you_run_it}下

``` log
2025/02/09 07:49:32 - mmengine - INFO - Iter(train) [ 8859/96000]  lr: 9.7914e-06  eta: 18 days, 16:13:38  time: 16.1764  data_time: 0.0440  memory: 54047  loss: 0.0010  tflops: 34.3366  tokens_per_sec: 1012.8326
2025/02/09 07:49:48 - mmengine - INFO - Iter(train) [ 8860/96000]  lr: 9.7913e-06  eta: 18 days, 16:12:58  time: 16.3557  data_time: 0.0441  memory: 54047  loss: 0.0017  tflops: 33.9602  tokens_per_sec: 1001.7315
```

## 4. 性能指标
基于上述模型和数据集，在不低于2000iter的迭代次数，收敛到loss<0.0013