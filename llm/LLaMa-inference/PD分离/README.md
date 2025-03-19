# LLaMa PD分离推理


## 准备工作

- 代码下载：https://github.com/vllm-project/vllm.git (0590ec3fd9857063c43c80df281e24c16c51b2ec)
- 安装：参考 https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#build-wheel-from-source, 需根据厂商环境进行适配
- 模型： LLAMA2 7B [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)


## 启动及数据采集

### 数据集
使用 `sonnet` [数据集](https://github.com/vllm-project/vllm/blob/main/benchmarks/sonnet.txt)



### 性能测试
新增`disagg_performance_benchmark_new.sh`到目录：`benchmarks/disagg_benchmarks`下，

测试命令：

```bash
sh disagg_performance_benchmark.sh disagg_prefill
```

即可生成测试日志到当前目录下。

日志格式：
```bash

============ Serving Benchmark Result ============
Successful requests:                     20        
Benchmark duration (s):                  23.25     
Total input tokens:                      2199      
Total generated tokens:                  1668      
Request throughput (req/s):              0.86      
Output token throughput (tok/s):         71.74     
Total Token throughput (tok/s):          166.32    
---------------Time to First Token----------------
Mean TTFT (ms):                          40.30     
Median TTFT (ms):                        39.17     
P99 TTFT (ms):                           46.15     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.90     
Median TPOT (ms):                        10.85     
P99 TPOT (ms):                           11.25     
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.76     
Median ITL (ms):                         10.72     
P99 ITL (ms):                            13.59     
----------------End-to-end Latency----------------
Mean E2EL (ms):                          938.06    
Median E2EL (ms):                        935.99    
P99 E2EL (ms):                           1029.90   
==================================================
GPU 0, memory usage:
64.42 GB
GPU 1, memory usage:
64.25 GB


```



### 日志解析（可选）
提供`log_parser_v2.py` 一键解析工具，生成相应的评测数据。

安装依赖：
``` bash
pip install pandas openpyxl
```

使用步骤：
1. 确保所有日志文件存放在指定目录（默认：`./logs`）
2. 运行脚本：python log_parser_v2.py 即生成报告文件 `benchmark_report.xlsx` 到当前目录下


`benchmark_report.xlsx` 输出性能指标如下：

|qps	|input len	|Median TTFT (ms)	|Median TPOT (ms)|
| ---- | ---- | ---- | ---- |
|1	|128	|39.17	|10.85|
|1	|256	|52.5	|10.95|
|1	|512	|69.3	|12.1|
|1	|1024	|108.51	|16.66|
|1	|2048	|173.68	|25.1|
|...	|..	|...	|... |

