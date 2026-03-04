# LLaMa PD分离推理


## 准备工作

- 代码下载：https://github.com/sgl-project/sglang.git (5e2cda6158e670e64b926a9985d65826c537ac82)
- 安装：参考 https://docs.sglang.io/get_started/install.html, 需根据厂商环境进行适配
- 模型： Llama-3.1-8B-Instruct [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)


## 启动及数据采集

### 数据集
使用 `ShareGPT_V3_unfiltered_cleaned_split` [数据集](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json)


### 启动Server
```
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct
  --disaggregation-mode prefill \
  --port 30000

python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1

python -m sglang_router.launch_router --pd-disaggregation --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30001 --host 0.0.0.0 --port 8000  

```

### 性能测试

测试脚本统一采用`sglang benchmark`，可使用`pip install sglang`安装。

评测命令参考如下：

```
backend="sglang"
dataset_name="random"
dataset_path="/path/to/ShareGPT_V3_unfiltered_cleaned_split.json"

            TRANSFORMERS_OFFLINE=1 \
            python3 -m sglang.bench_serving \
                --random-range-ratio 1 \
                --backend ${backend} \
                --dataset-name random \
                --dataset-path ${dataset_path} \
                --random-input-len $input_len \
                --random-output-len $out_len \
                --num-prompts $no_prompts \
                --max-concurrency $concurrency \
                --request-rate ${qps} \
                --host 127.0.0.1 --port 8000 \
                --flush-cache \
                --output-file "${current_date}/speed_in${input_len}_out${out_len}_n${no_prompts}_pd_llama.csv" \
                --seed 42 2>&1 | tee ${current_date}/speed_in${input_len}_out${out_len}_n${no_prompts}_pd_llama.log
```

保持`qps`和评测方案一致，保持`random-range-ratio`和`seed`一致。其余参数可自行调整，以发挥推理引擎和芯片的最佳性能。建议配置`concurrency`为`$(( qps * 2 ))`，`no_prompts`为`$(( qps * 100 ))`，可参考`bench_pd.sh`。


基准`Benchmark`的详细输出日志可联系`Deeplink`评测团队获取。

## 评测结果
```bash
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    64.0      
Max request concurrency:                 128       
Successful requests:                     6400      
Benchmark duration (s):                  271.12    
Total input tokens:                      3276800   
Total input text tokens:                 3276800   
Total input vision tokens:               0         
Total generated tokens:                  3276800   
Total generated tokens (retokenized):    3276879   
Request throughput (req/s):              23.61     
Input token throughput (tok/s):          12086.34  
Output token throughput (tok/s):         12086.34  
Peak output token throughput (tok/s):    13873.00  
Peak concurrent requests:                189       
Total token throughput (tok/s):          24172.68  
Concurrency:                             126.92    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   5376.75   
Median E2E Latency (ms):                 5377.32   
---------------Time to First Token----------------
Mean TTFT (ms):                          74.66     
Median TTFT (ms):                        62.73     
P99 TTFT (ms):                           271.46    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.38     
Median TPOT (ms):                        10.40     
P99 TPOT (ms):                           10.70     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           10.37     
Median ITL (ms):                         10.20     
P95 ITL (ms):                            13.11     
P99 ITL (ms):                            43.91     
Max ITL (ms):                            320.67    
==================================================

```


