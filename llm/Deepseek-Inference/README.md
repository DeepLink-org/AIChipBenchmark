# Deepseek 推理
## step 1 Model
下载模型，DeepSeek-R1 671B 模型位于huggingface中，model_path:[deepseek-ai/DeepSeek-R1](https://hf-mirror.com/deepseek-ai/DeepSeek-R1)


## step 2 Inference Server
使用`vllm/sglang/lmdeploy`等推理框架将推理服务部署起来，请参考各框架使用指导。评测对框架不做限制。

以`lmdeploy 0.9.0`为例，服务启动参考命令如下：

``` Bash
dp=32
ep=32

export MAX_PREFILL_TOKENS="8192"
export CACHERATIO="0.8"
export MAXBSZ="128"

lmdeploy serve api_server $model_path \
    --cache-max-entry-count ${CACHERATIO} \
    --max-prefill-token-num ${MAX_PREFILL_TOKENS} \
    --dp $dp \
    --ep $ep \
    --log-level ERROR \
    --max-batch-size ${MAXBSZ} \
    --proxy-url $proxy_url \
    --backend $backend \
    --nnodes $nnodes \
    --enable-microbatch \
    --node-rank $node_rank
```

当前lmdeploy已支持`FlashMLA`、`DeepGEMM`、`DeepEP`、`two-mircobatch`、`EPLB`等特性。


### EPLB（Expert Parallelism Load Balance） 
评测允许模拟完美的专家负载均衡，参考实现见`deepseek_v2.py`（`lmdeploy/pytorch/models/deepseek_v2.py`），关键代码如下，可作为参考，适配不同框架。

``` Python
--- a/lmdeploy/pytorch/models/deepseek_v2.py
+++ b/lmdeploy/pytorch/models/deepseek_v2.py
@@ -734,6 +734,12 @@ class DeepseekV2MoE(nn.Module):
         batch_size, sequence_length, hidden_dim = hidden_states.shape
         hidden_states = hidden_states.view(-1, hidden_dim)
         topk_weights, topk_ids = self.gate(hidden_states)
+        
+        ranks = torch.distributed.get_world_size()
+        shape = topk_ids.shape
+        topk_ids = (torch.arange(0, topk_ids.numel(), dtype=torch.int32, device=topk_ids.device) % ranks) * self.num_experts / ranks
+        topk_ids = topk_ids.reshape(shape).to(dtype=torch.int64)
+    
         out_states = self.experts(
             hidden_states,
             topk_weights,
@@ -908,6 +914,11 @@ class DeepseekV2DecoderLayer(nn.Module):
         hidden_states = hidden_states.view(-1, hidden_dim)
         topk_weights, topk_idx = self.mlp.gate(hidden_states)
 
+        ranks = torch.distributed.get_world_size()
+        shape = topk_idx.shape
+        topk_idx = (torch.arange(0, topk_idx.numel(), dtype=torch.int32, device=topk_idx.device) % ranks) * self.mlp.num_experts / ranks
+        topk_idx = topk_idx.reshape(shape).to(dtype=torch.int64)
+
         topk_weights = self.mlp.experts.renormalize(topk_weights)
         topk_weights = topk_weights.to(torch.float32)
         topk_idx = topk_idx.to(torch.int64)

```



## step 3 Benchmark

测试脚本统一采用`sglang benchmark`，可使用`pip install sglang`安装。


评测命令参考如下：

```
backend="lmdeploy"
dataset_name="random"
dataset_path="/mnt/139_nvme2/dongkaixing/lmdeploy/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"

            TRANSFORMERS_OFFLINE=1 \
            python3 -m sglang.bench_serving \
                --random-range-ratio 1 \
                --backend ${backend} \
                --dataset-name random \
                --dataset-path ${dataset_path} \
                --random-input-len $input_len \
                --random-output-len $out_len \
                --num-prompts $no_prompts \
                --host 10.130.8.${node} --port 8000 \
                --output-file "${current_date}/speed_in${input_len}_out${out_len}_n${no_prompts}_dsv3.csv" \
                --seed 42 2>&1 | tee ${current_date}/speed_in${input_len}_out${out_len}_n${no_prompts}_dsv3.log
```

数据集统一采用`ShareGPT_V3_unfiltered_cleaned_split.json`，保持`random-range-ratio`和`seed`一致。其余参数可自行调整，以发挥推理引擎和芯片的最佳性能。

基准`Benchmark`的详细输出日志可联系`Deeplink`评测团队获取。


## step 4 Result

本项评测关注Deepseek推理中的高吞吐场景，根据下面参考日志输出，吞吐取`Output token throughput (tok/s)`。

```
============ Serving Benchmark Result ============
Backend:                                 lmdeploy  
Traffic request rate:                    inf       
Max request concurrency:                 not set   
Successful requests:                     2000      
Benchmark duration (s):                  127.11    
Total input tokens:                      4096000   
Total generated tokens:                  4096000   
Total generated tokens (retokenized):    833623    
Request throughput (req/s):              15.73     
Input token throughput (tok/s):          32223.82  
Output token throughput (tok/s):         32223.82  
Total token throughput (tok/s):          64447.64  
Concurrency:                             1949.99   
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   123932.63 
Median E2E Latency (ms):                 123990.52 
---------------Time to First Token----------------
Mean TTFT (ms):                          18867.14  
Median TTFT (ms):                        18679.39  
P99 TTFT (ms):                           33717.70  
---------------Inter-Token Latency----------------
Mean ITL (ms):                           118.64    
Median ITL (ms):                         127.52    
P95 ITL (ms):                            167.02    
P99 ITL (ms):                            365.22    
Max ITL (ms):                            3469.54   
==================================================

```

