#!/bin/bash

# Requirement: 2x GPUs.


# Model: meta-llama/Meta-Llama-3.1-8B-Instruct
# Query: 1024 input tokens, 6 output tokens, QPS 2/4/6/8, 100 requests
# Resource: 2x GPU
# Approaches:
# 2. Chunked prefill: 2 vllm instance with tp=4, equivalent to 1 tp=4 instance with QPS 4
# 3. Disaggregated prefill: 1 prefilling instance and 1 decoding instance
# Prefilling instance: max_output_token=1
# Decoding instance: force the input tokens be the same across requests to bypass prefilling

set -ex
gpu_ratio=0.8
gpu_id1=0
gpu_id2=1

kill_gpu_processes() {
  # kill all processes on GPU.
  pgrep pt_main_thread | xargs -r kill -9
  pgrep python3 | xargs -r kill -9
  for port in 8000 8100 8200; do lsof -t -i:$port | xargs -r kill -9; done
  sleep 1
}

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

launch_disagg_prefill() {
  model="meta-llama/Llama-2-7b-chat-hf" 
  # disagg prefill
  CUDA_VISIBLE_DEVICES=$gpu_id1 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8100 \
    --max-model-len 10000 \
    --gpu-memory-utilization $gpu_ratio \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  CUDA_VISIBLE_DEVICES=$gpu_id2 python3 \
    -m vllm.entrypoints.openai.api_server \
    --model $model \
    --port 8200 \
    --max-model-len 10000 \
    --gpu-memory-utilization $gpu_ratio \
    --kv-transfer-config \
    '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":5e9}' &

  wait_for_server 8100
  wait_for_server 8200
  python3 disagg_prefill_proxy_server.py &
  sleep 1
}


benchmark() {
  results_folder="./results"
  model="meta-llama/Llama-2-7b-chat-hf"
  dataset_name="sonnet"
  dataset_path="../sonnet_4x.txt"
  num_prompts=20
  qps=$1
  prefix_len=$4
  input_len=$3
  # output_len=$2
  output_len=$3
  tag=$5

  python3 ../benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name $dataset_name \
          --dataset-path $dataset_path \
          --sonnet-input-len $input_len \
          --sonnet-output-len "$output_len" \
          --sonnet-prefix-len $prefix_len \
          --num-prompts $num_prompts \
          --port 8000 \
          --save-result \
          --result-dir $results_folder \
          --result-filename "$tag"-input_len-"$input_len"-qps-"$qps".json \
          --request-rate "$qps" \
          --percentile-metrics "ttft,tpot,itl,e2el" > "$tag"-input_len-"$input_len"-qps-"$qps"-num_prompts-"$num_prompts"-gpu_ratio-"$gpu_ratio".log 2>&1

    for gpu_id in $gpu_id1 $gpu_id2; do
        echo "GPU $gpu_id memory usage:" >> "$tag"-input_len-"$input_len"-qps-"$qps"-num_prompts-"$num_prompts"-gpu_ratio-"$gpu_ratio".log
        nvidia-smi --id=$gpu_id --query-gpu=memory.used --format=csv,noheader,nounits | awk '{ printf "%.2f GB\n", $1 / 1024 }' >> "$tag"-input_len-"$input_len"-qps-"$qps"-num_prompts-"$num_prompts"-gpu_ratio-"$gpu_ratio".log
    done
  sleep 2
}


main() {

  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)
  (which lsof) || (apt-get -y install lsof)

  pip install quart httpx matplotlib aiohttp datasets

  cd "$(dirname "$0")"

  cd ..
  # create sonnet-4x.txt so that we can sample 2048 tokens for input
  echo "" > sonnet_4x.txt
  for _ in {1..4}
  do
    cat sonnet.txt >> sonnet_4x.txt
  done
  cd disagg_benchmarks

  rm -rf results
  mkdir results

  default_output_len=6 # not used
  default_prefix_len=50

  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

  launch_disagg_prefill

  for qps in 1 2 3 4 5 6 7 8; do
    for inputlen in 128 256 512 1024 2048; do
          benchmark $qps $default_output_len $inputlen $default_prefix_len disagg_prefill
      done
    done
  kill_gpu_processes

  # python3 visualize_benchmark_results.py

}


main "$@"