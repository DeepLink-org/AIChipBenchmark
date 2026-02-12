set -ex

current_date=$(date +"%Y%m%d_%H%M%S")
mkdir -p /mnt/nvme1n1/dongkaixing/sglang/logs/${current_date}

backend="sglang"
dataset_name="random"
dataset_path="/mnt/nvme1n1/dongkaixing/sglang/ShareGPT_V3_unfiltered_cleaned_split.json"

for input_len in 128 256 512 1024 2048
do
    for qps in 1 8 16 32 64 
    do
        out_len=$input_len
        concurrency=$(( qps * 2 ))
        no_prompts=$(( qps * 100 ))
        echo "running with output-len $out_len and num-prompts $no_prompts"
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
            --output-file "/mnt/nvme1n1/dongkaixing/sglang/logs/${current_date}/speed_in${input_len}_out${out_len}_n${no_prompts}_pd_llama.csv" \
            --seed 42 2>&1 | tee /mnt/nvme1n1/dongkaixing/sglang/logs/${current_date}/speed_in${input_len}_out${out_len}_n${no_prompts}_pd_llama.log

        sleep 2
    done
done

