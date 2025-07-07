set -ex

if [ $# -eq 0 ]; then
    echo "Error: Please provide a node number as argument"
    exit 1
fi
node=$1


current_date=$(date +"%Y%m%d_%H%M%S")
mkdir -p ${current_date}


backend="lmdeploy"
dataset_name="random"
dataset_path="ShareGPT_V3_unfiltered_cleaned_split.json"


for input_len in 512 1024 2048 4096 8192 32768 
do
    for out_len in 1024 2048 4096 8192
    do
        no_prompts=$(( (2048 * 2048 * 2000) / (input_len * out_len) ))

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
            --host 10.130.8.${node} --port 8000 \
            --output-file "${current_date}/speed_in${input_len}_out${out_len}_n${no_prompts}_dsv3.csv" \
            --seed 42 2>&1 | tee ${current_date}/speed_in${input_len}_out${out_len}_n${no_prompts}_dsv3.log

            sleep 60
    done
done
