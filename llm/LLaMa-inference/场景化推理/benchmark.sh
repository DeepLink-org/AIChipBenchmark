# #!/bin/bash

########################################## TurboMind engine: fp16 or bf16 ##########################################
# 7B. gemm_tune -> profile_throughput
tp=1
max_batch_size=256
cache_max_entry_count=0.95
model_path="/mnt/lustrenew/dongkaixing1.vendor/model/workspace/llama2/7B"
# optional
CUDA_VISIBLE_DEVICES="6" srun -p pat_rd --gres=gpu:8 python3 -m lmdeploy.turbomind.generate_gemm_config --tensor-para-size ${tp} --max-batch-size ${max_batch_size} --model-path ${model_path}
# test
for concurrency in 4 8 16 32 48 64 96 128
do
    CUDA_VISIBLE_DEVICES="6" srun -p pat_rd --gres=gpu:8 python3 profile.py ${model_path} --tp ${tp} --concurrency ${concurrency} --cache-max-entry-count ${cache_max_entry_count} --csv llama2_tb_7b_thr_${concurrency}.csv
done
rm gemm_config.in


# 13B. gemm_tune -> profile_throughput
tp=1
max_batch_size=256
cache_max_entry_count=0.9
model_path="/mnt/lustrenew/dongkaixing1.vendor/model/workspace/llama2/13B"
# optional
CUDA_VISIBLE_DEVICES="6" srun -p pat_rd --gres=gpu:8 python3 -m lmdeploy.turbomind.generate_gemm_config --tensor-para-size ${tp} --max-batch-size ${max_batch_size} --model-path ${model_path}
# test
for concurrency in 4 8 16 32 48 64 96 128
do
    CUDA_VISIBLE_DEVICES="6" srun -p pat_rd --gres=gpu:8 python3 profile.py ${model_path} --tp ${tp} --concurrency ${concurrency} --cache-max-entry-count ${cache_max_entry_count} --csv llama2_tb_13b_thr_${concurrency}.csv
done
rm gemm_config.in


# 65B
tp=8
max_batch_size=256
cache_max_entry_count=0.9
model_path="/mnt/lustrenew/dongkaixing1.vendor/model/workspace/llama/65B"
for concurrency in 4 8 16 32 48 64 96 128
do
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" srun -p pat_rd --gres=gpu:8 python3 profile.py ${model_path} --tp ${tp} --concurrency ${concurrency} --cache-max-entry-count ${cache_max_entry_count} --csv llama_tb_65b_thr_${concurrency}.csv
done


# 70B
tp=8
max_batch_size=256
cache_max_entry_count=0.9
model_path="/mnt/lustrenew/dongkaixing1.vendor/model/workspace/llama2/70B"
for concurrency in 4 8 16 32 48 64 96 128
do
    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" srun -p pat_rd --gres=gpu:8 python3 profile.py ${model_path} --tp ${tp} --concurrency ${concurrency} --cache-max-entry-count ${cache_max_entry_count} --csv llama2_tb_70b_thr_${concurrency}.csv
done
