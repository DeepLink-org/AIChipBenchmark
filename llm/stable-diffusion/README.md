# Stable Diffusion 训练


## 准备工作

- 代码下载：https://github.com/huggingface/diffusers
- 安装：pip install git+https://github.com/huggingface/diffusers
- 环境依赖：torch==2.1.0, transformers==4.38.1, diffusers==v0.27.0.dev, datasets==2.16.1, accelerate==0.23.0等
- 模型权重：stable diffusion v1-4 https://huggingface.co/CompVis/stable-diffusion-v1-4
            stable diffusion v1-5 https://huggingface.co/runwayml/stable-diffusion-v1-5
            stable diffusion v2-1 https://huggingface.co/stabilityai/stable-diffusion-2-1

- 数据集：https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions
    使用diffusers/examples/text_to_image/train_text_to_image.py 辅助进行启动训练, 这里需要对train_text_to_image.py做以下方面修改：
    - 数据加载本地的datasets，如果环境可以联网，可以不必修改dataset相关改动; 
    - step信息修改为实时显示'train_loss';
    - 性能测试时，加入时间戳，取时间段计算
    - 其他修改，针对该数据集的convert function修改等
    具体修改可以参考本目录下的train_text_to_image.py（TODO::sd_train modify）

- 集群环境：本文档测试使用1424集群环境，A100算力，IB为200； 以下设置和数据均基于此环境

## 配置
使用`accelerate`辅助进行启动训练，`accelerate_config.yaml`文件内容参考：
```
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: 'fp16'
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
```

不启用以下优化：
- torch dynamo
- FullyShardedDataParallel


关键配置：
- mixed_precision：开启混合精度fp16
- 关闭gradient checkpointing
- 单机8卡或多机多卡, 基准值测试使用单机8卡或4机32卡


## 启动及数据采集

启动命令参考：

```bash
#!/bin/sh
export MODEL_NAME="/mnt/afs/huayil/models/stable-diffusion-v1-5"
#export MODEL_NAME="/mnt/afs/liuyipeng/workspace_2/stable-diffusion-2-1"
export OUTPUT_DIR="/mnt/afs/huayil/code/lora/pokemon"
export HUB_MODEL_ID="pokemon-lora"
export DATASET_NAME="pokemon/train-00000-of-00001-566cc9b19d7203f8.parquet"

#python train_text_to_image_loss_test.py \
accelerate launch --multi_gpu  train_text_to_image_loss_test.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --resolution=960 \
  --center_crop \
  --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=100 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --checkpointing_steps=1500 \
  --seed=1337 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --output_dir=$OUTPUT_DIR \
#多机时需设置 accelerate 其他参数，参考：
#accelerate launch --config_file  ./accelerate_config.yaml \
#    --num_machines "$SLURM_NNODES" \
#    --num_processes "$num_processes" \
#    --gpu_ids "$SLURM_STEP_GPUS" \
#    --machine_rank "$SLURM_PROCID" \
#    --main_process_ip "$head_node_ip" \
#    --main_process_port 29512 \

```


```bash
# 集群srun 以上sd_train.sh脚本
# 单机8卡
srun -p pat_rd -n 1 --ntasks-per-node=1 --gpus-per-task=8 sd_train.sh

# 4机32卡
srun -p pat_rd -n 4 --ntasks-per-node=1 --gpus-per-task=8 sd_train.sh
```

关键参数说明：
- train_batch_size：per device batch size，设置为8
- gradient_accumulation_steps：梯度累计，设置为4
- resolution: 分辨率，多机多卡在性能测试时设置为960; 功能测试时设置为256，loss下降较为明显。单机8卡下使用默认配置512
- max_train_steps: 性能迭代次数，设置为100，为了消除step波动的影响，`time(s/it)`取100个step的均值。可[参考] (## 训练目标)
- gradient checkpointing：处于性能考虑，关闭梯度检查点功能
- 模型选择：可[参考] (## 训练目标)

性能指标`IPS`计算：
- dp：基准数据使用单机8卡时，`dp=8`； 使用4机32卡时，`dp=32`
- time: 每次迭代时间，可从日志中得到
- gbsz = train_batch_size * gradient_accumulation_steps * dp。
- IPS = gbsz / time / #gpu


### 日志参考

#### 单机8卡

```
06/20/2024 20:22:01 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 3
Local process index: 3
Device: cuda:3

Mixed precision type: fp16

06/20/2024 20:22:01 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 2
Local process index: 2
Device: cuda:2

Mixed precision type: fp16

06/20/2024 20:22:01 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 1
Local process index: 1
Device: cuda:1

Mixed precision type: fp16

06/20/2024 20:22:01 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 5
Local process index: 5
Device: cuda:5

Mixed precision type: fp16

06/20/2024 20:22:01 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 6
Local process index: 6
Device: cuda:6

Mixed precision type: fp16

06/20/2024 20:22:01 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 4
Local process index: 4
Device: cuda:4

Mixed precision type: fp16

06/20/2024 20:22:01 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 7
Local process index: 7
Device: cuda:7

Mixed precision type: fp16

06/20/2024 20:22:01 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 0
Local process index: 0
Device: cuda:0

Mixed precision type: fp16

{'variance_type', 'rescale_betas_zero_snr', 'sample_max_value', 'prediction_type', 'clip_sample_range', 'dynamic_thresholding_ratio', 'thresholding', 'timestep_spacing'} was not found in config. Values will be initialized to default values.
{'scaling_factor', 'latents_mean', 'force_upcast', 'latents_std'} was not found in config. Values will be initialized to default values.
{'cross_attention_norm', 'conv_in_kernel', 'transformer_layers_per_block', 'dropout', 'num_attention_heads', 'reverse_transformer_layers_per_block', 'addition_embed_type', 'attention_type', 'class_embeddings_concat', 'resnet_out_scale_factor', 'encoder_hid_dim_type', 'class_embed_type', 'time_embedding_type', 'resnet_time_scale_shift', 'time_embedding_act_fn', 'projection_class_embeddings_input_dim', 'num_class_embeds', 'resnet_skip_time_act', 'addition_time_embed_dim', 'time_embedding_dim', 'timestep_post_act', 'mid_block_type', 'only_cross_attention', 'dual_cross_attention', 'time_cond_proj_dim', 'addition_embed_type_num_heads', 'upcast_attention', 'mid_block_only_cross_attention', 'conv_out_kernel', 'use_linear_projection', 'encoder_hid_dim'} was not found in config. Values will be initialized to default values.
{'cross_attention_norm', 'conv_in_kernel', 'transformer_layers_per_block', 'dropout', 'num_attention_heads', 'reverse_transformer_layers_per_block', 'addition_embed_type', 'attention_type', 'class_embeddings_concat', 'resnet_out_scale_factor', 'encoder_hid_dim_type', 'class_embed_type', 'time_embedding_type', 'resnet_time_scale_shift', 'time_embedding_act_fn', 'projection_class_embeddings_input_dim', 'num_class_embeds', 'resnet_skip_time_act', 'addition_time_embed_dim', 'time_embedding_dim', 'timestep_post_act', 'mid_block_type', 'only_cross_attention', 'dual_cross_attention', 'time_cond_proj_dim', 'addition_embed_type_num_heads', 'upcast_attention', 'mid_block_only_cross_attention', 'conv_out_kernel', 'use_linear_projection', 'encoder_hid_dim'} was not found in config. Values will be initialized to default values.
06/20/2024 20:22:49 - INFO - __main__ - ***** Running training *****
06/20/2024 20:22:49 - INFO - __main__ -   Num examples = 833
06/20/2024 20:22:49 - INFO - __main__ -   Num Epochs = 26
06/20/2024 20:22:49 - INFO - __main__ -   Instantaneous batch size per device = 8
06/20/2024 20:22:49 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 256
06/20/2024 20:22:49 - INFO - __main__ -   Gradient Accumulation steps = 4
06/20/2024 20:22:49 - INFO - __main__ -   Total optimization steps = 101
Steps:   0%|          | 0/101 [00:00<?, ?it/s]/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
Steps:   0%|          | 0/101 [00:07<?, ?it/s, lr=1e-5, step_loss=0.0211]/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
Steps:   1%|          | 1/101 [00:09<15:51,  9.52s/it, lr=1e-5, step_loss=0.0413]06/20/2024 20:22:59 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:22:59 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:22:59 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:22:59 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:22:59 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:22:59 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:22:59 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:22:59 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
Steps: elapsed_time:{}| 100/101 [11:31<00:06,  6.21s/it, lr=1e-5, step_loss=0.0379]
 693.5912295281887
elapsed_time:{}
 693.5907026845962
elapsed_time:{}
 693.5912974942476
elapsed_time:{}
 693.5915133804083
elapsed_time:{}
 693.6219155732542
elapsed_time:{}
 693.5917899236083
elapsed_time:{}
 693.5924326516688
elapsed_time:{}
 693.5089196749032
[2024-06-20 20:34:23,132] [INFO] [real_accelerator.py:161:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Steps: 100%|██████████| 101/101 [11:33<00:00,  6.75s/it, lr=1e-5, step_loss=0.0111]{'image_encoder', 'requires_safety_checker'} was not found in config. Values will be initialized to default values.
                                                                     Loaded safety_checker as StableDiffusionSafetyChecker from `safety_checker` subfolder of /mnt/lustrenew/share_data/PAT/datasets/StableDiffusion/stable-diffusion-v1-5/.          | 0/7 [00:00<?, ?it/s]
                                                                             {'prediction_type', 'timestep_spacing'} was not found in config. Values will be initialized to default values.
Loaded scheduler as PNDMScheduler from `scheduler` subfolder of /mnt/lustrenew/share_data/PAT/datasets/StableDiffusion/stable-diffusion-v1-5/.
Loaded tokenizer as CLIPTokenizer from `tokenizer` subfolder of /mnt/lustrenew/share_data/PAT/datasets/StableDiffusion/stable-diffusion-v1-5/.
Loaded feature_extractor as CLIPImageProcessor from `feature_extractor` subfolder of /mnt/lustrenew/share_data/PAT/datasets/StableDiffusion/stable-diffusion-v1-5/.
Loading pipeline components...: 100%|██████████| 7/7 [00:01<00:00,  3.74it/s]
Configuration saved in /mnt/lustrenew/liuyipeng/workspace/stable_diffusion/sddata/train/loss_test/cc-2000_2/vae/config.json
Model weights saved in /mnt/lustrenew/liuyipeng/workspace/stable_diffusion/sddata/train/loss_test/cc-2000_2/vae/diffusion_pytorch_model.safetensors
Configuration saved in /mnt/lustrenew/liuyipeng/workspace/stable_diffusion/sddata/train/loss_test/cc-2000_2/unet/config.json
Model weights saved in /mnt/lustrenew/liuyipeng/workspace/stable_diffusion/sddata/train/loss_test/cc-2000_2/unet/diffusion_pytorch_model.safetensors
Configuration saved in /mnt/lustrenew/liuyipeng/workspace/stable_diffusion/sddata/train/loss_test/cc-2000_2/scheduler/scheduler_config.json
Configuration saved in /mnt/lustrenew/liuyipeng/workspace/stable_diffusion/sddata/train/loss_test/cc-2000_2/model_index.json
Steps: 100%|██████████| 101/101 [11:52<00:00,  7.06s/it, lr=1e-5, step_loss=0.0111]

```
#### 4机32卡

```
06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 28
Local process index: 4
Device: cuda:4

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 27
Local process index: 3
Device: cuda:3

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 25
Local process index: 1
Device: cuda:1

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 30
Local process index: 6
Device: cuda:6

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 26
Local process index: 2
Device: cuda:2

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 24
Local process index: 0
Device: cuda:0

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 29
Local process index: 5
Device: cuda:5

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 31
Local process index: 7
Device: cuda:7

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 14
Local process index: 6
Device: cuda:6

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 8
Local process index: 0
Device: cuda:0

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 12
Local process index: 4
Device: cuda:4

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 15
Local process index: 7
Device: cuda:7

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 9
Local process index: 1
Device: cuda:1

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 11
Local process index: 3
Device: cuda:3

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 10
Local process index: 2
Device: cuda:2

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 13
Local process index: 5
Device: cuda:5

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 23
Local process index: 7
Device: cuda:7

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 21
Local process index: 5
Device: cuda:5

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 22
Local process index: 6
Device: cuda:6

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 16
Local process index: 0
Device: cuda:0

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 17
Local process index: 1
Device: cuda:1

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 20
Local process index: 4
Device: cuda:4

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 19
Local process index: 3
Device: cuda:3

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 18
Local process index: 2
Device: cuda:2

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 5
Local process index: 5
Device: cuda:5

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 0
Local process index: 0
Device: cuda:0

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 1
Local process index: 1
Device: cuda:1

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 4
Local process index: 4
Device: cuda:4

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 6
Local process index: 6
Device: cuda:6

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 3
Local process index: 3
Device: cuda:3

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 7
Local process index: 7
Device: cuda:7

Mixed precision type: fp16

06/20/2024 20:39:30 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 32
Process index: 2
Local process index: 2
Device: cuda:2

Mixed precision type: fp16

{'prediction_type', 'sample_max_value', 'dynamic_thresholding_ratio', 'rescale_betas_zero_snr', 'timestep_spacing', 'thresholding', 'variance_type', 'clip_sample_range'} was not found in config. Values will be initialized to default values.
{'prediction_type', 'variance_type', 'clip_sample_range', 'rescale_betas_zero_snr', 'thresholding', 'dynamic_thresholding_ratio', 'sample_max_value', 'timestep_spacing'} was not found in config. Values will be initialized to default values.
{'sample_max_value', 'rescale_betas_zero_snr', 'dynamic_thresholding_ratio', 'thresholding', 'clip_sample_range', 'timestep_spacing', 'prediction_type', 'variance_type'} was not found in config. Values will be initialized to default values.
{'rescale_betas_zero_snr', 'sample_max_value', 'clip_sample_range', 'prediction_type', 'variance_type', 'thresholding', 'timestep_spacing', 'dynamic_thresholding_ratio'} was not found in config. Values will be initialized to default values.
{'force_upcast', 'latents_mean', 'scaling_factor', 'latents_std'} was not found in config. Values will be initialized to default values.
{'time_embedding_act_fn', 'conv_out_kernel', 'dropout', 'addition_embed_type_num_heads', 'resnet_out_scale_factor', 'encoder_hid_dim', 'transformer_layers_per_block', 'reverse_transformer_layers_per_block', 'mid_block_type', 'num_attention_heads', 'class_embed_type', 'num_class_embeds', 'time_cond_proj_dim', 'resnet_skip_time_act', 'use_linear_projection', 'timestep_post_act', 'cross_attention_norm', 'conv_in_kernel', 'upcast_attention', 'only_cross_attention', 'time_embedding_type', 'class_embeddings_concat', 'addition_time_embed_dim', 'addition_embed_type', 'resnet_time_scale_shift', 'projection_class_embeddings_input_dim', 'dual_cross_attention', 'encoder_hid_dim_type', 'time_embedding_dim', 'attention_type', 'mid_block_only_cross_attention'} was not found in config. Values will be initialized to default values.
{'time_embedding_act_fn', 'conv_out_kernel', 'dropout', 'addition_embed_type_num_heads', 'resnet_out_scale_factor', 'encoder_hid_dim', 'transformer_layers_per_block', 'reverse_transformer_layers_per_block', 'mid_block_type', 'num_attention_heads', 'class_embed_type', 'num_class_embeds', 'time_cond_proj_dim', 'resnet_skip_time_act', 'use_linear_projection', 'timestep_post_act', 'cross_attention_norm', 'conv_in_kernel', 'upcast_attention', 'only_cross_attention', 'time_embedding_type', 'class_embeddings_concat', 'addition_time_embed_dim', 'addition_embed_type', 'resnet_time_scale_shift', 'projection_class_embeddings_input_dim', 'dual_cross_attention', 'encoder_hid_dim_type', 'time_embedding_dim', 'attention_type', 'mid_block_only_cross_attention'} was not found in config. Values will be initialized to default values.
{'scaling_factor', 'latents_mean', 'latents_std', 'force_upcast'} was not found in config. Values will be initialized to default values.
{'latents_std', 'scaling_factor', 'latents_mean', 'force_upcast'} was not found in config. Values will be initialized to default values.
{'force_upcast', 'scaling_factor', 'latents_std', 'latents_mean'} was not found in config. Values will be initialized to default values.
{'num_attention_heads', 'resnet_skip_time_act', 'addition_embed_type', 'use_linear_projection', 'dropout', 'addition_time_embed_dim', 'reverse_transformer_layers_per_block', 'addition_embed_type_num_heads', 'class_embeddings_concat', 'mid_block_type', 'dual_cross_attention', 'time_cond_proj_dim', 'num_class_embeds', 'upcast_attention', 'mid_block_only_cross_attention', 'encoder_hid_dim_type', 'attention_type', 'class_embed_type', 'encoder_hid_dim', 'transformer_layers_per_block', 'conv_in_kernel', 'projection_class_embeddings_input_dim', 'only_cross_attention', 'resnet_out_scale_factor', 'time_embedding_act_fn', 'resnet_time_scale_shift', 'conv_out_kernel', 'time_embedding_dim', 'time_embedding_type', 'cross_attention_norm', 'timestep_post_act'} was not found in config. Values will be initialized to default values.
{'attention_type', 'resnet_out_scale_factor', 'cross_attention_norm', 'timestep_post_act', 'dropout', 'mid_block_type', 'time_cond_proj_dim', 'transformer_layers_per_block', 'class_embeddings_concat', 'encoder_hid_dim_type', 'resnet_skip_time_act', 'projection_class_embeddings_input_dim', 'dual_cross_attention', 'encoder_hid_dim', 'num_class_embeds', 'upcast_attention', 'addition_time_embed_dim', 'conv_in_kernel', 'use_linear_projection', 'time_embedding_act_fn', 'reverse_transformer_layers_per_block', 'mid_block_only_cross_attention', 'time_embedding_dim', 'class_embed_type', 'conv_out_kernel', 'resnet_time_scale_shift', 'addition_embed_type_num_heads', 'num_attention_heads', 'only_cross_attention', 'time_embedding_type', 'addition_embed_type'} was not found in config. Values will be initialized to default values.
{'transformer_layers_per_block', 'conv_out_kernel', 'addition_embed_type', 'addition_embed_type_num_heads', 'projection_class_embeddings_input_dim', 'attention_type', 'cross_attention_norm', 'time_cond_proj_dim', 'mid_block_type', 'time_embedding_type', 'reverse_transformer_layers_per_block', 'resnet_out_scale_factor', 'use_linear_projection', 'dual_cross_attention', 'dropout', 'mid_block_only_cross_attention', 'encoder_hid_dim_type', 'time_embedding_act_fn', 'resnet_skip_time_act', 'addition_time_embed_dim', 'time_embedding_dim', 'encoder_hid_dim', 'class_embeddings_concat', 'resnet_time_scale_shift', 'only_cross_attention', 'num_attention_heads', 'conv_in_kernel', 'upcast_attention', 'class_embed_type', 'timestep_post_act', 'num_class_embeds'} was not found in config. Values will be initialized to default values.
{'attention_type', 'resnet_out_scale_factor', 'cross_attention_norm', 'timestep_post_act', 'dropout', 'mid_block_type', 'time_cond_proj_dim', 'transformer_layers_per_block', 'class_embeddings_concat', 'encoder_hid_dim_type', 'resnet_skip_time_act', 'projection_class_embeddings_input_dim', 'dual_cross_attention', 'encoder_hid_dim', 'num_class_embeds', 'upcast_attention', 'addition_time_embed_dim', 'conv_in_kernel', 'use_linear_projection', 'time_embedding_act_fn', 'reverse_transformer_layers_per_block', 'mid_block_only_cross_attention', 'time_embedding_dim', 'class_embed_type', 'conv_out_kernel', 'resnet_time_scale_shift', 'addition_embed_type_num_heads', 'num_attention_heads', 'only_cross_attention', 'time_embedding_type', 'addition_embed_type'} was not found in config. Values will be initialized to default values.
{'num_attention_heads', 'resnet_skip_time_act', 'addition_embed_type', 'use_linear_projection', 'dropout', 'addition_time_embed_dim', 'reverse_transformer_layers_per_block', 'addition_embed_type_num_heads', 'class_embeddings_concat', 'mid_block_type', 'dual_cross_attention', 'time_cond_proj_dim', 'num_class_embeds', 'upcast_attention', 'mid_block_only_cross_attention', 'encoder_hid_dim_type', 'attention_type', 'class_embed_type', 'encoder_hid_dim', 'transformer_layers_per_block', 'conv_in_kernel', 'projection_class_embeddings_input_dim', 'only_cross_attention', 'resnet_out_scale_factor', 'time_embedding_act_fn', 'resnet_time_scale_shift', 'conv_out_kernel', 'time_embedding_dim', 'time_embedding_type', 'cross_attention_norm', 'timestep_post_act'} was not found in config. Values will be initialized to default values.
{'transformer_layers_per_block', 'conv_out_kernel', 'addition_embed_type', 'addition_embed_type_num_heads', 'projection_class_embeddings_input_dim', 'attention_type', 'cross_attention_norm', 'time_cond_proj_dim', 'mid_block_type', 'time_embedding_type', 'reverse_transformer_layers_per_block', 'resnet_out_scale_factor', 'use_linear_projection', 'dual_cross_attention', 'dropout', 'mid_block_only_cross_attention', 'encoder_hid_dim_type', 'time_embedding_act_fn', 'resnet_skip_time_act', 'addition_time_embed_dim', 'time_embedding_dim', 'encoder_hid_dim', 'class_embeddings_concat', 'resnet_time_scale_shift', 'only_cross_attention', 'num_attention_heads', 'conv_in_kernel', 'upcast_attention', 'class_embed_type', 'timestep_post_act', 'num_class_embeds'} was not found in config. Values will be initialized to default values.
06/20/2024 20:40:22 - INFO - __main__ - ***** Running training *****
06/20/2024 20:40:22 - INFO - __main__ -   Num examples = 833
06/20/2024 20:40:22 - INFO - __main__ -   Num Epochs = 101
06/20/2024 20:40:22 - INFO - __main__ -   Instantaneous batch size per device = 8
06/20/2024 20:40:22 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 1024
06/20/2024 20:40:22 - INFO - __main__ -   Gradient Accumulation steps = 4
06/20/2024 20:40:22 - INFO - __main__ -   Total optimization steps = 101
Steps:   0%|          | 0/101 [00:00<?, ?it/s]/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
Steps:   0%|          | 0/101 [00:00<?, ?it/s]/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
Steps:   0%|          | 0/101 [00:00<?, ?it/s]/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
Steps:   0%|          | 0/101 [00:00<?, ?it/s]/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:338: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  and inp.query.storage().data_ptr() == inp.key.storage().data_ptr()
Steps:   0%|          | 0/101 [00:33<?, ?it/s, lr=1e-5, step_loss=0.0268]/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
Steps:   0%|          | 0/101 [00:32<?, ?it/s, lr=1e-5, step_loss=0.0311]/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
Steps:   0%|          | 0/101 [00:30<?, ?it/s, lr=1e-5, step_loss=0.0278]/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
Steps:   0%|          | 0/101 [00:32<?, ?it/s, lr=1e-5, step_loss=0.0257]/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
/mnt/cache/liuyipeng/.conda/envs/sd_train/lib/python3.10/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
Steps:   1%|          | 1/101 [00:33<56:08, 33.68s/it, lr=1e-5, step_loss=0.0476]06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
Steps:   1%|          | 1/101 [00:35<59:34, 35.74s/it, lr=1e-5, step_loss=0.0496]06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
Steps:   1%|          | 1/101 [00:35<59:28, 35.68s/it, lr=1e-5, step_loss=0.0434]06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
Steps:   1%|          | 1/101 [00:35<59:34, 35.75s/it, lr=1e-5, step_loss=0.0428]06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
06/20/2024 20:40:56 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
Steps:  98%|████████�elapsed_time:{}4<00:16,  8.29s/it, lr=1e-5, step_loss=0.0361]]oss=0.0232]405]0.0184]oss=0.0333]4]45] step_loss=0.0369].0174]0196]
 866.6380344219797
elapsed_time:{}
elapsed_time:{}
elapsed_time:{}
 866.6167547570076
 866.6386504959955
elapsed_time:{}
 866.627238985151
elapsed_time:{}
elapsed_time:{}
 866.7564153652638
elapsed_time:{}
elapsed_time:{}
 866.6157430419698
elapsed_time:{}
  866.6249729509866.6511749930214elapsed_time:{}


 866.9944244362414
 866.4238834618591
elapsed_time:{}
 866.6385778060067elapsed_time:{}

 866.6161850308999
 866.6382480629836
elapsed_time:{}
elapsed_time:{}
 866.6613488829753
elapsed_time:{}
 866.6388123740035
 865.5202609989792
elapsed_time:{}
elapsed_time:{}
elapsed_time:{}
 866.6163311819546
 866.6386015549942
 867.5679918788373elapsed_time:{}
 867.1083981897682

elapsed_time:{}
elapsed_time:{}
 866.6161276500206
elapsed_time:{}
 866.6266785811167
 866.6161461300217
elapsed_time:{}
elapsed_time:{}
elapsed_time:{}
 866.661210500868
elapsed_time:{}
 866.873313786462
elapsed_time:{}
 864.5553235230036
 866.6275836660061
 866.6168938609771
elapsed_time:{}
elapsed_time:{}
 866.6274600799661
 867.3286428637803
elapsed_time:{}
elapsed_time:{}
 867.3791535068303
 866.6269400171004
Steps: 100%|██████████| 101/101 [14:27<00:00,  8.59s/it, lr=1e-5, step_loss=0.0482]
Steps: 100%|██████████| 101/101 [14:26<00:00,  8.58s/it, lr=1e-5, step_loss=0.0422]
Steps: 100%|██████████| 101/101 [14:26<00:00,  8.58s/it, lr=1e-5, step_loss=0.0374]

```

## 训练目标
### 以下数据基于1424集群A100（IB:200），仅作参考

### 功能测试目标
#### 单机8卡
stable-diffusion-v1-4 训练 `step > 100` ，Loss小于 `0.0456`。

#### 4机32卡
stable-diffusion-v1-5 训练 `step > 5000` ，Loss小于 `0.015`。
stable-diffusion-v2-1 训练 `step > 5000` ，Loss小于 `0.15`。

### 性能测试目标
#### 单机8卡
stable-diffusion-v1-4 step均值 `2.040(s/it)` ，IPS `15.69`。
stable-diffusion-v1-5 step均值 `6.861(s/it)` ，IPS `4.664`。
stable-diffusion-v2-1 step均值 `6.099(s/it)` ，IPS `5.246`。

#### 4机32卡
stable-diffusion-v1-5 step均值 `8.574(s/it)` ，IPS `3.732`。
stable-diffusion-v2-1 step均值 `7.145(s/it)` ，IPS `4.478`。
