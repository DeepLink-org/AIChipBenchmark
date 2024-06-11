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


## 配置
使用`accelerate`辅助进行启动训练，`default_config.yaml`文件内容参考：
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

```
```bash
# 集群srun 以上sd_train.sh脚本
# 单机8卡
srun -p pat_rd -n1 -N1 --gres=gpu:8 bash sd_train.sh
# 4机32卡
srun -p pat_rd -n4 -N4 --gres=gpu:8 bash sd_train.sh

```

关键参数说明：
- train_batch_size：per device batch size，设置为8
- gradient_accumulation_steps：梯度累计，设置为4
- resolution: 分辨率，多机多卡在性能测试时设置为960; 功能测试时设置为256，loss下降较为明显。单机8卡下使用默认配置512
- max_train_steps: 性能迭代次数，设置为100，为了消除step波动的影响，`time(s/it)`取100个step的均值。
- gradient checkpointing：处于性能考虑，关闭梯度检查点功能
- 模型选择：[参考] (## 训练目标)

性能指标`IPS`计算：
- dp：基准数据使用单机8卡时，`dp=8`； 使用4机32卡时，`dp=32`
- time: 每次迭代时间，可从日志中得到
- gbsz = train_batch_size * gradient_accumulation_steps * dp。
- IPS = gbsz / time / #gpu


### 日志参考

#### 单机8卡

```
03/06/2024 17:32:04 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 1
Local process index: 1
Device: cuda:1

Mixed precision type: fp16

03/06/2024 17:32:04 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 2
Local process index: 2
Device: cuda:2

Mixed precision type: fp16

Detected kernel version 3.10.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
03/06/2024 17:32:04 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 0
Local process index: 0
Device: cuda:0

Mixed precision type: fp16

{'dynamic_thresholding_ratio', 'timestep_spacing', 'sample_max_value', 'variance_type', 'thresholding', 'prediction_type', 'rescale_betas_zero_snr', 'clip_sample_range'} was not found in config. Values will be initialized to default values.
03/06/2024 17:32:04 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 5
Local process index: 5
Device: cuda:5

Mixed precision type: fp16

03/06/2024 17:32:04 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 6
Local process index: 6
Device: cuda:6

Mixed precision type: fp16

03/06/2024 17:32:04 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 4
Local process index: 4
Device: cuda:4

Mixed precision type: fp16

03/06/2024 17:32:04 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 7
Local process index: 7
Device: cuda:7

Mixed precision type: fp16

03/06/2024 17:32:04 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 3
Local process index: 3
Device: cuda:3

Mixed precision type: fp16

{'force_upcast', 'norm_num_groups'} was not found in config. Values will be initialized to default values.
{'class_embeddings_concat', 'num_attention_heads', 'time_cond_proj_dim', 'time_embedding_type', 'resnet_skip_time_act', 'num_class_embeds', 'conv_out_kernel', 'upcast_attention', 'timestep_post_act', 'use_linear_projection', 'time_embedding_act_fn', 'cross_attention_norm', 'addition_time_embed_dim', 'addition_embed_type', 'time_embedding_dim', 'only_cross_attention', 'resnet_out_scale_factor', 'reverse_transformer_layers_per_block', 'encoder_hid_dim', 'class_embed_type', 'projection_class_embeddings_input_dim', 'addition_embed_type_num_heads', 'mid_block_type', 'resnet_time_scale_shift', 'conv_in_kernel', 'dropout', 'mid_block_only_cross_attention', 'dual_cross_attention', 'encoder_hid_dim_type', 'transformer_layers_per_block', 'attention_type'} was not found in config. Values will be initialized to default values.
{'class_embeddings_concat', 'num_attention_heads', 'time_cond_proj_dim', 'time_embedding_type', 'resnet_skip_time_act', 'num_class_embeds', 'conv_out_kernel', 'upcast_attention', 'timestep_post_act', 'use_linear_projection', 'time_embedding_act_fn', 'cross_attention_norm', 'addition_time_embed_dim', 'addition_embed_type', 'time_embedding_dim', 'only_cross_attention', 'resnet_out_scale_factor', 'reverse_transformer_layers_per_block', 'encoder_hid_dim', 'class_embed_type', 'projection_class_embeddings_input_dim', 'addition_embed_type_num_heads', 'mid_block_type', 'resnet_time_scale_shift', 'conv_in_kernel', 'dropout', 'mid_block_only_cross_attention', 'dual_cross_attention', 'encoder_hid_dim_type', 'transformer_layers_per_block', 'attention_type'} was not found in config. Values will be initialized to default values.
03/06/2024 17:32:14 - INFO - __main__ - ***** Running training *****
03/06/2024 17:32:14 - INFO - __main__ -   Num examples = 833
03/06/2024 17:32:14 - INFO - __main__ -   Num Epochs = 26
03/06/2024 17:32:14 - INFO - __main__ -   Instantaneous batch size per device = 8
03/06/2024 17:32:14 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 256
03/06/2024 17:32:14 - INFO - __main__ -   Gradient Accumulation steps = 4
03/06/2024 17:32:14 - INFO - __main__ -   Total optimization steps = 101

Steps:   0%|          | 0/101 [00:00<?, ?it/s]
Steps:   0%|          | 0/101 [00:03<?, ?it/s, lr=1e-5, step_loss=0.0799]
Steps:   0%|          | 0/101 [00:03<?, ?it/s, lr=1e-5, step_loss=0.0223]
Steps:   0%|          | 0/101 [00:04<?, ?it/s, lr=1e-5, step_loss=0.0609]
Steps:   1%|          | 1/101 [00:04<08:00,  4.80s/it, lr=1e-5, step_loss=0.0609]
Steps:   1%|          | 1/101 [00:04<08:00,  4.80s/it, lr=1e-5, step_loss=0.0603]
Steps:   1%|          | 1/101 [00:05<08:00,  4.80s/it, lr=1e-5, step_loss=0.148] 
Steps:   1%|          | 1/101 [00:06<08:00,  4.80s/it, lr=1e-5, step_loss=0.0432]
Steps:   1%|          | 1/101 [00:06<08:00,  4.80s/it, lr=1e-5, step_loss=0.135] 
Steps:   2%|▏         | 2/101 [00:07<05:40,  3.44s/it, lr=1e-5, step_loss=0.135]
Steps:   2%|▏         | 2/101 [00:07<05:40,  3.44s/it, lr=1e-5, step_loss=0.0375]
Steps:   2%|▏         | 2/101 [00:07<05:40,  3.44s/it, lr=1e-5, step_loss=0.0945]
Steps:   2%|▏         | 2/101 [00:08<05:40,  3.44s/it, lr=1e-5, step_loss=0.0359]
Steps:   2%|▏         | 2/101 [00:08<05:40,  3.44s/it, lr=1e-5, step_loss=0.0337]


Steps:  99%|█████████▉| 100/101 [03:22<00:01,  1.79s/it, lr=1e-5, step_loss=0.0925]
Steps:  99%|█████████▉| 100/101 [03:22<00:01,  1.79s/it, lr=1e-5, step_loss=0.0535]
Steps:  99%|█████████▉| 100/101 [03:23<00:01,  1.79s/it, lr=1e-5, step_loss=0.0323]
Steps: 100%|██████████| 101/101 [03:23<00:00,  1.98s/it, lr=1e-5, step_loss=0.0323]
Steps: 100%|██████████| 101/101 [03:23<00:00,  1.98s/it, lr=1e-5, step_loss=0.0456]{'image_encoder', 'requires_safety_checker'} was not found in config. Values will be initialized to default values.


Loading pipeline components...:   0%|          | 0/7 [00:00<?, ?it/s]�[ALoaded tokenizer as CLIPTokenizer from `tokenizer` subfolder of /mnt/lustrenew/share_data/PAT/datasets/StableDiffusion/stable-diffusion-v1-4.
{'prediction_type', 'timestep_spacing'} was not found in config. Values will be initialized to default values.
Loaded scheduler as PNDMScheduler from `scheduler` subfolder of /mnt/lustrenew/share_data/PAT/datasets/StableDiffusion/stable-diffusion-v1-4.
Loaded feature_extractor as CLIPImageProcessor from `feature_extractor` subfolder of /mnt/lustrenew/share_data/PAT/datasets/StableDiffusion/stable-diffusion-v1-4.
`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["id2label"]` will be overriden.
`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["bos_token_id"]` will be overriden.
`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["eos_token_id"]` will be overriden.
Loaded safety_checker as StableDiffusionSafetyChecker from `safety_checker` subfolder of /mnt/lustrenew/share_data/PAT/datasets/StableDiffusion/stable-diffusion-v1-4.


Loading pipeline components...:  86%|████████▌ | 6/7 [00:00<00:00, 16.17it/s]�[A
Loading pipeline components...: 100%|██████████| 7/7 [00:00<00:00, 18.85it/s]
Configuration saved in /mnt/lustrenew/dongkaixing1.vendor/sd-pokemon-model-test/vae/config.json
Model weights saved in /mnt/lustrenew/dongkaixing1.vendor/sd-pokemon-model-test/vae/diffusion_pytorch_model.safetensors
Configuration saved in /mnt/lustrenew/dongkaixing1.vendor/sd-pokemon-model-test/unet/config.json
Model weights saved in /mnt/lustrenew/dongkaixing1.vendor/sd-pokemon-model-test/unet/diffusion_pytorch_model.safetensors
Configuration saved in /mnt/lustrenew/dongkaixing1.vendor/sd-pokemon-model-test/scheduler/scheduler_config.json
Configuration saved in /mnt/lustrenew/dongkaixing1.vendor/sd-pokemon-model-test/model_index.json

Steps: 100%|██████████| 101/101 [03:32<00:00,  2.11s/it, lr=1e-5, step_loss=0.0456]

```
#### 4机32卡
```
03/20/2024 20:36:46 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 1
Local process index: 1
Device: cuda:1

Mixed precision type: fp16

03/20/2024 20:36:46 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 6
Local process index: 6
Device: cuda:6

Mixed precision type: fp16

03/20/2024 20:36:46 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 5
Local process index: 5
Device: cuda:5

Mixed precision type: fp16

03/20/2024 20:36:46 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 3
Local process index: 3
Device: cuda:3

Mixed precision type: fp16

03/20/2024 20:36:46 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 4
Local process index: 4
Device: cuda:4

Mixed precision type: fp16

03/20/2024 20:36:46 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 7
Local process index: 7
Device: cuda:7

Mixed precision type: fp16

03/20/2024 20:36:46 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 2
Local process index: 2
Device: cuda:2
                        

Mixed precision type: fp16

03/20/2024 20:36:46 - INFO - __main__ - Distributed environment: MULTI_GPU  Backend: nccl
Num processes: 8
Process index: 0
Local process index: 0
Device: cuda:0

Mixed precision type: fp16

{'thresholding', 'dynamic_thresholding_ratio', 'timestep_spacing', 'sample_max_value', 'rescale_betas_zero_snr', 'clip_sample_range', 'variance_type'} was not found in config. Values will be initialized to default values.
{'latents_mean', 'scaling_factor', 'latents_std', 'force_upcast'} was not found in config. Values will be initialized to default values.
{'mid_block_type', 'attention_type', 'reverse_transformer_layers_per_block', 'encoder_hid_dim_type', 'resnet_skip_time_act', 'addition_embed_type', 'conv_out_kernel', 'cross_attention_norm', 'resnet_out_scale_factor', 'encoder_hid_dim', 'time_embedding_type', 'time_embedding_act_fn', 'resnet_time_scale_shift', 'mid_block_only_cross_attention', 'conv_in_kernel', 'timestep_post_act', 'num_attention_heads', 'class_embed_type', 'projection_class_embeddings_input_dim', 'time_cond_proj_dim', 'transformer_layers_per_block', 'time_embedding_dim', 'dropout', 'addition_embed_type_num_heads', 'addition_time_embed_dim', 'class_embeddings_concat'} was not found in config. Values will be initialized to default values.
{'mid_block_type', 'attention_type', 'reverse_transformer_layers_per_block', 'encoder_hid_dim_type', 'resnet_skip_time_act', 'addition_embed_type', 'conv_out_kernel', 'cross_attention_norm', 'resnet_out_scale_factor', 'encoder_hid_dim', 'time_embedding_type', 'time_embedding_act_fn', 'resnet_time_scale_shift', 'mid_block_only_cross_attention', 'conv_in_kernel', 'timestep_post_act', 'num_attention_heads', 'class_embed_type', 'projection_class_embeddings_input_dim', 'time_cond_proj_dim', 'transformer_layers_per_block', 'time_embedding_dim', 'dropout', 'addition_embed_type_num_heads', 'addition_time_embed_dim', 'class_embeddings_concat'} was not found in config. Values will be initialized to default values.
args.gradient_checkpointing =  False
args.gradient_checkpointing =  False
args.gradient_checkpointing =  False
args.gradient_checkpointing =  False
args.gradient_checkpointing =  False
args.gradient_checkpointing =  False
args.gradient_checkpointing =  False
args.gradient_checkpointing =  False
^MGenerating train split: 0 examples [00:00, ? examples/s]^MGenerating train split: 833 examples [00:00, 2328.42 examples/s]^MGenerating train split: 833 examples [00:00, 2322.70 examples/s]
Parameter 'transform'=<function main.<locals>.preprocess_train at 0x7fe1afcefc70> of the transform datasets.arrow_dataset.Dataset.set_format couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.
03/20/2024 20:37:04 - WARNING - datasets.fingerprint - Parameter 'transform'=<function main.<locals>.preprocess_train at 0x7fe1afcefc70> of the transform datasets.arrow_dataset.Dataset.set_format couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.
wandb: Tracking run with wandb version 0.16.4
wandb: W&B syncing is set to `offline` in this directory.
wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
03/20/2024 20:37:17 - INFO - __main__ - ***** Running training *****
03/20/2024 20:37:17 - INFO - __main__ -   Num examples = 833
03/20/2024 20:37:17 - INFO - __main__ -   Num Epochs = 25
03/20/2024 20:37:17 - INFO - __main__ -   Instantaneous batch size per device = 8
03/20/2024 20:37:17 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 256
03/20/2024 20:37:17 - INFO - __main__ -   Gradient Accumulation steps = 4
03/20/2024 20:37:17 - INFO - __main__ -   Total optimization steps = 100
^MSteps:   0%|          | 0/100 [00:00<?, ?it/s]//mnt/afs/huayil/python_packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
//mnt/afs/huayil/python_packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations

  warnings.warn(                                                                                                                                                                                       ^MSteps:   1%|          | 1/100 [00:14<24:07, 14.63s/it]//mnt/afs/huayil/python_packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(                                                                                                                                                                                       //mnt/afs/huayil/python_packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(                                                                                                                                                                                       //mnt/afs/huayil/python_packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(                                                                                                                                                                                       //mnt/afs/huayil/python_packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(                                                                                                                                                                                       //mnt/afs/huayil/python_packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations                                                                                                                                                                      warnings.warn(                                                                                                                                                                                       //mnt/afs/huayil/python_packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
  warnings.warn(
03/20/2024 20:37:32 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
03/20/2024 20:37:32 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
03/20/2024 20:37:32 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
03/20/2024 20:37:32 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
03/20/2024 20:37:32 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
03/20/2024 20:37:32 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
03/20/2024 20:37:32 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
03/20/2024 20:37:32 - INFO - torch.nn.parallel.distributed - Reducer buckets have been rebuilt in this iteration.
^MSteps:   2%|▏         | 2/100 [00:26<21:13, 12.99s/it]^MSteps:   3%|▎         | 3/100 [00:38<20:01, 12.39s/it]^MSteps:   4%|▍         | 4/100 [00:44<15:53,  9.94s/it]^MSteps:   5%|▌         | 5/100 [00:57<17:35, 11.12s/it]^MSteps:   6%|▌         | 6/100 [01:09<17:41, 11.29s/it]^MSteps:   7%|▋         | 7/100 [01:20<17:40, 11.41s/it]^MSteps:   8%|▊         | 8/100 [01:26<14:55,  9.73s/it]^MSteps:   9%|▉         | 9/100 [01:40<16:28, 10.86s/it]^MSteps:  10%|█         | 10/100 [01:51<16:39, 11.10s/it]^MSteps:  11%|█         | 11/100 [02:03<16:43, 11.27s/it]^MSteps:  12%|█▏        | 12/100 [02:09<14:15,  9.72s/it]^MSteps:  13%|█▎        | 13/100 [02:23<15:42, 10.83s/it]^MSteps:  14%|█▍        | 14/100 [02:34<15:48, 11.03s/it]^MSteps:  15%|█▌        | 15/100 [02:46<15:55, 11.25s/it]^MSteps:  16%|█▌        | 16/100 [02:52<13:36,  9.72s/it]^MSteps:  17%|█▋        | 17/100 [03:05<14:56, 10.80s/it]^MSteps:  18%|█▊        | 18/100 [03:17<15:06, 11.05s/it]^MSteps:  19%|█▉        | 19/100 [03:30<15:15, 11.30s/it]^MSteps:  20%|██        | 20/100 [03:35<12:57,  9.72s/it]^MSteps:  21%|██        | 21/100 [03:48<14:12, 10.79s/it]^MSteps:  22%|██▏       | 22/100 [03:59<14:19, 11.02s/it]^MSteps:  23%|██▎       | 23/100 [04:15<14:29, 11.30s/it]^MSteps:  24%|██▍       | 24/100 [04:18<12:18,  9.71s/it]^MSteps:  25%|██▌       | 25/100 [04:35<13:38, 10.91s/it]^MSteps:  26%|██▌       | 26/100 [04:43<13:38, 11.06s/it]^MSteps:  27%|██▋       | 27/100 [04:58<13:44, 11.30s/it]^MSteps:  28%|██▊       | 28/100 [05:04<11:46,  9.82s/it]^MSteps:  29%|██▉       | 29/100 [05:14<12:46, 10.79s/it]^MSteps:  30%|███       | 30/100 [05:30<13:00, 11.15s/it]^MSteps:  31%|███       | 31/100 [05:37<12:55, 11.24s/it]^MSteps:  32%|███▏      | 32/100 [05:43<11:01,  9.72s/it]^MSteps:  33%|███▎      | 33/100 [06:01<12:13, 10.95s/it]^MSteps:  34%|███▍      | 34/100 [06:08<12:10, 11.06s/it]^MSteps:  35%|███▌      | 35/100 [06:25<12:16, 11.34s/it]^MSteps:  36%|███▌      | 36/100 [06:31<10:31,  9.86s/it]^MSteps:  37%|███▋      | 37/100 [06:45<11:30, 10.96s/it]^MSteps:  38%|███▊      | 38/100 [06:50<11:22, 11.01s/it]^MSteps:  39%|███▉      | 39/100 [07:03<11:27, 11.28s/it]^MSteps:  40%|████      | 40/100 [07:14<09:50,  9.84s/it]^MSteps:  41%|████      | 41/100 [07:23<10:39, 10.84s/it]^MSteps:  42%|████▏     | 42/100 [07:39<10:46, 11.15s/it]^MSteps:  43%|████▎     | 43/100 [07:46<10:42, 11.27s/it]^MSteps:  44%|████▍     | 44/100 [07:52<09:06,  9.75s/it]^MSteps:  45%|████▌     | 45/100 [08:11<09:59, 10.89s/it]^MSteps:  46%|████▌     | 46/100 [08:17<10:00, 11.11s/it]^MSteps:  47%|████▋     | 47/100 [08:34<09:57, 11.28s/it]^MSteps:  48%|████▊     | 48/100 [08:40<08:29,  9.80s/it]^MSteps:  49%|████▉     | 49/100 [08:49<09:13, 10.86s/it]^MSteps:  50%|█████     | 50/100 [09:06<09:17, 11.14s/it]^MSteps:  51%|█████     | 51/100 [09:12<09:12, 11.28s/it]^MSteps:  52%|█████▏    | 52/100 [09:18<07:48,  9.76s/it]^MSteps:  53%|█████▎    | 53/100 [09:34<08:31, 10.88s/it]^MSteps:  54%|█████▍    | 54/100 [09:43<08:29, 11.07s/it]^MSteps:  55%|█████▌    | 55/100 [10:01<08:28, 11.31s/it]^MSteps:  56%|█████▌    | 56/100 [10:07<07:12,  9.83s/it]^MSteps:  57%|█████▋    | 57/100 [10:20<07:49, 10.92s/it]^MSteps:  58%|█████▊    | 58/100 [10:32<07:47, 11.13s/it]^MSteps:  59%|█████▉    | 59/100 [10:38<07:41, 11.25s/it]^MSteps:  60%|██████    | 60/100 [10:50<06:32,  9.80s/it]^MSteps:  61%|██████    | 61/100 [10:57<07:02, 10.83s/it]^MSteps:  62%|██████▏   | 62/100 [11:15<07:03, 11.14s/it]^MSteps:  63%|██████▎   | 63/100 [11:21<06:56, 11.27s/it]^MSteps:  64%|██████▍   | 64/100 [11:27<05:51,  9.75s/it]^MSteps:  65%|██████▌   | 65/100 [11:47<06:21, 10.90s/it]^MSteps:  66%|██████▌   | 66/100 [11:55<06:17, 11.10s/it]^MSteps:  67%|██████▋   | 67/100 [12:10<06:12, 11.30s/it]^MSteps:  68%|██████▊   | 68/100 [12:16<05:14,  9.82s/it]^MSteps:  69%|██████▉   | 69/100 [12:30<05:39, 10.94s/it]^MSteps:  70%|███████   | 70/100 [12:42<05:34, 11.16s/it]^MSteps:  71%|███████   | 71/100 [12:50<05:27, 11.29s/it]^MSteps:  72%|███████▏  | 72/100 [13:00<04:34,  9.82s/it]^MSteps:  73%|███████▎  | 73/100 [13:09<04:53, 10.89s/it]^MSteps:  74%|███████▍  | 74/100 [13:25<04:48, 11.11s/it]^MSteps:  75%|███████▌  | 75/100 [13:33<04:41, 11.28s/it]^MSteps:  76%|███████▌  | 76/100 [13:39<03:54,  9.78s/it]^MSteps:  77%|███████▋  | 77/100 [13:56<04:11, 10.94s/it]^MSteps:  78%|███████▊  | 78/100 [14:04<04:04, 11.11s/it]^MSteps:  79%|███████▉  | 79/100 [14:19<03:57, 11.30s/it]^MSteps:  80%|████████  | 80/100 [14:26<03:16,  9.81s/it]^MSteps:  81%|████████  | 81/100 [14:39<03:27, 10.94s/it]^MSteps:   82%|████████▏ | 82/100 [14:44<03:19, 11.09s/it]^MSteps:  83%|████████▎ | 83/100 [15:03<03:12, 11.31s/it]^MSteps:  84%|████████▍ | 84/100 [15:09<02:37,  9.85s/it]^MSteps:  85%|████████▌ | 85/100 [15:23<02:44, 10.97s/it]^MSteps:  86%|████████▌ | 86/100 [15:34<02:36, 11.18s/it]^MSteps:  87%|████████▋ | 87/100 [15:38<02:26, 11.27s/it]^MSteps:  88%|████████▊ | 88/100 [15:53<01:58,  9.87s/it]^MSteps:  89%|████████▉ | 89/100 [16:06<02:00, 10.98s/it]^MSteps:  90%|█████████ | 90/100 [16:18<01:51, 11.19s/it]^MSteps:  91%|█████████ | 91/100 [16:30<01:41, 11.33s/it]^MSteps:  92%|█████████▏| 92/100 [16:36<01:18,  9.84s/it]^MSteps:  93%|█████████▎| 93/100 [16:49<01:16, 10.96s/it]^MSteps:  94%|█████████▍| 94/100 [16:53<01:06, 11.11s/it]^MSteps:  95%|█████████▌| 95/100 [17:13<00:56, 11.34s/it]^MSteps:  96%|█████████▌| 96/100 [17:19<00:39,  9.87s/it]^MSteps:  97%|█████████▋| 97/100 [17:33<00:32, 10.97s/it]^MSteps:  98%|█████████▊| 98/100 [17:44<00:22, 11.18s/it]^MSteps:  99%|█████████▉| 99/100 [17:47<00:11, 11.28s/it]elapsed_time:{}
 1085.4250347237103
elapsed_time:{}
 1085.4276695521548
^MSteps: 100%|██████████| 100/100 [18:02<00:00,  9.83s/it]elapsed_time:{}
 1082.981711922679
elapsed_time:{}
 1085.457329112105
elapsed_time:{}
 1085.4297886248678
elapsed_time:{}
 1085.428821219597
elapsed_time:{}
 1085.4304721239023
elapsed_time:{}
 1085.4298149743117
{'image_encoder'} was not found in config. Values will be initialized to default values.

^MLoading pipeline components...:   0%|          | 0/6 [00:00<?, ?it/s]^[[ALoaded feature_extractor as CLIPImageProcessor from `feature_extractor` subfolder of /mnt/afs/liuyipeng/workspace_2/stable-diffusion-2-1.
{'timestep_spacing', 'sample_max_value', 'clip_sample_range', 'dynamic_thresholding_ratio', 'rescale_betas_zero_snr', 'thresholding'} was not found in config. Values will be initialized to default values.
Loaded scheduler as DDIMScheduler from `scheduler` subfolder of /mnt/afs/liuyipeng/workspace_2/stable-diffusion-2-1.
Loaded tokenizer as CLIPTokenizer from `tokenizer` subfolder of /mnt/afs/liuyipeng/workspace_2/stable-diffusion-2-1.
^MLoading pipeline components...: 100%|██████████| 6/6 [00:00<00:00, 88.23it/s]                                           
Configuration saved in /mnt/afs/liuyipeng/workspace_2/output/train/pokemon_2222/vae/config.jsoMoModel weights saved in /mnt/afs/liuyipeng/workspace_2/output/train/pokemon_2222/vae/diffusion_pytorch_model.safetensors
Configuration saved in /mnt/afs/liuyipeng/workspace_2/output/train/pokemon_2222/unet/config.jsoMoModel weights saved in /mnt/afs/liuyipeng/workspace_2/output/train/pokemon_2222/unet/diffusion_pytorch_model.safetensors
Configuration saved in /mnt/afs/liuyipeng/workspace_2/output/train/pokemon_2222/scheduler/scheduler_config.json
Configuration saved in /mnt/afs/liuyipeng/workspace_2/output/train/pokemon_2222/model_index.json
wandb: - 0.000 MB of 0.000 MB uploaded^Mwandb:
wandb:
wandb: Run history:
wandb: train_loss ▇▅▅▂▅▆▂▆▇▆▆▁▆▇▂▇▅▂▆▅▆█▂▆▆▃▅▆▃▆▆▇▆▁▆▆▁█▇▂
wandb:
wandb: Run summary:
wandb: train_loss 0.07285
wandb:
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /mnt/afs/liuyipeng/wandb/offline-run-20240320_203711-3t42745c
wandb: Find logs at: ./wandb/offline-run-20240320_203711-3t42745c/logs
03/20/2024 20:55:21 - WARNING - urllib3.connectionpool - Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'OSError('Tunnel connection failed: 407 Proxy Authentication Required')': /api/4504800232407040/envelope/
03/20/2024 20:55:21 - WARNING - urllib3.connectionpool - Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'OSError('Tunnel connection failed: 407 Proxy Authentication Required')': /api/4504800232407040/envelope/
03/20/2024 20:55:21 - WARNING - urllib3.connectionpool - Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'OSError('Tunnel connection failed: 407 Proxy Authentication Required')': /api/4504800232407040/envelope/
03/20/2024 20:55:22 - WARNING - urllib3.connectionpool - Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'OSError('Tunnel connection failed: 407 Proxy Authentication Required')': /api/4504800232407040/envelope/
03/20/2024 20:55:22 - WARNING - urllib3.connectionpool - Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'OSError('Tunnel connection failed: 407 Proxy Authentication Required')': /api/4504800232407040/envelope/
03/20/2024 20:55:22 - WARNING - urllib3.connectionpool - Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'OSError('Tunnel connection failed: 407 Proxy Authentication Required')': /api/4504800232407040/envelope/
//mnt/afs/huayil/python_packages/wandb/sdk/wandb_run.py:2171: UserWarning: Run (3t42745c) is finished. The call to `_console_raw_callback` will be ignored. Please make sure that you are using an active run.
  lambda data: self._console_raw_callback("stderr", data),
^MSteps: 100%|██████████| 100/100 [18:10<00:00, 10.91s/it

```

## 训练目标

### 功能测试目标
#### 单机8卡
stable-diffusion-v1-4 训练 `step =  100` ，Loss小于 `0.0456`。

#### 4机32卡
stable-diffusion-v1-5 训练 `step > 5000` ，Loss小于 `0.015`。
stable-diffusion-v2-1 训练 `step > 5000` ，Loss小于 `0.15`。

### 性能测试目标

#### 4机32卡
stable-diffusion-v1-5 step均值 `10.804(s/it)` ，IPS `2.962`。
stable-diffusion-v2-1 step均值 `10.724(s/it)` ，IPS `2.985`。
