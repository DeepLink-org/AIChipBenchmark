# Stable Diffusion 训练


## 准备工作

- 代码下载：https://github.com/huggingface/diffusers/tree/v0.26.3
- 安装：https://github.com/huggingface/diffusers/tree/v0.26.3/examples/text_to_image#installing-the-dependencies
- 环境依赖：`torch==2.1.0`, `transformers==4.31.0`, `diffusers==v0.26.3`, `datasets==2.18.0`等。
- 模型权重：stable diffusion v1-4 https://huggingface.co/CompVis/stable-diffusion-v1-4
- 数据集：https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions



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
- 单机8卡


## 启动及数据采集

启动命令：

```bash
accelerate launch  --multi_gpu  --mixed_precision="fp16" --config_file default_config.yaml train_text_to_image.py   --pretrained_model_name_or_path=$MODEL_NAME   --dataset_name=$DATASET_NAME   --use_ema   --resolution=512 --center_crop --random_flip   --train_batch_size=8   --gradient_accumulation_steps=4 --max_train_steps=101   --learning_rate=1e-05   --max_grad_norm=1   --lr_scheduler="constant" --lr_warmup_steps=0   --output_dir="sd-pokemon-model"
```



关键参数说明：
- train_batch_size：per device batch size，设置为8
- gradient_accumulation_steps： 梯度累计，设置为4
- max_train_steps: 性能迭代次数，设置为101，为了消除最后一个step的影响，`time(s/it)`取第100个step即可
- gradient checkpointing：处于性能考虑，关闭梯度检查点功能


性能指标`IPS`计算：
- dp：基准数据使用单机8卡，因此`dp=8`
- time: 每次迭代时间，可从日志中得到
- gbsz = train_batch_size * gradient_accumulation_steps * dp。因此本例为`256`
- IPS = gbsz / time / #gpu


### 日志参考
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
dkx, len(train_dataloader) 14
dkx, num_update_steps_per_epoch 4
03/06/2024 17:32:14 - INFO - __main__ - ***** Running training *****
03/06/2024 17:32:14 - INFO - __main__ -   Num examples = 833
03/06/2024 17:32:14 - INFO - __main__ -   Num Epochs = 26
03/06/2024 17:32:14 - INFO - __main__ -   Instantaneous batch size per device = 8
03/06/2024 17:32:14 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 256
03/06/2024 17:32:14 - INFO - __main__ -   Gradient Accumulation steps = 4
03/06/2024 17:32:14 - INFO - __main__ -   Total optimization steps = 101
dkx, len(train_dataloader) 14
dkx, num_update_steps_per_epoch 4
dkx, len(train_dataloader) 14
dkx, num_update_steps_per_epoch 4
dkx, len(train_dataloader) 14
dkx, num_update_steps_per_epoch 4
dkx, len(train_dataloader) 14
dkx, num_update_steps_per_epoch 4
dkx, len(train_dataloader) 14
dkx, num_update_steps_per_epoch 4
dkx, len(train_dataloader) 14
dkx, num_update_steps_per_epoch 4
dkx, len(train_dataloader) 14
dkx, num_update_steps_per_epoch 4

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


Loading pipeline components...:   0%|          | 0/7 [00:00<?, ?it/s][ALoaded tokenizer as CLIPTokenizer from `tokenizer` subfolder of /mnt/lustrenew/share_data/PAT/datasets/StableDiffusion/stable-diffusion-v1-4.
{'prediction_type', 'timestep_spacing'} was not found in config. Values will be initialized to default values.
Loaded scheduler as PNDMScheduler from `scheduler` subfolder of /mnt/lustrenew/share_data/PAT/datasets/StableDiffusion/stable-diffusion-v1-4.
Loaded feature_extractor as CLIPImageProcessor from `feature_extractor` subfolder of /mnt/lustrenew/share_data/PAT/datasets/StableDiffusion/stable-diffusion-v1-4.
`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["id2label"]` will be overriden.
`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["bos_token_id"]` will be overriden.
`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The value `text_config["eos_token_id"]` will be overriden.
Loaded safety_checker as StableDiffusionSafetyChecker from `safety_checker` subfolder of /mnt/lustrenew/share_data/PAT/datasets/StableDiffusion/stable-diffusion-v1-4.


Loading pipeline components...:  86%|████████▌ | 6/7 [00:00<00:00, 16.17it/s][A
Loading pipeline components...: 100%|██████████| 7/7 [00:00<00:00, 18.85it/s]
Configuration saved in /mnt/lustrenew/dongkaixing1.vendor/sd-pokemon-model-test/vae/config.json
Model weights saved in /mnt/lustrenew/dongkaixing1.vendor/sd-pokemon-model-test/vae/diffusion_pytorch_model.safetensors
Configuration saved in /mnt/lustrenew/dongkaixing1.vendor/sd-pokemon-model-test/unet/config.json
Model weights saved in /mnt/lustrenew/dongkaixing1.vendor/sd-pokemon-model-test/unet/diffusion_pytorch_model.safetensors
Configuration saved in /mnt/lustrenew/dongkaixing1.vendor/sd-pokemon-model-test/scheduler/scheduler_config.json
Configuration saved in /mnt/lustrenew/dongkaixing1.vendor/sd-pokemon-model-test/model_index.json

Steps: 100%|██████████| 101/101 [03:32<00:00,  2.11s/it, lr=1e-5, step_loss=0.0456]

```

## 训练目标
训练`step = 101` ，Loss小于 `0.0456`。


