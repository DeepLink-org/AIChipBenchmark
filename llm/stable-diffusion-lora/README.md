# Stable Diffusion lora finetune


## 准备工作

- 代码下载:  https://github.com/huggingface/diffusers
- 安装： pip install git+https://github.com/huggingface/diffusers
- 环境依赖：`torch==2.1.0`, `transformers==4.38.1`, `diffusers==v0.27.0.dev`, `datasets==2.16.1`等。
- 模型权重：stable diffusion v1-5 https://github.com/runwayml/stable-diffusion
            stable diffusion v2-1 https://github.com/Stability-AI/stablediffusion
- 数据集：https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions



## 配置
使用下载的diffuser库中diffusers/examples/text\_to\_image/train\_text\_to\_image\_lora.py`辅助进行启动训练，：
这里需要对train_text_to_image_lora.py做一些修改以实时显示train_loss并阅读本地的datasets，如果环境可以联网，可以不必修改dataset相关改动
train_text_to_image_lora.py 的diff结果如下, 也可以参考本目录下的train_text_to_image_lora.py

```
iff --git a/examples/text_to_image/train_text_to_image_lora.py b/examples/text_to_image/train_text_to_image_lora.py
index 71b99f15..79c71da4 100644
--- a/examples/text_to_image/train_text_to_image_lora.py
+++ b/examples/text_to_image/train_text_to_image_lora.py
@@ -21,6 +21,7 @@ import math
 import os
 import random
 import shutil
+import time
 from pathlib import Path

 import datasets
@@ -50,12 +51,16 @@ from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_
 from diffusers.utils.import_utils import is_xformers_available
 from diffusers.utils.torch_utils import is_compiled_module

+os.environ["WANDB_DISABLED"]="true"

 # Will error if the minimal version of diffusers is not installed. Remove at your own risks.
-check_min_version("0.28.0.dev0")
+check_min_version("0.27.0.dev0")

 logger = get_logger(__name__, log_level="INFO")

+def fprintf(file_t, format_string, *args):
+    with open(file_t,"a") as f:
+        f.write(format_string.format(*args))

 def save_model_card(
     repo_id: str,
@@ -497,7 +502,7 @@ def main():

             xformers_version = version.parse(xformers.__version__)
             if xformers_version == version.parse("0.0.16"):
-                logger.warning(
+                logger.warn(
                     "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/d
ocs/diffusers/main/en/optimization/xformers for more details."
                 )
             unet.enable_xformers_memory_efficient_attention()
@@ -548,11 +553,10 @@ def main():
     if args.dataset_name is not None:
         # Downloading and loading a dataset from the hub.
         dataset = load_dataset(
-            args.dataset_name,
-            args.dataset_config_name,
-            cache_dir=args.cache_dir,
-            data_dir=args.train_data_dir,
+             "parquet",
+              data_files={'train':"/mnt/afs/huayil/datasets/"+args.dataset_name}
         )
+        #dataset = load_dataset("THUDM/ImageRewardDB", "1k")
     else:
         data_files = {}
         if args.train_data_dir is not None:
@@ -623,11 +627,29 @@ def main():
         model = model._orig_mod if is_compiled_module(model) else model
         return model

-    def preprocess_train(examples):
+    def preprocess_train_(examples):
         images = [image.convert("RGB") for image in examples[image_column]]
         examples["pixel_values"] = [train_transforms(image) for image in images]
         examples["input_ids"] = tokenize_captions(examples)
         return examples
+
+    def preprocess_train(examples):
+        #print(examples[image_column])
+        from io import BytesIO
+        from PIL import Image
+
+        def convert(image_col):
+            bytes_io = BytesIO(image_col['bytes'])
+            with Image.open(bytes_io) as img:
+                rgb = img.convert("RGB")
+            return rgb
+
+
+        images = [convert(image) for image in examples[image_column]]
+        #images = [image.convert("RGB") for image in examples[image_column]]
+        examples["pixel_values"] = [train_transforms(image) for image in images]
+        examples["input_ids"] = tokenize_captions(examples)
+        return examples

     with accelerator.main_process_first():
         if args.max_train_samples is not None:
@@ -728,7 +750,15 @@ def main():
         # Only show the progress bar once on each machine.
         disable=not accelerator.is_local_main_process,
     )
-
+
+    train_loss_log = "./train_loss.log"
+    with open(train_loss_log, "w"):
+        pass
+    train_loss_avg_length = 1
+    train_loss_count = 0
+    avg_train_loss = 0
+
+    start_time = time.perf_counter()
     for epoch in range(first_epoch, args.num_train_epochs):
         unet.train()
         train_loss = 0.0
@@ -809,6 +839,16 @@ def main():
             if accelerator.sync_gradients:
                 progress_bar.update(1)
                 global_step += 1
+
+                train_loss_count += 1
+                avg_train_loss += train_loss
+
+                if train_loss_count % train_loss_avg_length ==0:
+                    avg_train_loss = avg_train_loss/train_loss_avg_length
+                    # when testing train loss, please remove the '#' below
+                    #fprintf(train_loss_log,"train_loss:{}\n", avg_train_loss)
+                    avg_train_loss = 0
+
+    start_time = time.perf_counter()
     for epoch in range(first_epoch, args.num_train_epochs):
         unet.train()
         train_loss = 0.0
@@ -809,6 +839,16 @@ def main():
             if accelerator.sync_gradients:
                 progress_bar.update(1)
                 global_step += 1
+
+                train_loss_count += 1
+                avg_train_loss += train_loss
+
+                if train_loss_count % train_loss_avg_length ==0:
+                    avg_train_loss = avg_train_loss/train_loss_avg_length
+                    # when testing train loss, please remove the '#' below
+                    #fprintf(train_loss_log,"train_loss:{}\n", avg_train_loss)
+                    avg_train_loss = 0
+
                 accelerator.log({"train_loss": train_loss}, step=global_step)
                 train_loss = 0.0

@@ -854,6 +894,9 @@ def main():
             progress_bar.set_postfix(**logs)

             if global_step >= args.max_train_steps:
+                end_time = time.perf_counter()
+                elapsed = end_time - start_time
+                fprintf(train_loss_log,"elapsed_time:{}\n", elapsed)
                 break

         if accelerator.is_main_process:
```

关键配置：
- mixed_precision：开启混合精度fp16
- 单机1卡


## 启动及数据采集

启动命令：
```
#!/bin/sh
export MODEL_NAME="/mnt/afs/huayil/models/stable-diffusion-v1-5"
#export MODEL_NAME="/mnt/afs/liuyipeng/workspace_2/stable-diffusion-2-1"
export OUTPUT_DIR="/mnt/afs/huayil/code/lora/pokemon"
export HUB_MODEL_ID="pokemon-lora"
export DATASET_NAME="pokemon/train-00000-of-00001-566cc9b19d7203f8.parquet"

python /mnt/afs/huayil/code/diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=256 \
  --center_crop \
  --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=5000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb \
  --checkpointing_steps=50000 \
  --validation_epochs=50000 \
  --validation_prompt="A pokemon with blue eyes." \
  --report_to="wandb" \
  --seed=1337 \
  --mixed_precision="fp16" \
  --image_column="image" \
  --caption_column="text"
```
注意，需要按照芯片端到端性能评测方案在功能测试和性能测试时改动对应resolution

accelerate相关的环境变量为
```
Copy-and-paste the text below in your GitHub issue

- `Accelerate` version: 0.27.2
- Platform: Linux-3.10.0-957.el7.x86_64-x86_64-with-glibc2.17
- Python version: 3.10.9
- Numpy version: 1.26.4
- PyTorch version (GPU?): 2.0.0+cu118 (False)
- PyTorch XPU available: False
- PyTorch NPU available: False
- System RAM: 503.36 GB
- `Accelerate` default config:
        Not found
```


关键参数说明：
- train_batch_size：per device batch size，设置为8
- gradient_accumulation_steps： 梯度累计，设置为4
- max_train_steps: 性能迭代次数，设置为2000
- resolution：在功能测试时为256，性能测试时改为960
- gradient checkpointing：处于性能考虑，关闭梯度检查点功能


性能指标`IPS`计算：
- dp：基准数据使用单机1卡，因此`dp=1`
- time: 每次迭代时间，可从日志中得到
- gbsz = train_batch_size * gradient_accumulation_steps * dp。因此本例为`32`
- IPS = gbsz / time / #gpu


### 日志参考
```

```
b pt-uwro96fh submitted successfully, please wait for scheduling!
job pt-uwro96fh scheduled successfully
pt-uwro96fh-worker-0 logs: Detected kernel version 4.18.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
pt-uwro96fh-worker-0 logs: 03/22/2024 15:02:04 - INFO - __main__ - Distributed environment: NO
pt-uwro96fh-worker-0 logs: Num processes: 1
pt-uwro96fh-worker-0 logs: Process index: 0
pt-uwro96fh-worker-0 logs: Local process index: 0
pt-uwro96fh-worker-0 logs: Device: cuda
pt-uwro96fh-worker-0 logs:
pt-uwro96fh-worker-0 logs: Mixed precision type: fp16
pt-uwro96fh-worker-0 logs:
pt-uwro96fh-worker-0 logs: {'sample_max_value', 'dynamic_thresholding_ratio', 'thresholding', 'clip_sample_range', 'timestep_spacing', 'prediction_type', 'rescale_betas_zero_snr', 'variance_type'} was not found in config. Values will be initialized to default values.
pt-uwro96fh-worker-0 logs: {'scaling_factor', 'force_upcast', 'latents_std', 'latents_mean'} was not found in config. Values will be initialized to default values.
pt-uwro96fh-worker-0 logs: {'encoder_hid_dim_type', 'time_embedding_type', 'class_embed_type', 'mid_block_type', 'time_embedding_dim', 'resnet_time_scale_shift', 'resnet_out_scale_factor', 'addition_embed_type', 'transformer_layers_per_block', 'time_cond_proj_dim', 'num_class_embeds', 'addition_embed_type_num_heads', 'reverse_transformer_layers_per_block', 'use_linear_projection', 'attention_type', 'projection_class_embeddings_input_dim', 'class_embeddings_concat', 'dual_cross_attention', 'timestep_post_act', 'cross_attention_norm', 'num_attention_heads', 'time_embedding_act_fn', 'resnet_skip_time_act', 'conv_in_kernel', 'encoder_hid_dim', 'dropout', 'addition_time_embed_dim', 'only_cross_attention', 'conv_out_kernel', 'upcast_attention', 'mid_block_only_cross_attention'} was not found in config. Values will be initialized to default values.
Generating train split: 833 examples [00:00, 1985.19 examples/s]
pt-uwro96fh-worker-0 logs: Parameter 'transform'=<function main.<locals>.preprocess_train at 0x7f565c6b4940> of the transform datasets.arrow_dataset.Dataset.set_format couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.
pt-uwro96fh-worker-0 logs: 03/22/2024 15:02:17 - WARNING - datasets.fingerprint - Parameter 'transform'=<function main.<locals>.preprocess_train at 0x7f565c6b4940> of the transform datasets.arrow_dataset.Dataset.set_format couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.
pt-uwro96fh-worker-0 logs: wandb: Tracking run with wandb version 0.16.4
pt-uwro96fh-worker-0 logs: wandb: W&B syncing is set to `offline` in this directory.
pt-uwro96fh-worker-0 logs: wandb: Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
pt-uwro96fh-worker-0 logs: 03/22/2024 15:02:20 - INFO - __main__ - ***** Running training *****
pt-uwro96fh-worker-0 logs: 03/22/2024 15:02:20 - INFO - __main__ -   Num examples = 833
pt-uwro96fh-worker-0 logs: 03/22/2024 15:02:20 - INFO - __main__ -   Num Epochs = 186
pt-uwro96fh-worker-0 logs: 03/22/2024 15:02:20 - INFO - __main__ -   Instantaneous batch size per device = 8
pt-uwro96fh-worker-0 logs: 03/22/2024 15:02:20 - INFO - __main__ -   Total train batch size (w. parallel, distributed & accumulation) = 32
pt-uwro96fh-worker-0 logs: 03/22/2024 15:02:20 - INFO - __main__ -   Gradient Accumulation steps = 4
pt-uwro96fh-worker-0 logs: 03/22/2024 15:02:20 - INFO - __main__ -   Total optimization steps = 5000
Steps:   1%|          | 27/5000 [00:17<37:10,  2.23it/s, lr=0.0001, step_loss=0.0852]03/22/2024 15:02:37 - INFO - __main__ - Running validation...
pt-uwro96fh-worker-0 logs:  Generating 4 images with prompt: A pokemon with blue eyes..
pt-uwro96fh-worker-0 logs: {'image_encoder', 'requires_safety_checker'} was not found in config. Values will be initialized to default values.
pt-uwro96fh-worker-0 logs:                                           Loaded safety_checker as StableDiffusionSafetyChecker from `safety_checker` subfolder of /mnt/afs/huayil/models/stable-diffusion-v1-5.eline components...:   0%|          | 0/7 [00:00<?, ?it/s]
pt-uwro96fh-worker-0 logs:                                                   Loaded tokenizer as CLIPTokenizer from `tokenizer` subfolder of /mnt/afs/huayil/models/stable-diffusion-v1-5.
pt-uwro96fh-worker-0 logs: Loaded feature_extractor as CLIPImageProcessor from `feature_extractor` subfolder of /mnt/afs/huayil/models/stable-diffusion-v1-5.
pt-uwro96fh-worker-0 logs: {'scaling_factor', 'force_upcast', 'latents_std', 'latents_mean'} was not found in config. Values will be initialized to default values.
pt-uwro96fh-worker-0 logs: Loaded vae as AutoencoderKL from `vae` subfolder of /mnt/afs/huayil/models/stable-diffusion-v1-5.
pt-uwro96fh-worker-0 logs:                                                   Loaded text_encoder as CLIPTextModel from `text_encoder` subfolder of /mnt/afs/huayil/models/stable-diffusion-v1-5.
pt-uwro96fh-worker-0 logs: ...:  57%|█████▋    | 4/7 [00:03<00:02,  1.38it/s]{'prediction_type', 'timestep_spacing'} was not found in config. Values will be initialized to default values.
pt-uwro96fh-worker-0 logs: Loaded scheduler as PNDMScheduler from `scheduler` subfolder of /mnt/afs/huayil/models/stable-diffusion-v1-5.
Loading pipeline components...: 100%|██████████| 7/7 [00:04<00:00,  1.47it/s]


## 训练目标
训练`step = 2000` ，
在train_loss.log 中的train_loss的20次平均值对于 stable diffusion v1.5 小于 `0.079`, 
在train_loss.log 中的train_loss的10次平均值对于 stable diffusion v2.1 小于 `0.268`。
