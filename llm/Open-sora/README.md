# Open-sora2.0 推理（基于Pytorch）


## 准备工作

- 代码下载：https://github.com/hpcaitech/Open-Sora
- 安装：https://github.com/hpcaitech/Open-Sora/blob/main/README.md

- 环境依赖：`torch==2.4.0+cu124`, `torchvision==0.19.0+cu124`, `transformers==4.51.3`等。

```bash
# create a virtual env and activate (conda as an example)
conda create -n opensora python=3.10
conda activate opensora

# download the repo
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora

# Ensure torch >= 2.4.0
pip install -v . # for development mode, `pip install -v -e .`
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 # install xformers according to your cuda version
pip install flash-attn --no-build-isolation
```

- 通过下载flash attention 3获得更快的速度

```bash
git clone https://github.com/Dao-AILab/flash-attention # 4f0640d5
cd flash-attention/hopper
python setup.py install
```
- 模型下载：

```bash
git lfs install
git clone https://huggingface.co/hpcai-tech/Open-Sora-v2
```
- 模型权重：

  替换 `./Open-Sora/configs/diffusion/inference/256px.py`。

  model: `./Open-Sora-v2/Open_Sora_v2.safetensors`
  
  ae: `./Open-Sora-v2/hunyuan_vae.safetensors`
  
  t5: `./Open-Sora-v2/google/t5-v1_1-xxl`

  clip: `./Open-Sora-v2/openai/clip-vit-large-patch14`

## 启动

### 配置
`256px.py` 文件内容参考
```
save_dir = "samples"  # save directory
seed = 42  # random seed (except seed for z)
batch_size = 1
dtype = "fp16" 
cond_type = "t2v"
# conditional inference options:
# t2v: text-to-video
# i2v_head: image-to-video (head)
# i2v_tail: image-to-video (tail)
# i2v_loop: connect images
# v2v_head_half: video extension with first half
# v2v_tail_half: video extension with second half

dataset = dict(type="text")
sampling_option = dict(
    resolution="256px",  # 256px or 768px
    aspect_ratio="16:9",  # 9:16 or 16:9 or 1:1
    num_frames=129,  # number of frames
    num_steps=50,  # number of steps
    shift=True,
    temporal_reduction=4,
    is_causal_vae=True,
    guidance=7.5,  # guidance for text-to-video
    guidance_img=3.0,  # guidance for image-to-video
    text_osci=True,  # enable text guidance oscillation
    image_osci=True,  # enable image guidance oscillation
    scale_temporal_osci=True,
    method="i2v",  # hard-coded for now
    seed=None,  # random seed for z
)
motion_score = "4"  # motion score for video generation
fps_save = 24  # fps for video generation and saving

# Define model components
model = dict(
    type="flux",
    from_pretrained="/mnt/139_nvme2/cyx/Open-Sora-v2/Open_Sora_v2.safetensors",
    guidance_embed=False,
    fused_qkv=False,
    use_liger_rope=True,
    # model architecture
    in_channels=64,
    vec_in_dim=768,
    context_in_dim=4096,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=19,
    depth_single_blocks=38,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    cond_embed=True,
)
ae = dict(
    type="hunyuan_vae",
    from_pretrained="/mnt/139_nvme2/cyx/Open-Sora-v2/hunyuan_vae.safetensors",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    use_spatial_tiling=True,
    use_temporal_tiling=False,
)
t5 = dict(
    type="text_embedder",
    from_pretrained="/mnt/139_nvme2/cyx/Open-Sora-v2/google/t5-v1_1-xxl",
    max_length=512,
    shardformer=True,
)
clip = dict(
    type="text_embedder",
    from_pretrained="/mnt/139_nvme2/cyx/Open-Sora-v2/openai/clip-vit-large-patch14",
    max_length=77,
)
```

### 性能指标

测试命令：

基础的命令行推理:
```bash
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/256px.py \
--prompt "raining, sea" \
--offload True \
```
日志参考：
```
[07/07/25 16:52:07] INFO     colossalai - colossalai - INFO: /opt/miniconda3/envs/opensora/lib/python3.10/site-packages/colossalai/initialize.py:75 launch                                      
                    INFO     colossalai - colossalai - INFO: Distributed environment is initialized, world size: 1                                                                              
[[34m2025-07-07 16:52:07[0m] Inference configuration:
 {'ae': {'from_pretrained': '/mnt/139_nvme2/cyx/Open-Sora-v2/hunyuan_vae.safetensors',
        'in_channels': 3,
        'latent_channels': 16,
        'layers_per_block': 2,
        'out_channels': 3,
        'type': 'hunyuan_vae',
        'use_spatial_tiling': True,
        'use_temporal_tiling': False},
 'batch_size': 1,
 'clip': {'from_pretrained': '/mnt/139_nvme2/cyx/Open-Sora-v2/openai/clip-vit-large-patch14',
          'max_length': 77,
          'type': 'text_embedder'},
 'cond_type': 't2v',
 'config_path': 'configs/diffusion/inference/256px.py',
 'dataset': {'type': 'text'},
 'dtype': 'fp16',
 'fps_save': 24,
 'model': {'axes_dim': [16, 56, 56],
           'cond_embed': True,
           'context_in_dim': 4096,
           'depth': 19,
           'depth_single_blocks': 38,
           'from_pretrained': '/mnt/139_nvme2/cyx/Open-Sora-v2/Open_Sora_v2.safetensors',
           'fused_qkv': False,
           'guidance_embed': False,
           'hidden_size': 3072,
           'in_channels': 64,
           'mlp_ratio': 4.0,
           'num_heads': 24,
           'qkv_bias': True,
           'theta': 10000,
           'type': 'flux',
           'use_liger_rope': True,
           'vec_in_dim': 768},
 'motion_score': '4',
 'offload': True,
 'prompt': 'raining, sea',
 'sampling_option': {'aspect_ratio': '16:9',
                     'guidance': 7.5,
                     'guidance_img': 3.0,
                     'image_osci': True,
                     'is_causal_vae': True,
                     'method': 'i2v',
                     'num_frames': 129,
                     'num_steps': 50,
                     'resolution': '256px',
                     'scale_temporal_osci': True,
                     'seed': None,
                     'shift': True,
                     'temporal_reduction': 4,
                     'text_osci': True},
 'save_dir': 'samples',
 'seed': 42,
 't5': {'from_pretrained': '/mnt/139_nvme2/cyx/Open-Sora-v2/google/t5-v1_1-xxl',
        'max_length': 512,
        'shardformer': True,
        'type': 'text_embedder'}}
[[34m2025-07-07 16:52:07[0m] Building dataset...
[[34m2025-07-07 16:52:08[0m] Dataset contains 1 samples.
[[34m2025-07-07 16:52:08[0m] Building models...
[[34m2025-07-07 16:52:08[0m] Loading checkpoint from /mnt/139_nvme2/cyx/Open-Sora-v2/Open_Sora_v2.safetensors
[[34m2025-07-07 16:52:12[0m] Model loaded successfully
[[34m2025-07-07 16:52:13[0m] Loading checkpoint from /mnt/139_nvme2/cyx/Open-Sora-v2/hunyuan_vae.safetensors
[[34m2025-07-07 16:52:13[0m] Model loaded successfully

Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
Loading checkpoint shards:  50%|█████     | 1/2 [00:04<00:04,  4.99s/it]
Loading checkpoint shards: 100%|██████████| 2/2 [00:10<00:00,  5.22s/it]
Loading checkpoint shards: 100%|██████████| 2/2 [00:10<00:00,  5.19s/it]
[[34m2025-07-07 16:52:24[0m] CUDA max memory max memory allocated at build model: 33.6 GB
[[34m2025-07-07 16:52:24[0m] CUDA max memory max memory reserved at build model: 34.4 GB

Inference progress:   0%|          | 0/1 [00:00<?, ?it/s][[34m2025-07-07 16:52:24[0m] Generating video...
Saved to samples/video_256px/prompt_0000.mp4

Inference progress: 100%|██████████| 1/1 [00:48<00:00, 48.54s/it]
Inference progress: 100%|██████████| 1/1 [00:48<00:00, 48.56s/it]
[[34m2025-07-07 16:53:12[0m] Inference finished.
[[34m2025-07-07 16:53:12[0m] CUDA max memory max memory allocated at inference: 52.5 GB
[[34m2025-07-07 16:53:12[0m] CUDA max memory max memory reserved at inference: 70.1 GB
[rank0]:[W707 16:53:13.634091240 ProcessGroupNCCL.cpp:1168] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
```
