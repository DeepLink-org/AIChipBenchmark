# Open-sora 推理（基于Pytorch）


## 准备工作

- 代码下载：https://github.com/hpcaitech/Open-Sora
- 安装：https://github.com/hpcaitech/Open-Sora/blob/main/docs/zh_CN/README.md
- 环境依赖：`torch==2.1.2`, `torchvision==0.16.2`, `transformers==4.39.1`, `apex==0.1`等。
  
  `apex`版本需要与`transformers`版本对应才可以正确安装`apex.optimizers.FusedAdam`扩展，安装过程中需保证torch版本与cuda版本一致。
```bash
git clone https://github.com/NVIDIA/apex
cd apex
# 将apex版本回退回23.05
git checkout 0da3ffb92ee6fbe5336602f0e3989db1cd16f880
# 如果是集群则需要提交至计算节点安装
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
- 模型权重：

  替换 `OpenSora/configs/opensora-v1-2/inference/sample.py`。

  OpenSora-STDiT-v3: https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3
  
  OpenSora-VAE-v1.2: https://huggingface.co/hpcai-tech/OpenSora-VAE-v1.2
  
  t5-v1_1-xxl: https://huggingface.co/DeepFloyd/t5-v1_1-xxl/tree/main

  替换 `Open-Sora/opensora/models/vae/vae.py`。
  
  pixart_sigma_sdxlvae_T5_diffusers: https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/vae
- 修改 `Open-Sora/scripts/inference.py`，方便在推理时直接指定视频时长。
```bash
# num_frames = get_num_frames(cfg.num_frames)
time_len = cfg.get("time_len", None)
num_frames = int(time_len/2*3*17)
```
- 添加 `Open-Sora/opensora/utils/config_utils.py`
```bash
parser.add_argument("--time-len", default=None, type=int, help="Time length for the video")
```

## 启动

### 配置
`sample.py` 文件内容参考
```
time_len = None
image_size = None
fps = 24
frame_interval = 1
save_fps = 24

save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "fp16"
condition_frame_length = 5
align = 5

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="/mnt/lustrenew/share_data/PAT/datasets/huggingface/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="/mnt/lustrenew/share_data/PAT/datasets/huggingface/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained="/mnt/lustrenew/share_data/PAT/datasets/huggingface/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5
flow = 5

```

### 性能指标

测试命令：

基础的命令行推理:
```bash
python scripts/inference.py configs/opensora-v1-2/inference/sample.py \
  --time-len 2 \
  --image-size 1280 720 \
  --prompt "a beautiful waterfall" \
```
```
参数说明：
- time-len：时长
- image-size: 分辨率
```

日志参考：
```
[2024-08-14 16:19:56,070] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2024-08-14 16:20:25] Inference configuration:
 {'aes': 6.5,
 'align': 5,
 'batch_size': 1,
 'condition_frame_length': 5,
 'config': '/mnt/lustrenew/dongkaixing1.vendor/llm/chenyuxiao/Open-Sora/configs/opensora-v1-2/inference/sample.py',
 'dtype': 'fp16',
 'flow': 5,
 'fps': 24,
 'frame_interval': 1,
 'image_size': [1280, 720],
 'model': {'enable_flash_attn': True,
           'enable_layernorm_kernel': True,
           'from_pretrained': '/mnt/lustrenew/share_data/PAT/datasets/huggingface/OpenSora-STDiT-v3',
           'qk_norm': True,
           'type': 'STDiT3-XL/2'},
 'multi_resolution': 'STDiT2',
 'prompt': ['a beautiful waterfall'],
 'prompt_as_path': False,
 'save_dir': './samples/samples/',
 'save_fps': 24,
 'scheduler': {'cfg_scale': 7.0,
               'num_sampling_steps': 30,
               'type': 'rflow',
               'use_timestep_transform': True},
 'seed': 42,
 'text_encoder': {'from_pretrained': '/mnt/lustrenew/share_data/PAT/datasets/huggingface/t5-v1_1-xxl',
                  'model_max_length': 300,
                  'type': 't5'},
 'time_len': 2,
 'vae': {'from_pretrained': '/mnt/lustrenew/share_data/PAT/datasets/huggingface/OpenSora-VAE-v1.2',
         'micro_batch_size': 4,
         'micro_frame_size': 17,
         'type': 'OpenSoraVAE_V1_2'}}
[2024-08-14 16:20:25] Building models...
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]/mnt/cachenew/share/platform/env/miniconda3.10/envs/lazyllm2/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
Loading checkpoint shards: 100%|██████████| 2/2 [00:59<00:00, 29.98s/it]
[2024-08-14 16:22:23] Model checkpoint loaded from /mnt/lustrenew/share_data/PAT/datasets/huggingface/OpenSora-VAE-v1.2
[2024-08-14 16:24:57] Model checkpoint loaded from /mnt/lustrenew/share_data/PAT/datasets/huggingface/OpenSora-STDiT-v3
100%|██████████| 1/1 [02:12<00:00, 132.90s/it]
[2024-08-14 16:27:13] Inference finished.
[2024-08-14 16:27:13] Saved 1 samples to ./samples/samples/
```
