# stable-diffusion 推理（基于TensorRT）


## 准备工作

- 代码下载：https://github.com/NVIDIA/TensorRT/tree/v9.1.0/demo/Diffusion
- 安装：https://github.com/NVIDIA/TensorRT/blob/v9.1.0/demo/Diffusion/README.md
- 环境依赖：`torch==2.1.1`, `onnx==1.14.0`, `transformers==4.31.0`, `tensorrt==9.1.0.post12.dev4`等。
- 模型权重：
v1-5: https://huggingface.co/runwayml/stable-diffusion-v1-5
v2-1: https://huggingface.co/stabilityai/stable-diffusion-2-1

替换 `demo/Diffusion/demo_txt2img.py`。


### 性能指标

测试命令：

```bash
python3 demo_txt2img.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN --version 1.5 --batch-size 1 --height 768 --width 768
```
参数说明：
- version：模型权重
- batch-size： batch size
- height/width: 图片长宽



日志参考：
```
Exported graph: graph(%input_ids : Int(*, 77, strides=[77, 1], requires_grad=0, device=cuda:0),
      %text_model.embeddings.token_embedding.weight : Float(49408, 768, strides=[768, 1], requires_grad=1, device=cuda:0),

...

[I] Initializing StableDiffusion txt2img demo using TensorRT
[I] Create directory: TensorRT/demo/Diffusion/sd1.5_height_768_width_768/engine
[I] Load tokenizer pytorch model from: pytorch_model/1.5/TXT2IMG/tokenizer
Exporting model: TensorRT/demo/Diffusion/sd1.5_height_768_width_768/onnx/clip/model.onnx
[I] Load CLIP pytorch model from: pytorch_model/1.5/TXT2IMG/text_encoder
[I] Folding Constants | Pass 1
[I]     Total Nodes | Original:  1582, After Folding:  1016 |   566 Nodes Folded
[I] Folding Constants | Pass 2
[I]     Total Nodes | Original:  1016, After Folding:   840 |   176 Nodes Folded
[I] Folding Constants | Pass 3
[I]     Total Nodes | Original:   840, After Folding:   840 |     0 Nodes Folded
Found cached model: TensorRT/demo/Diffusion/sd1.5_height_768_width_768/onnx/unet/model.onnx
Generating optimizing model: TensorRT/demo/Diffusion/sd1.5_height_768_width_768/onnx/unet.opt/model.onnx
[I] Folding Constants | Pass 1
[I]     Total Nodes | Original:  4609, After Folding:  3302 |  1307 Nodes Folded
[I] Folding Constants | Pass 2
[I]     Total Nodes | Original:  3302, After Folding:  2563 |   739 Nodes Folded
[I] Folding Constants | Pass 3
[I]     Total Nodes | Original:  2563, After Folding:  2563 |     0 Nodes Folded
Found cached model: TensorRT/demo/Diffusion/sd1.5_height_768_width_768/onnx/vae/model.onnx
Generating optimizing model: TensorRT/demo/Diffusion/sd1.5_height_768_width_768/onnx/vae.opt/model.onnx
[I] Folding Constants | Pass 1
[I]     Total Nodes | Original:   585, After Folding:   445 |   140 Nodes Folded
[I] Folding Constants | Pass 2
[I]     Total Nodes | Original:   445, After Folding:   424 |    21 Nodes Folded
[I] Folding Constants | Pass 3
[I]     Total Nodes | Original:   424, After Folding:   424 |     0 Nodes Folded
Building TensorRT engine for TensorRT/demo/Diffusion/sd1.5_height_768_width_768/onnx/clip.opt/model.onnx: TensorRT/demo/Diffusion/sd1.5_height_768_width_768/engine/clip.trt9.1.0.post12.dev4.plan
[W] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage and speed up TensorRT initialization. See "Lazy Loading" section of CUDA documentation https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#lazy-loading
[I] Configuring with profiles:[
        Profile 0:
            {input_ids [min=(1, 77), opt=(1, 77), max=(4, 77)]}
    ]
[I] Building engine with configuration:
    Flags                  | [FP16]
    Engine Capability      | EngineCapability.DEFAULT
    Memory Pools           | [WORKSPACE: 81228.56 MiB, TACTIC_DRAM: 81228.56 MiB]
    Tactic Sources         | []
    Profiling Verbosity    | ProfilingVerbosity.DETAILED
    Preview Features       | [FASTER_DYNAMIC_SHAPES_0805, DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
[I] Finished engine building in 17.004 seconds
[I] Saving engine to TensorRT/demo/Diffusion/sd1.5_height_768_width_768/engine/clip.trt9.1.0.post12.dev4.plan
Building TensorRT engine for TensorRT/demo/Diffusion/sd1.5_height_768_width_768/onnx/unet.opt/model.onnx: TensorRT/demo/Diffusion/sd1.5_height_768_width_768/engine/unet.trt9.1.0.post12.dev4.plan
[I] Configuring with profiles:[
        Profile 0:
            {sample [min=(2, 4, 96, 96), opt=(2, 4, 96, 96), max=(8, 4, 96, 96)],
             encoder_hidden_states [min=(2, 77, 768), opt=(2, 77, 768), max=(8, 77, 768)],
             timestep [min=[1], opt=[1], max=[1]]}
    ]
[I] Building engine with configuration:
    Flags                  | [FP16]
    Engine Capability      | EngineCapability.DEFAULT
    Memory Pools           | [WORKSPACE: 81228.56 MiB, TACTIC_DRAM: 81228.56 MiB]
    Tactic Sources         | []
    Profiling Verbosity    | ProfilingVerbosity.DETAILED
    Preview Features       | [FASTER_DYNAMIC_SHAPES_0805, DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
[I] Finished engine building in 207.799 seconds
[I] Saving engine to TensorRT/demo/Diffusion/sd1.5_height_768_width_768/engine/unet.trt9.1.0.post12.dev4.plan
Building TensorRT engine for TensorRT/demo/Diffusion/sd1.5_height_768_width_768/onnx/vae.opt/model.onnx: TensorRT/demo/Diffusion/sd1.5_height_768_width_768/engine/vae.trt9.1.0.post12.dev4.plan
[I] Configuring with profiles:[
        Profile 0:
            {latent [min=(1, 4, 96, 96), opt=(1, 4, 96, 96), max=(4, 4, 96, 96)]}
    ]
[I] Building engine with configuration:
    Flags                  | [FP16]
    Engine Capability      | EngineCapability.DEFAULT
    Memory Pools           | [WORKSPACE: 81228.56 MiB, TACTIC_DRAM: 81228.56 MiB]
    Tactic Sources         | []
    Profiling Verbosity    | ProfilingVerbosity.DETAILED
    Preview Features       | [FASTER_DYNAMIC_SHAPES_0805, DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
[I] Finished engine building in 135.211 seconds
[I] Saving engine to TensorRT/demo/Diffusion/sd1.5_height_768_width_768/engine/vae.trt9.1.0.post12.dev4.plan
Loading TensorRT engine: TensorRT/demo/Diffusion/sd1.5_height_768_width_768/engine/clip.trt9.1.0.post12.dev4.plan
[I] Loading bytes from TensorRT/demo/Diffusion/sd1.5_height_768_width_768/engine/clip.trt9.1.0.post12.dev4.plan
Loading TensorRT engine: TensorRT/demo/Diffusion/sd1.5_height_768_width_768/engine/unet.trt9.1.0.post12.dev4.plan
[I] Loading bytes from TensorRT/demo/Diffusion/sd1.5_height_768_width_768/engine/unet.trt9.1.0.post12.dev4.plan
Loading TensorRT engine: TensorRT/demo/Diffusion/sd1.5_height_768_width_768/engine/vae.trt9.1.0.post12.dev4.plan
[I] Loading bytes from TensorRT/demo/Diffusion/sd1.5_height_768_width_768/engine/vae.trt9.1.0.post12.dev4.plan
[I] Warming up ..
[I] Running StableDiffusion pipeline
|-----------------|--------------|
|     Module      |   Latency    |
|-----------------|--------------|
|      CLIP       |      2.65 ms |
|    UNet x 50    |   1923.51 ms |
|     VAE-Dec     |     46.68 ms |
|-----------------|--------------|
|    Pipeline     |   1972.95 ms |
|-----------------|--------------|
Throughput: 0.51 image/s
Saving image 1 / 1 to: output/txt2img-fp16-a_beautifu-1-5959.png

--------------------------------------------------
laoding model time is: 549.137106180191 s
--------------------------------------------------

GPU Memory Usage: 12.46 GB

```


