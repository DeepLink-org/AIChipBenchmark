# LLaVA-InternLM2-20B 预训练

## 准备工作

1. 根据 [XTuner 文档](https://github.com/InternLM/xtuner/tree/770bac38bc905794eb38e53de4f54f98e30a77dc?tab=readme-ov-file#installation)准备运行环境；
   - 注意，默认安装步骤不一定能正确安装 PyTorch 版本（例如使用 N 卡 CUDA 版本 11.8 应该额外安装 `pytorch-cuda=11.8`），请注意检查。
   - 本次测试需要使用 DeepSpeed（如果不启用则需要修改依赖及配置）。
2. 下载预训练数据集 [liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main)，并**解压** `images.zip`；
3. （可选）下载预训练模型。如果没有提前下载，它们会在预训练阶段自动下载：
   - [internlm/internlm2-chat-20b](https://huggingface.co/internlm/internlm2-chat-20b)
   - [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)
4. 将修改好的配置文件 llava_internlm2_chat_20b_clip_vit_large_p14_336_e1_gpu8_pretrain.py 复制至工作目录；
   - 原版配置可参考 [XTuner 仓库](https://github.com/InternLM/xtuner/blob/770bac38bc905794eb38e53de4f54f98e30a77dc/xtuner/configs/llava/internlm2_chat_20b_clip_vit_large_p14_336/pretrain/llava_internlm2_chat_20b_clip_vit_large_p14_336_e1_gpu8_pretrain.py)；
   - 如果没有提前下载预训练数据集，则需要修改配置文件中的模型路径为模型名称。

      ```python
      llm_name_or_path = 'internlm/internlm2-chat-20b'
      visual_encoder_name_or_path = 'openai/clip-vit-large-patch14-336'
      ```

5. 最终按照如下布局准备测试文件夹。

   ```text
   ./llava
   ├── data
   │   └── llava_data
   │       └── LLaVA-Pretrain
   ├── llava_internlm2_chat_20b_clip_vit_large_p14_336_e1_gpu8_pretrain.py
   └── model
       ├── clip-vit-large-patch14-336
       └── internlm2-chat-20b
   ```

## 开始训练

使用 XTuner 启动预训练。

### 4 节点 8 卡（共 32 卡）

分布式训练依赖环境中的 `$WORLD_SIZE`、`$MASTER_PORT`、`$MASTER_ADDR`、`$RANK` 等环境变量。

```bash
NPROC_PER_NODE=8 NNODES=$WORLD_SIZE PORT=$MASTER_PORT ADDR=$MASTER_ADDR NODE_RANK=$RANK xtuner train llava_internlm2_chat_20b_clip_vit_large_p14_336_e1_gpu8_pretrain.py --deepspeed deepspeed_zero2
```

其它情况可以调整 `NPROC_PER_NODE`、`NNODES` 等参数进行训练。

### 1 节点 8 卡

单节点训练如下所示，其中 `NPROC_PER_NODE` 即当前节点可用卡的数量。

```bash
NPROC_PER_NODE=8 xtuner train llava_internlm2_chat_20b_clip_vit_large_p14_336_e1_gpu8_pretrain.py --deepspeed deepspeed_zero2
```

不涉及分布式训练，可用于单机调试。

## 性能指标

训练过程中，它会在当前工作目录下生成 `work_dirs`，用于保存运行日志（但不包含报错信息）、性能数据、checkpoint 等文件。

其中性能数据文件 `./work_dirs/<配置名称>/<日期>/vis_data/<日期>.json` 以 JSON stream 的形式记录了性能指标，可以采集其中的 `loss`、`tflops`、`tokens_per_sec` 等指标。例如：

```json
{"lr": 0.0006877425862068985, "data_time": 0.4952368974685669, "loss": 2.6702206134796143, "time": 14.25826325416565, "tflops": 19.10478222297824, "tokens_per_sec": 107.58153684713727, "iter": 360, "memory": 49335, "step": 360}
{"lr": 0.0007068994827586228, "data_time": 0.4770257234573364, "loss": 2.719205904006958, "time": 14.272274851799011, "tflops": 21.366428213207822, "tokens_per_sec": 120.29798370142593, "iter": 370, "memory": 49335, "step": 370}
{"lr": 0.0007260563793103471, "data_time": 0.49270365238189695, "loss": 2.797193455696106, "time": 14.245970010757446, "tflops": 18.356761561585994, "tokens_per_sec": 103.37482784717935, "iter": 380, "memory": 49121, "step": 380}
```

## 训练目标

以 A100 为基准，能在 500 个 step 内将 loss 降低至 2.3 以内。
