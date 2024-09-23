# InternVL 训练

## 准备工作

- 代码下载：https://github.com/OpenGVLab/InternVL/tree/main commit 764fdc9f3ee102bc6c2def02c2d0ca1e94336d06 (HEAD -> main, origin/main, origin/HEAD)
- 数据集：https://drive.google.com/file/d/1dqqa3MnrxMXaU_K9JA6C83je32ibwdOY/view
- 模型权重：
  - vision模型：https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5 
  - llm模型：https://huggingface.co/internlm/internlm2-chat-20b

## 配置

  - Global Batch Size <= 2048 
  - seq_len=8192 InternVL2要求8k上下文长度
  - freeze_llm True 冻结llm decoder
  - freeze_mlp False
  - freeze_backbone False
  - learning_rate 2e-4
  
说明：

1. 以上超参为统一值不允许修改，仅允许在原版代码上进行兼容性修改和为满足显存要求进行的并行策略修改（zero stage等）。
2. 如果厂商硬件限制，可以进行batch size 和 acumulation steps调整，global batch size 不超过2048即可。


## 启动

在云资源上启动分布式运行环境，多节点32卡的运行命令如下所示：

```bash
a808x -N4 bash -c "cd /mnt/huangye/workspace; export PATH=/mnt/huangye/workspace/conda_env/internvl/bin:$PATH; export PYTHONPATH=/mnt/huangye/workspace/conda_env/internvl/lib/python3.9/site-packages:$PYTHONPATH; cd /mnt/huangye/workspace/InternVL/internvl_chat; pwd; export; bash -c ' PARTITION='yjpc_a800' GPUS=32 PER_DEVICE_BATCH_SIZE=2 sh shell/internlm2_20b_dynamic/test.sh' ; sleep 100"
```

shell/internlm2_20b_dynamic/test.sh 脚本参考`test.sh`.

meta_path数据文件internvl_1_5.json内容格式如下：

```json
{
  "ai2d_train_12k": {
    "root": "/mnt/huangye/workspace/data/ai2d_images/",
    "annotation": "/mnt/huangye/workspace/data/opensource/ai2d_train_12k.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 12413
  }
}
```

标注文件见`ai2d_train_12k.jsonl`.

## 性能指标

根据训练日志， 采集其中`loss`、`train_samples_per_second`等，例如：

```json
{'train_runtime': 2029.3181, 'train_samples_per_second': 6.055, 'train_steps_per_second': 0.003, 'train_loss': 6.906131823857625, 'epoch': 0.99}
```

### 性能指标计算
  - TGS：`train_samples_per_second * seq_len` / `#gpu` 为了消除step波动的影响，取至少15个step的`train_samples_per_second`均值.


## 训练目标

训练step > 250 或者训练时间大于72小时，loss 小于基准值。 
日志举例：
```bash	
83%|████████▎ | 5/6 [28:30<05:28, 328.42s/it] {'loss': 4.5385, 'learning_rate': 1e-05, 'epoch': 0.82}
```