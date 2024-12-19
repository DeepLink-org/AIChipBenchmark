# Mixtral 8x7B 微调

## 准备工作

1. 根据 [XTuner 文档](https://github.com/InternLM/xtuner/blob/v0.1.23/README_zh-CN.md#%E5%AE%89%E8%A3%85)准备运行环境；
   - 本次测试需要使用 DeepSpeed（如果不启用则需要修改依赖及配置）。
2. 将修改的配置文件，移动到当前目录下，准备微调。
注意：配置文件进行了`per_device batchsize`、`ThroughputHook`、`logging interval`的修改，注意按需适配。



## 替换throughput_hook.py
由于xtuner不支持整个`optimizer step`的`TGS`计算（参考：https://github.com/InternLM/xtuner/issues/967）,如果开启了`gradient accumulation`（即accumulative_counts不等于1），需要用提供的`throughput_hook.py`替换`xtuner/engine/hooks/throughput_hook.py`。


## 开始微调

使用 XTuner 启动预训练。

### QLora微调（单机单卡）


```bash
srun -p Intern5 --job-name=mixtral1 --nodes=1 --gres=gpu:1 --ntasks-per-node=1 xtuner train ./mixtral_8x7b_instruct_qlora_oasst1_e3_copy.py --deepspeed deepspeed_zero2 --launcher slurm
```


### Full全参微调（双机16卡）


```bash
srun -p Intern5 --job-name=mixtral2 --nodes=2 --gres=gpu:8 --ntasks-per-node=8 xtuner train mixtral_8x7b_instruct_full_oasst1_e3_copy2.py --deepspeed deepspeed_zero3 --launcher slurm
```

不涉及分布式训练，可用于单机调试。

## 性能指标

训练过程中，它会在当前工作目录下生成 `work_dirs`，用于保存运行日志（但不包含报错信息）、性能数据、checkpoint 等文件。


日志文件 `./work_dirs/<配置名称>/<日期>/<日期>.log` 也记录了逐个`step`的`loss`值和`tokens_per_sec`数据。
```
2024/12/11 13:19:58 - mmengine - INFO - Iter(train) [380/387]  lr: 2.1080e-07  eta: 0:02:08  time: 18.3345  data_time: 0.0554  memory: 65462  loss: 0.6420  tflops: 116.4251  tokens_per_sec: 1785.3621
2024/12/11 13:20:16 - mmengine - INFO - Iter(train) [381/387]  lr: 1.6141e-07  eta: 0:01:50  time: 18.3352  data_time: 0.0554  memory: 65459  loss: 0.6397  tflops: 116.5561  tokens_per_sec: 1787.3699
2024/12/11 13:20:35 - mmengine - INFO - Iter(train) [382/387]  lr: 1.1859e-07  eta: 0:01:31  time: 18.3368  data_time: 0.0561  memory: 65461  loss: 0.6420  tflops: 116.5291  tokens_per_sec: 1786.9560
2024/12/11 13:20:53 - mmengine - INFO - Iter(train) [383/387]  lr: 8.2362e-08  eta: 0:01:13  time: 18.3363  data_time: 0.0561  memory: 65460  loss: 0.6388  tflops: 116.6099  tokens_per_sec: 1788.1952
2024/12/11 13:21:11 - mmengine - INFO - Iter(train) [384/387]  lr: 5.2714e-08  eta: 0:00:55  time: 18.3374  data_time: 0.0563  memory: 65461  loss: 0.6324  tflops: 116.5887  tokens_per_sec: 1787.8706
2024/12/11 13:21:29 - mmengine - INFO - Iter(train) [385/387]  lr: 2.9653e-08  eta: 0:00:36  time: 18.3352  data_time: 0.0558  memory: 65462  loss: 0.6215  tflops: 116.6584  tokens_per_sec: 1788.9389
2024/12/11 13:21:48 - mmengine - INFO - Iter(train) [386/387]  lr: 1.3179e-08  eta: 0:00:18  time: 18.3343  data_time: 0.0558  memory: 65463  loss: 0.6225  tflops: 116.5881  tokens_per_sec: 1787.8610
2024/12/11 13:22:06 - mmengine - INFO - Iter(train) [387/387]  lr: 3.2949e-09  eta: 0:00:00  time: 18.3340  data_time: 0.0558  memory: 65464  loss: 0.6251  tflops: 116.5461  tokens_per_sec: 1787.2171
```


### 性能指标计算
- TGS: 为了消除step波动的影响，`tokens_per_sec`去除最前面和最后面各`5`个step，取均值。

可以使用本代码仓提供的`tools/calc.py`计算`tokens_per_sec`的均值，
```bash
python calc.py <log_file_path> <start_iter> <end_iter>
```
### batch size说明

- QLora微调（单机单卡） 微调参考配置`mixtral_8x7b_instruct_qlora_oasst1_e3_copy.py `，功能和性能指标，采用统一的`batchsize=16`。

- Full全参微调（双机16卡） 微调参考配置`mixtral_8x7b_instruct_full_oasst1_e3_copy2.py `，功能和性能指标，采用统一的`batchsize=256`。


## 训练目标

由于使用相同的数据集，以QLora微调以例，能在 500 - 1000 个 step 内将 loss 降低至 `0.6251` 以内。
