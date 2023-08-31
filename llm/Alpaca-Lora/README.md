# Alpaca-Lora 微调

## 准备工作

- 代码下载：git clone https://github.com/tloen/alpaca-lora.git
- 安装：参考 https://github.com/codertimo/BERT-pytorch/blob/master/requirements.txt , 适配后代码依赖 `transformers(>=4.12.5)`，代码依赖的torch版本在`1.7.0`已验证，环境需根据厂商环境进行适配。
- 数据集：AlpacaDataCleaned https://huggingface.co/datasets/yahma/alpaca-cleaned


## 启动及数据采集

```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length
```

相关配置说明：

- base_model：表示模型路径
- data_path：数据集路径
- output_dir ：输出路径
- batch_size/micro_batch_size ：批大小
- num_epochs ：训练次数
- cutoff_len ：句子最大长度，这个参数严重影响训练时间与模型性能
- lora_r：lora的秩，一般取2，4，8，64



### 性能指标

根据训练日志，采集其中Loss数值和相关性能指标。
```bash
{'loss': 0.8603, 'learning_rate': 3.5714285714285714e-06, 'epoch': 2.91}
{'loss': 0.8456, 'learning_rate': 2.631578947368421e-06, 'epoch': 2.93}
{'loss': 0.8863, 'learning_rate': 1.6917293233082707e-06, 'epoch': 2.96}
{'loss': 0.8408, 'learning_rate': 7.518796992481203e-07, 'epoch': 2.98}
{'train_runtime': 3581.7097, 'train_samples_per_second': 41.678, 'train_steps_per_second': 0.325, 'train_tokens_per_second(tgs)': 1228.267, 'train_loss': 0.9499486045739085, 'epoch': 2.99}
```

`tgs`指标依赖以下PR：https://github.com/huggingface/transformers/pull/25858

### 稳定性指标
对于稳定性指标，记录前100个step的Loss。



## 训练目标
训练epoch > 290或者训练时间大于72小时，训练Loss小于 1.01。