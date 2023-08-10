# BERT 预训练

## 准备工作

- 代码下载：https://github.com/codertimo/BERT-pytorch.git
- 安装：参考 https://github.com/codertimo/BERT-pytorch/blob/master/requirements.txt , 适配后代码依赖 `transformers(>=4.12.5)`，代码依赖的torch版本在`1.7.0`已验证，环境需根据厂商环境进行适配。
- 数据集：wikipedia数据集 https://huggingface.co/datasets/wikipedia 

### 代码适配修改
`BERT`完成预训练需要进行如下适配：

替换

`BERT-pytorch/bert_pytorch/dataset/dataset.py`

`BERT-pytorch/bert_pytorch/dataset/trainer/pretrain.py`

为本目录下相应文件。


并将`main.py`加入到
`BERT-pytorch/bert_pytorch/`目录下。



## 启动及数据采集

单机多卡
```bash
cd BERT-pytorch/bert_pytorch
mkdir output_path
python main.py -c /path/to/dataset/wikipedia_bert -o output_path/bert --cuda_devices 2 3 4 5

```


### 性能指标

根据训练日志，采集其中Loss数值
```bash
INFO:root:Epoch [0], Step [93180], next_loss[0.17973171174526215], mask_loss[3.3067405223846436], avg_loss: 4.6533, next_avg_acc: 88.94230382266771, mask_avg_acc33.582920239352674, lr9.15978787878788e-05
```
### 稳定性指标

对于稳定性指标，记录前100个step的Loss。



## 训练目标
训练step > 100000或者训练时间大于72小时，Loss小于 5.4795。