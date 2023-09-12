# BERT 预训练

## 准备工作

- 代码下载：https://github.com/codertimo/BERT-pytorch.git (commit d10dc4f9d5a6f2ca74380f62039526eb7277c671)
- 安装：参考 https://github.com/codertimo/BERT-pytorch/blob/master/requirements.txt , 适配后代码依赖 `transformers(>=4.12.5)`，代码依赖的torch版本在`1.7.0`/`1.8.0`已验证，环境需根据厂商环境进行适配
- 数据集：原始wikipedia数据集 https://huggingface.co/datasets/wikipedia 需做预处理，预处理耗时很长，可以直接下载预处理后的数据集做训练  https://huggingface.co/datasets/yeegnauh/bert_wikipedia
- Tokenizer：https://huggingface.co/bert-base-uncased 下载 bert-base-uncased 的 tokenizer到dataset文件夹下 

### 代码适配修改
`BERT`完成预训练需要进行如下适配：

替换

`BERT-pytorch/bert_pytorch/dataset/dataset.py`

`BERT-pytorch/bert_pytorch/dataset/trainer/pretrain.py`

为本目录下相应文件。


并将`main.py`加入到 `BERT-pytorch/bert_pytorch/`目录下。



## 启动及数据采集

单机多卡示例
```bash
cd BERT-pytorch/bert_pytorch
mkdir output_path
python main.py -c /path/to/dataset/bert_wikipedia -o output_path/bert --cuda_devices 2 3 4 5

```


### 性能指标

根据训练日志，采集其中tgs数值
```bash
INFO:root:Epoch [0], Step [500], next_loss[0.9829702377319336], mask_loss[9.950705528259277], avg_loss: 12.1177, next_avg_acc: 52.94411177644711, loss: 10.933675765991211, lr: 5.01e-06, tgs= 23955.620667248597, tgs_sum= 1178411.637945293
```


### 训练目标

训练时间大于72小时，Loss收敛，记录avg_loss。


## 数据预处理(直接使用处理后的数据集则无需此步)

```
pip install apache_beam mwparserfromhell datasets 
pip install nltk
cd dataset
# generate.py处理全部的wikipedia数据运行时间会比较久
python generate.py 
```
生成的16个txt文件共19G左右, md5值如下：

29a5148663da5471df7a09e956d1fd4d  to0.txt
0dd5f7e99f4e62a84894a71d422bc72e  to1.txt
8d43af21a290f7c902b1789dd7a6c5e6  to10.txt
8953894ec81a2e11d5df0ba9d8d29678  to11.txt
1c45218d16a44f8be51bd7861bf2308b  to12.txt
04745b0d21b2e53587f8c373ebf51f83  to13.txt
aeba3e763208733eb309a361ea77ef86  to14.txt
6db625573b3ec326d4de24f54aa6d6bd  to15.txt
93a618a36cb3f31c0ce71ef1ec9652f7  to2.txt
3b884731fde8a7e8364cc8465a59e6bd  to3.txt
0e1843ccb4b1dd124163b8f11def8377  to4.txt
512eab6883e0aea6b0ed3e5ae67d465f  to5.txt
91b2d592082005b3d4f97ce77b873cc5  to6.txt
d28c083c88c1202265f000e3a5f6683d  to7.txt
403b6245fcb9daf59b60d5471745e934  to8.txt
4cafff88a7fc748919d558ce858d343f  to9.txt

