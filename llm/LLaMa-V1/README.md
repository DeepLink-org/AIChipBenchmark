# LLaMa-V1 推理


## 准备工作

- 代码下载：https://github.com/InternLM/lmdeploy
- 安装：参考 https://github.com/InternLM/lmdeploy/blob/main/README_zh-CN.md, 需根据厂商环境进行适配
- 数据集：LLAMA-V1数据集（7B/65B），https://github.com/facebookresearch/llama/blob/llama_v1/README.md



## 启动及数据采集

### 模型转换

```python
python3 -m lmdeploy.serve.turbomind.deploy llama /path/to/llama/model llama /path/to/tokenizer.model /path/to/your/model

```
命令将原生llama模型文件，转换成TurboMind格式的文件，进行下面的推理。


### 性能指标
启动性能测试的命令，参考 https://github.com/InternLM/lmdeploy/blob/main/benchmark/README.md

```python
python profile_generation.py \
 /path/to/your/model \
 --concurrency 8 --input_seqlen 512 --output_seqlen 512 --test_round 10 --tp 8

```
提供`bash test.sh` 测试脚本参考。提供并行测试功能，可自动读取`config.csv`启动测试。



### 模型初始化加载指标
```python
python profile_ckp_time.py  /path/to/your/model --tp 8
```

提供`bash test_ckp.sh MODEL_NAME` 测试脚本参考。提供暖身轮测试和多轮数据平均功能。


