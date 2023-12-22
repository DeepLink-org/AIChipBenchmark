# LLaMa 推理


## 准备工作

- 代码下载：https://github.com/InternLM/lmdeploy
- 安装：参考 https://github.com/InternLM/lmdeploy/blob/v0.0.14/README.md, 需根据厂商环境进行适配（lmdeploy==0.0.14）
- 模型：LLAMA-V1（7B/65B） & LLAMA-V2（7B/70B），https://github.com/facebookresearch/llama/blob/main/README.md



## 启动及数据采集

###  量化(可选)
参考：https://github.com/InternLM/lmdeploy/blob/v0.0.14/docs/zh_cn/w4a16.md#4bit-%E6%9D%83%E9%87%8D%E9%87%8F%E5%8C%96


### 模型转换

```bash
lmdeploy convert llama2 /path/to/your/model --tp 2

```
命令将原生llama模型文件，转换成TurboMind格式的文件，进行下面的推理。

参考：https://github.com/InternLM/lmdeploy/blob/v0.0.14/docs/zh_cn/serving.md

量化模型转换：https://github.com/InternLM/lmdeploy/blob/v0.0.14/docs/zh_cn/w4a16.md#4bit-%E6%9D%83%E9%87%8D%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86

#### 配置修改
修改`/path/to/your/model`中的`triton_models/weights/config.ini`中
```
[llama]
...
max_batch_size = 128
...
cache_max_entry_count = 128
...
```


### 性能指标
启动性能测试的命令，参考 https://github.com/InternLM/lmdeploy/blob/v0.0.14/benchmark/README.md，需根据厂商环境进行适配。

#### 开启流式推理
修改`profile_generation.py`中`stream_infer`接口为：
```python
...
        for outputs in chatbot.stream_infer(session_id,
                                            input_ids,
                                            request_output_len=output_seqlen,
                                            sequence_start=True,
                                            sequence_end=True,
                                            ignore_eos=True,
                                            stream_output=True):

...

        for _ in range(warmup_round):
            for _ in chatbot.stream_infer(session_id,
                                          input_ids=input_ids,
                                          request_output_len=output_seqlen,
                                          sequence_start=True,
                                          sequence_end=True,
                                          ignore_eos=True,
                                          stream_output=True):
                continue
...

```

测试命令：

```bash

 python profile_generation.py  \
        --model-path /path/to/your/model \
        --concurrency  1 8 16 32 64 \
        --prompt-tokens 256 256 256 512 512 512 1024 \
        --completion-tokens 128 512 1024 128 512 1024 1024 \
        --tp 2 \
        --dst-csv results.csv

```


### 模型初始化加载指标
模型加载命令如下，需根据厂商环境进行适配。
```bash
python profile_ckp_time.py  /path/to/your/model --tp 8
```

提供`bash test_ckp.sh MODEL_NAME` 测试脚本参考。提供暖身轮测试和多轮数据平均功能。


