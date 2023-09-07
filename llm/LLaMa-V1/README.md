# LLaMa-V1 推理


## 准备工作

- 代码下载：https://github.com/InternLM/lmdeploy
- 安装：参考 https://github.com/InternLM/lmdeploy/blob/main/README_zh-CN.md, 需根据厂商环境进行适配（lmdeploy==0.0.5）
- 数据集：LLAMA-V1数据集（7B/65B），https://github.com/facebookresearch/llama/blob/llama_v1/README.md



## 启动及数据采集

### 模型转换

```python
python3 -m lmdeploy.serve.turbomind.deploy llama /path/to/llama/model llama /path/to/tokenizer.model /path/to/your/model

```
命令将原生llama模型文件，转换成TurboMind格式的文件，进行下面的推理。

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
启动性能测试的命令，参考 https://github.com/InternLM/lmdeploy/blob/main/benchmark/README.md，需根据厂商环境进行适配。

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
python profile_generation.py \
 /path/to/your/model \
 --concurrency 8 --input_seqlen 512 --output_seqlen 512 --test_round 10 --tp 8

```
提供`bash test.sh` 测试脚本参考。提供并行测试功能，可自动读取`config.csv`启动测试。



### 模型初始化加载指标
模型加载命令如下，需根据厂商环境进行适配。
```bash
python profile_ckp_time.py  /path/to/your/model --tp 8
```

提供`bash test_ckp.sh MODEL_NAME` 测试脚本参考。提供暖身轮测试和多轮数据平均功能。


