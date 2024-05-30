# LLaMa 场景化推理


## 准备工作

- 代码下载：https://github.com/InternLM/lmdeploy/tree/v0.4.1
- 安装：参考 https://github.com/InternLM/lmdeploy/blob/v0.4.1/docs/zh_cn/get_started.md, 需根据厂商环境进行适配（lmdeploy==0.4.1）
- 模型：LLAMA-V1（65B） & LLAMA-V2（7B/13B/70B），https://github.com/facebookresearch/llama/blob/main/README.md


## 启动及数据采集

### 数据集
https://pjlab-my.sharepoint.cn/:u:/g/personal/zhongpu_pjlab_org_cn/Ebnbrcbe9U1Nifdrfcr6tG8B-hvy-XstA258lqqzz8YJzw?e=Kh3uGJ

将`overall_input_token_ids.txt`和`overall_output_length.txt`解压到`benchmark`目录。

### 模型转换

```bash
lmdeploy convert llama2 /path/to/llama2/70B --dst-path /path/to/workspace/llama2/70B --tp 8
```
命令将原生llama模型文件，转换成TurboMind格式的文件，进行下面的推理。


### 性能指标
新增`profile.py`,`benchmark.sh`到`benchmark`目录，启动性能测试的命令，参考 https://github.com/InternLM/lmdeploy/blob/v0.4.1/benchmark/README.md ， 需根据厂商环境进行适配。


测试命令：

```bash
 python3 profile.py ${model_path} --tp ${tp} \
        --concurrency ${concurrency} \
        --cache-max-entry-count ${cache_max_entry_count} \
        --csv llama2_tb_13b_thr_${concurrency}.csv
```

提供`bash benchmark.sh` 测试脚本供参考。脚本提供批量化测试功能。


`llama-7B` 输出性能指标如下：

| batch | num_promts | RPS | RPM | FTL(ave)(s) | FTL(min)(s) | FTL(max)(s) | 50%(s) | 75%(s) | 95%(s) | 99%(s) | throughput(out tok/s) | throughput(total tok/s) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 128 | 10000 | 5.049 | 302.963 | 0.152 | 0.057 | 4.544 | 0.044 | 0.055 | 0.089 | 0.112 | 2582.949 | 5167.244 |