# stable-diffusion 推理


## 准备工作

- 代码下载：https://github.com/CompVis/stable-diffusion
- 安装：https://github.com/CompVis/stable-diffusion/blob/main/README.md
- 模型权重：https://huggingface.co/runwayml/stable-diffusion-v1-5

替换 `scripts/txt2img.py`，并添加`scripts/txt2img_ckp.py`。


### 性能指标

测试命令：

```bash
python scripts/txt2img.py --prompt \
  "Emma Watson as a powerful mysterious sorceress, casting lightning magic, detailed clothing, digital painting" \
  --plms --n_samples 1 --skip_grid  --skip_save --H 512 --W 512
```
提供`bash test.sh` 测试脚本参考。提供并行测试功能，可自动读取`config.csv`启动测试。



### 模型初始化加载指标
模型加载命令如下，需根据厂商环境进行适配。
```bash
python scripts/txt2img_ckp.py
```

提供`bash test_ckp.sh` 测试脚本参考。提供暖身轮测试和多轮测试数据获取功能。


