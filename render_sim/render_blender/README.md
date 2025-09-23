# 渲染评测 （基于blender）


## 环境准备

- 下载 [blender](https://www.blender.org/download) （可根据厂商适配）
- 下载场景文件：
    - [monster](https://ftp.nluug.nl/pub/graphics/blender/release/BlenderBenchmark2.0/scenes/monster.tar.bz2)
    - [classroom](https://ftp.nluug.nl/pub/graphics/blender/release/-BlenderBenchmark2.0/scenes/classroom.tar.bz2)
    - [junkshop](https://ftp.nluug.nl/pub/graphics/blender/release/BlenderBenchmark2.0/scenes/junkshop.tar.bz2)
- 推理测试脚本：[blender_benchmark_script](render_sim/render_blender/blender_benchmark_script)（可根据厂商适配）
- 精度对比脚本：[compare_script](render_sim/render_blender/compare_script)


## 暖身轮

在环境准备完成后，首先对渲染场景进行暖身轮测试，测试参考命令，可根据实际环境适配：
```bash
/path/to/blender --background     --factory-startup -noaudio --debug-cycles --enable-autoexec --engine CYCLES /path/to/scene/monster/main.blend   --python /path/to/blender_benchmark_script/main.py -- --device-type OPTIX --warm-up
```

参数说明：
- --engine CYCLES：使用cycles进行渲染
- --device-type OPTIX：cycles的backend使用OPTIX
- --warm-up：进行暖身轮


## 测试执行

在暖身轮结束后，进行测试执行，测试参考命令，可根据实际环境适配：

```bash
/path/to/blender --background     --factory-startup -noaudio --debug-cycles --enable-autoexec --engine CYCLES /path/to/scene/monster/main.blend   --python /path/to/blender_benchmark_script/main.py -- --device-type OPTIX --output monster.png
```

参数说明：
- --output monster.png：如果需要保存渲染图片，可使用此参数指定图片保存路径和图片名。

其余参数同暖身轮测试

## 结果评估

在测试中指定`--output`可输出图片，可以将渲染的图片与[基准图片](render_sim/render_blender/compare_script/baseline_image)进行对比，计算像素级差异、PSNR等指标。对比命令参考：
```bash
pip install -r /path/to/compare_script/requirements.txt

python /path/to/compare_script/main.py /path/to/baseline_image/monster.png  /path/to/render_image/monster2.png
```

