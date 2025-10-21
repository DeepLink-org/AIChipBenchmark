# 渲染评测 （基于genesis）


## 环境准备
- 代码下载：
- 测试脚本：https://github.com/zhouxian/genesis-speed-benchmark.git (aa79858a2de08713f7faee0b69937c2f106457bc)
- Genesis：https://github.com/Genesis-Embodied-AI/Genesis.git (aeabb23e8515471f3c309901dc54ab0b7970934a)


- 环境配置
```bash
pip install genesis-world 
pip install open3d
pip install pybind11
```
- LuisaRender库安装编译参考：https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/visualization.html 

- 为便于使用，官方提供编译完成的 [LuisaRender](https://drive.google.com/drive/folders/1Ah580EIylJJ0v2vGOeSBU_b8zPDWESxS) 库。

## 测试执行

进行测试执行，测试参考命令，可根据实际环境适配：

```bash
python ./go2/vs_isaacgym/test_genesis.py -B 512 -v
```
通过修改 renderer 对渲染方式进行修改
```bash
# renderer=gs.renderers.Rasterizer(), ##########光栅渲染
renderer=gs.renderers.RayTracer() ############光锥渲染
```

参数说明：
- -B：表示 batch size（并行环境数）,传给 n_envs = args.B，决定同时跑多少个一模一样的仿真环境，用来测吞吐量。
- -v：表示 visualize。只要命令行里出现 -v，args.v 就为 True，show_viewer=args.v 就会打开可视化窗口

## 结果评估

  - 单环境帧率（FPS）
FPS_single = 200 / (t1 - t0)
含义：每秒钟能完成多少次“物理仿真 + 渲染相机画面”这一完整循环。
  - 总帧率（总吞吐量）
FPS_total = FPS_single × n_envs
含义：并行运行的 n_envs 个环境合在一起，系统每秒能处理多少帧。

输出示例：
```
[Genesis] [11:45:18] [INFO] Running at 638,534.53 FPS (1247.14 FPS per env, 512 envs).
[Genesis] [11:45:18] [INFO] Running at 637,425.13 FPS (1244.97 FPS per env, 512 envs).
[Genesis] [11:45:18] [INFO] Running at 637,469.20 FPS (1245.06 FPS per env, 512 envs).
[Genesis] [11:45:18] [INFO] Running at 635,926.84 FPS (1242.04 FPS per env, 512 envs).
[Genesis] [11:45:18] [INFO] Running at 636,299.75 FPS (1242.77 FPS per env, 512 envs).
per env: 1,177.00 FPS
total  : 602,621.93 FPS
[Genesis] [11:45:18] [INFO] 💤 Exiting Genesis and caching compiled kernels...
```
