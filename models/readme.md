# open-mmlab模型多卡训练性能测试

## 1. 拉取模型代码
建议版本：[mmcv1.5.3]<https://github.com/open-mmlab/mmcv/tree/v1.5.3>、[mmclassification0.23.2]<https://github.com/open-mmlab/mmpretrain/tree/v0.23.2>、[mmdetection2.24.1]<https://github.com/open-mmlab/mmdetection/tree/v2.24.1>、[mmsegemention0.24.1]<https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1>
命令参考：
```
git clone https://github.com/open-mmlab/mmcv.git
git clone https://github.com/open-mmlab/mmclassification.git
git clone https://github.com/open-mmlab/mmdetection.git
git clone https://github.com/open-mmlab/mmsegmentation.git
```

## 2. 安装mmcv，替换hook文件
mmcv安装参考官网 <https://mmcv.readthedocs.io/en/latest/get_started/build.html>
将 ${MMCV_PATH}/mmcv/runner/hooks/iter_timer.py  替换为本目录下的 iter_timer.py文件。
指定环境变量
```sh
export PYTHONPATH=${MMCV_PATH}:$PYTHONPATH
```

## 3. 运行测试脚本
```sh
sh models_run.sh {frame}
```
训练日志保存在'perf_workdir'文件夹。
