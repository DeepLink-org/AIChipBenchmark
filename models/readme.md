# 模型测试
模型测试工具按照《测试指标与测试方法》进行模型的精度和性能评测。

模型测试工具基于 Pytorch 框架，对于非Pytorch框架，本仓库的代码和训练使用的超参数可以用于参考，测试方法依照《测试指标与测试方法》进行。

本仓库支持的模型列表如下：

```
├── classification
│   ├── densenet121
│   ├── efficientnet-b2
│   ├── inceptionv3
│   ├── mobilenetv2
│   ├── resnet50
│   ├── senet50
│   ├── shufflenetv2
│   ├── swintransformer
│   └── vgg16
├── detection
│   ├── cascade_rcnn_r50
│   ├── centernet_r18
│   ├── faster_rcnn_r50
│   ├── fcos_r50
│   ├── mask_rcnn_r50
│   ├── retinanet
│   ├── solo
│   ├── ssd300
│   ├── swin
│   └── yolo_v3
├── segmentation
│   ├── apcnet
│   ├── DeeplabV3-R50
│   ├── FCN-R50
│   ├── nnunet
│   └── PSPNet-R50

```

不同的模型其实现代码有所不同，具体如下：
- nnUNet 使用[NVIDIA/DEepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet)的实现
- inceptionv3参考[本仓库实现](/networks/imagenet_example/README.md)
- 其余模型均使用`OpenMMLab`的实现，测试方法参考[OpenMMLab官方文档](https://openmmlab.com/)

对于所有的模型，其仓库均有相应的训练脚本，按照相应仓库的文档进行训练即可。下面提供自动测试工具的使用方法，但该工具需要满足一定的环境条件要求，请参考文档使用。

# 自动测试工具使用流程
`tools`提供了可以在**Slurm环境**下运行的测试脚本。非Slurm环境下不能使用本工具，可以参考本工具的测试流程进行训练任务。


1. 部署pytorch环境，确保环境中使用Pytorch1.8 with CUDA11

2. 设置环境
    - 下载`mmdetection,mmclassification,mmsegmentation,mmcv,nvdia-DeepLeariningExamples`仓库，并设置相应的环境变量：`MMCLS_PATH=/path/to/mmclassification`, `MMCV_PATH=/path/to/mmcv`, `MMDET_PATH=/path/to/mmdetection`, `MMSEG_PATH=/path/to/mmsegmentation`, `NNUNET_PATH=/path/to/DeepLearningExamples`。**若没有设置，则会运行出错**。
      - 下载可以参考如下指令：
          ```sh
          git clone https://github.com/open-mmlab/mmcv.git
          git clone https://github.com/open-mmlab/mmclassification.git
          git clone https://github.com/open-mmlab/mmdetection.git
          git clone https://github.com/open-mmlab/mmsegmentation.git
          git clone https://github.com/NVIDIA/DeepLearningExamples.git
          ```
    版本可参考：[mmcv1.5.3](https://github.com/open-mmlab/mmcv/tree/v1.5.3)、[mmclassification0.23.2](https://github.com/open-mmlab/mmpretrain/tree/v0.23.2)、[mmdetection2.24.1](https://github.com/open-mmlab/mmdetection/tree/v2.24.1)、[mmsegemention0.24.1](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1)
    - mmcv安装参考[官方文档](https://mmcv.readthedocs.io/en/latest/get_started/build.html)。收集性能数据时可参考本目录下的`iter_timer.py`文件，修改`${MMCV_PATH}/mmcv/runner/hooks/iter_timer.py`。并设定环境变量：
        ```sh
        export PYTHONPATH=${MMCV_PATH}:$PYTHONPATH
        ```
    - `export MAX_NODES=4`: 设置Slurm最多占用的节点数量，在测试openmm模型的时候，会查询当前使用的节点数量，如果超过这个值则新的任务会等待。

    - 数据集：
      - OpenMM要求在相应的仓库下将数据集根目录软链接到 `$MMREPO/data`，参考[官方文档](https://mmclassification.readthedocs.io/en/latest/getting_started.html#prepare-datasets)。例如在   `mmclassification`目录，软连接 `mmclassification/data` 到 `/path/to/dataset`.
      - `imagenet_example` 数据集环境参考[imagenet_example文档](/networks/imagenet_example/README.md)

    - nnUNet测试环境，具体测试环境参考官方文档进行搭建，下面给出简单示例：
      - 环境搭建：
        ```sh
        cd DeepLearningExamples/PyTorch/Segmentation/nnUNet
        pip install -r requirements.txt -U

        # install pytorch-lightning-1.5.10

        pip install pytorch-lightning==1.5.10 --user

        # install dali
        wget https://developer.download.nvidia.com/compute/redist/cuda/11.0/nvidia-dali/nvidia_dali-0.22.0-1313465-cp38-cp38-manylinux1_x86_64.whl

        pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

        # install apex

        git clone https://github.com/NVIDIA/apex
        cd apex
        srun -p $partition --gres=gpu:1 -n1 -N 1 python setup.py --cpp_ext --cuda_ext bdist_wheel
        pip install dist/apex*.whl --user

        ```
      - 下载数据集，参考：
        ```sh
        cd $NNUNET_PATH
        mkdir -p data
        pushd data
        # dataset ref: https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2
        wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C' -O task04.tar
        tar -xvf task04.tar
        popd

        python preprocess.py --data ./data --results ./data --task 04 --dim 3
        ```


3. 测试模型：`cd tools && sh run_all_models.sh slurm_partition /path/to/result model_configs.json`

    这一步会使用Slurm测试所有支持的的模型。无法在非Slurm环境下使用。

   - 参数解释：
        - slurm_partition: slurm分区
        - 输出的路径`your/path/to/result`，表示测试过程中的log等文件的存放地，以及测试结果文件 `model_results.json` 的存放地。该路径需要**绝对路径**。
        - 测试结果在 `your/path/to/result/model_results.json`

## 更新与安装指南（PyTorch 2.x 适配版）

由于原有仓库版本较老，在 PyTorch 2.x 版本上存在一些适配性问题，不利于在先进显卡上测试性能，因此使用更新适配的代码仓库。

### 环境要求

- PyTorch 2.x
- CUDA 11.8+ / 12.x
- 支持的显卡架构: Ampere (8.0, 8.6, 8.9), Hopper (9.0, 9.0a)

### 安装步骤

#### 1. 安装 onedl-mmcv（基础库）

```sh
git clone https://github.com/VBTI-development/onedl-mmcv.git
cd onedl-mmcv
git checkout 55264919c4651084882c2ba6f888834aee9a4627

export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;9.0a"
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
python -m pip install -e . -v --no-build-isolation
cd ..
```

#### 2. 安装 onedl-mmdetection（检测）

```sh
git clone https://github.com/VBTI-development/onedl-mmdetection.git
cd onedl-mmdetection
git checkout c43b35b7553db279de8609a321cfd7fa0b733492
pip install -e .
cd ..
```

#### 3. 安装 onedl-mmsegmentation（分割）

```sh
git clone https://github.com/VBTI-development/onedl-mmsegmentation.git
cd onedl-mmsegmentation
git checkout f2dc1d0758593eaec3b257ed185fea35c86e6d26
pip install -e .
cd ..
```

#### 4. 安装 onedl-mmpretrain（分类，原 mmclassification）

```sh
git clone https://github.com/VBTI-development/onedl-mmpretrain.git
cd onedl-mmpretrain
git checkout 128b6079ecc1d089577d1e99b1f786887f48a1c1
pip install -e .
cd ..
```

### 性能测试适配

#### 问题说明

原有的 `iter_timer.py` 由于 `IterTimerHook` 已经从 mmcv 移到了 mmengine，接口有变化，因此无法直接使用。

#### 解决方案

创建 `custom_iter_timer_hook.py`，并放置到以下目录：
- `onedl-mmdetection/`
- `onedl-mmsegmentation/`
- `onedl-mmpretrain/`

#### 配置示例

修改相应的 config 文件加载自定义 hook，例如测试分类模型 `resnet50`，使用pretrain.sh进行测试,默认测试FP32性能精度，可以通过修改optim_wrapper.type=AmpOptimWrapper来测试FP16性能，批量测试脚本为batch_detection.py，batch_pretrain.py，batch_segmentation.py：

```python
_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]

# 导入自定义 hook
custom_imports = dict(imports=['custom_iter_timer_hook'], allow_failed_imports=False)

# 添加性能统计 hook
custom_hooks = [dict(type='CustomIterTimerHook', begin_iter=200, end_iter=500)]

# 禁用默认 timer 避免冲突
default_hooks = dict(timer=None, checkpoint=None)
```