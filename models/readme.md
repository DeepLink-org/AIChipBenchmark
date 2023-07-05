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

