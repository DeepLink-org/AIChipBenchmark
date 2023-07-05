# 在S集群执行测试
此文档的一些相关环境设置仅适用于S集群。

# 模型测试
测试的模型如下：

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

每个目录下均有相关的环境和命令说明。


- 除了nnunet和ineptionv3外，其余模型均为openmmLab支持的，测试方法参考 [OPENMM 模型测试](/networks/openmm_readme.md)


# 测试流程

1. 部署pytorch环境
   在S集群上，可以运行 `source /mnt/cache/share_data/caif/pt1.8_model`

2. 设置环境
   - 下载`mmdet,mmcls,mmseg,mmcv, nvdia-DeepLeariningExamples`仓库，并设置相应的环境变量：`MMCLS_PATH`, `MMCV_PATH`, `MMDET_PATH`, `MMSEG_PATH`, `NNUNET_PATH`。**若没有设置，则会运行出错**。
     - 下载可以参考如下指令：
        ```sh
        git clone https://github.com/open-mmlab/mmcv.git
        git clone https://github.com/open-mmlab/mmclassification.git
        git clone https://github.com/open-mmlab/mmdetection.git
        git clone https://github.com/open-mmlab/mmsegmentation.git
        git clone https://github.com/NVIDIA/DeepLearningExamples.git
        ```

   - `export MAX_NODES=4`: 设置最多占用的节点数量，在测试openmm模型的时候，会查询当前使用的节点数量，如果超过这个值则新的任务会等待。

   - `export PYTHONPATH=/mnt/lustre/share/pymc/new:$PYTHONPATH` 设置inceptionv3测试使用的环境

   - 设置ceph环境：
     - `cp /mnt/cache/share_data/caif/ceph_env/petreloss.conf ~`
     - `cp /mnt/cache/share_data/caif/ceph_env/.s3cfg ~`

   - ceph环境下，需要拷贝meta文件到本地：
        ```sh
        cd mmclassification
        mkdir -p data/imagenet/meta
        aws s3 cp s3://openmmlab/datasets/classification/imagenet/meta/train.txt --no-sign-request data/imagenet/meta/train.txt
        aws s3 cp s3://openmmlab/datasets/classification/imagenet/meta/val.txt --no-sign-request data/imagenet/meta/val.txt

        cd mmdetection
        mkdir -p data/coco/annotations
        aws s3 cp s3://openmmlab/datasets/detection/coco/annotations/instances_train2017.json --no-sign-request ./data/coco/annotations
        aws s3 cp s3://openmmlab/datasets/detection/coco/annotations/instances_val2017.json --no-sign-request ./data/coco/annotations
        ```

    - 下载预训练模型：
        ```sh
        Download: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to ~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
        Download: "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth" to ~/.cache/torch/hub/checkpoints/swin_tiny_patch4_window7_224.pth
        ```

    - `imagenet_example` 要求在 `imagenet_example` 下创建软链接 `data/imagenet` 到 Imagenet 目录，目录结构为:
        ```
        data/imagenet/
        ├── meta
        ├── test
        ├── train
        └── val
        ```

3. 测试模型：`cd tools_internal && sh run_all_models.sh your/path/to/result`

    这一步会测试 inceptionv3, nnnet, openmm的模型。会发起很多测试任务。如果想要测试较少的模型，请查看相应的文档。

   - 参数解释：
     - 输出的路径`your/path/to/result`，表示测试过程中的log等文件的存放地，以及测试结果文件 `model_results.json` 的存放地。该路径需要**绝对路径**。
   - 测试结果：
    - 在 `your/path/to/result/model_results.json`


# 自定测试内容
如果你不想测试所有的模型，可以根据需要测试单个的网络

## 测试inceptionv3

inceptionv3不在Openmm的支持列表中，这里从`parrots.example`移植了部分代码，放在`imagenet_example`中。

### 测试方法：
1. source pytorch环境
2. `cd imagenet_example`
3. 运行测试
  ```sh
    # 16卡性能：
    sbatch -p caif_dev -n 16 --ntasks-per-node 8 --gres=gpu:8 sbatch_run.sh 1 $result_json

    # 精度测试：
    sbatch -p caif_dev -n 16 --ntasks-per-node 8 --gres=gpu:8 sbatch_run.sh 0 $result_json
  ```

## 测试nnunet
1. source pytorch环境
2. 运行所有测试
    ```
    sh -x run_nnunet.sh `pwd`/work_tmp/model_results.json
    ```
3. 若想运行单个测试，参考 [nnunet文档](/networks/segmentation/nnunet/readme_nv.md)

## 测试openmm模型
参考[OPENMM 模型测试](/openmm_readme.md)