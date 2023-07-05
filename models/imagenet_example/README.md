# test imagenet model perf and accuracy

## 介绍
本仓库实现了在Imagenet数据集训练InceptionV3的代码.

## 数据集环境
运行训练前, 需要在 `imagenet_example` 下创建软链接 `data/imagenet` 到 Imagenet 目录，目录结构为:
```
data/imagenet/
├── meta
├── test
├── train
└── val
```

## Slurm环境使用
- 性能测试：
`sbatch -p {your_partition} -n ngpu --ntasks-per-node ngpu --gres=gpu:ngpu sbatch_run.sh 1 your/output.json`

- 精度测试：
`sbatch -p {your_partition} -n ngpu --ntasks-per-node ngpu --gres=gpu:ngpu sbatch_run.sh 0 your/output.json`

结果会存到 `your/output.json` 中。