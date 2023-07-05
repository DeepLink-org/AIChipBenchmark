# env

pytorch1.8

```
pip install pytorch-lightning --user

# pytorch-lightning-1.5.10

# dali
wget https://developer.download.nvidia.com/compute/redist/cuda/11.0/nvidia-dali/nvidia_dali-0.22.0-1313465-cp38-cp38-manylinux1_x86_64.whl

pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110


# apex

git clone https://github.com/NVIDIA/apex
cd apex
pip install --user -v --disable-pip-version-check --no-cache-dir \
--global-option="--cpp_ext" --global-option="--cuda_ext" ./

```


model:
```
https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation/nnUNet

a810773

```

# 更新代码：
```
cd DeepLearningExamples
git apply aichipbenchmark/networks/segmentation/nnunet/patch
```


# train process

## get data



```
https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2

mkdir data && cd data

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C' -O task04.tar

tar -xvf task04.tar

cd ..

python preprocess.py --data ./data --results ./data --task 04 --dim 3

```


## accu

```
srun -p $partition --gres=gpu:8 --exclusive --ntasks-per-node 8 -n 8 python main.py --exec_mode train --task 04 --fold 0 --gpus 8 --dim 3 --data `pwd`/data/04_3d --results ./results_accu --epoch 1000

# DLL 2022-05-12 16:31:49.689297 - () dice_score : 92.13  Epoch : 291
```

## perf

In `main.py` (DeepLearningExamples/PyTorch/Segmentation/nnUNet/main.py), the global_batch_size should be:
`global_batch_size=batch_size * args.gpus * args.nodes` (fixed in a810773)


```
srun -p $partition --gres=gpu:8 -n 8 --ntasks-per-node 8 --exclusive  python scripts/benchmark.py --mode train --gpus 8 --dim 3 --batch_size 2 --results ./results --task 04

srun -p $partition --gres=gpu:8 -n 16 --ntasks-per-node 8 --exclusive --quotatype=auto  python scripts/benchmark.py --mode train --gpus 8 --dim 3 --batch_size 2 --results ./results --task 04 --nodes 2

1: DLL 2022-05-12 14:09:13.825000 - () throughput_train : 37.348  latency_train_mean : 53.551  latency_train_90 : 53.683  latency_train_95 : 53.727  latency_train_99 : 53.864
4: DLL 2022-05-17 11:08:38.680184 - () throughput_train : 141.478  latency_train_mean : 56.546  latency_train_90 : 56.778  latency_train_95 : 58.027  latency_train_99 : 59.835
8: DLL 2022-05-17 10:55:26.173050 - () throughput_train : 251.769  latency_train_mean : 63.55  latency_train_90 : 65.695  latency_train_95 : 65.79  latency_train_99 : 67.719
16: DLL 2022-05-17 14:56:10.423549 - () throughput_train : 403.595  latency_train_mean : 79.287  latency_train_90 : 66.903  latency_train_95 : 68.948  latency_train_99 : 157.486
```


# logs

## 数据路径之谜

虽然给传递的路径是 ./data , 但是代码不知何处会在路径前面加上'/',导致路径错误
```
Error when executing CPU operator readers__Numpy, instance name: "ReaderY", encountered:
[/opt/dali/dali/util/std_file.cc:29] Assert on "fp_ != nullptr" failed: Could not open file /./data/04_3d/hippocampus_368_y.npy: No such file or directory
```

解决办法：传递绝对路径