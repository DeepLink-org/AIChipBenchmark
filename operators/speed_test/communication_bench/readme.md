# 通信带宽测试
通信带宽测试采用`osu benchmarks`，在不同通信负载和通信节点条件下，测试All-Reduce 算子（使用sum op）的带宽和时延。

下面介绍工具的使用。

## 1. osu-benchmark下载 & 解压
```
cd comm

wget --no-check-certificate https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-5.9.tar.gz

srun -p your_partition -n 1 -N 1 --gres=gpu:1 tar -zxvf osu-micro-benchmarks-5.9.tar.gz

```
## 2. 编译 & 环境设置

按照如下命令进行编译(编译**需要gpu**):

```sh
cd osu-micro-benchmarks-5.9

export BINDIR=`pwd`/out

# Note: 需要根据环境指定路径，例如`CC=/user/lib/openmpi/bin/mpicc`
srun -p your_partition -n 1 -N 1 --gres=gpu:1 ./configure CC=/path/to/openmpi/bin/mpicc CXX=/path/to/openmpi/bin/mpicxx --enable-cuda --enable-ncclomb --prefix=$BINDIR --with-cuda-include=/path/to/cuda11.0-cudnn8.0/include  --with-cuda-libpath=/path/to/cuda11.0-cudnn8.0/lib64 --with-cuda=/path/to/cuda11.0-cudnn8.0 --with-nccl=/path/to/nccl-2.9.8-cuda11.0

srun -p your_partition -n 1 -N 1 --gres=gpu:1 make && make install

```


## 3. 运行测试
```
cd ../

sh -x test_all.sh your_partition $BINDIR/libexec/osu-micro-benchmarks/nccl/collective result.json

```
测试结果保存在`result.json`中.

