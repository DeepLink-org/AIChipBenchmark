# 通信带宽测试
通信带宽测试采用`osu benchmarks`或者`nccl-test`两种工具都可以，在不同通信负载和通信节点条件下，测试All-Reduce 算子（使用sum op）的带宽和时延。

下面介绍`osu benchmarks`工具的使用。

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

sh -x test_all.sh your_partition $BINDIR/libexec/osu-micro-benchmarks/nccl/collective result.json all-reduce

```
测试结果保存在`result.json`中.

下面介绍`nccl-test`工具的使用。

## 1. nccl-test下载 & 解压

| 工具     | 链接 | 描述  |
|--------------------------|--------|----|
|openmpi-4.1.8.tar.gz| [下载](https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.8.tar.gz)|openmpi4.1.8|
|ucx-1.15.0-rc1-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2|[下载](https://github.com/openucx/ucx/releases/download/v1.15.0-rc1/ucx-1.15.0-rc1-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2)|高性能、工业级的通信库，提供统一的、极致性能的通信抽象层|
|nccl-tests-2.17.9.tar.gz|[下载](https://codeload.github.com/NVIDIA/nccl-tests/tar.gz/refs/tags/v2.17.9)|NVIDIA 官方提供的一套基准测试工具集|

```sh
# 创建密钥

ssh-keygen

# 生成authorized_keys文件

install -m 600 /dev/null /root/.ssh/authorized_keys

# 配置免密登录(所有机器都需要配置，包括本机)

cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys

echo xxxxxx > /root/.ssh/authorized_keys 

# 添加客户端ssh配置(所有机器都需要配置)

vim /etc/ssh/ssh_config
...
Host gpu-node-1
    hostname gpu-node-1
    port 22

Host gpu-node-2
    hostname gpu-node-2
    port 22
...

```
## 2. 编译 & 环境设置

按照如下命令进行安装编译(编译**需要gpu**):

```sh
# 安装UCX,mpirun

tar xvf ucx-1.15.0-rc1-ubuntu22.04-mofed5-cuda12-x86_64.tar.bz2
apt install ./ucx-*.deb

tar xvf openmpi-4.1.8.tar.gz
cd openmpi-4.1.8
./configure --prefix=/usr/local/openmpi/4.1.8 --with-hwloc=internal --with-ucx
make && make install

# 验证

/usr/local/openmpi/4.1.8/bin/ompi_info |grep ucx

# 编译 nccl_test

tar xvf nccl-tests-2.17.9.tar.gz
make MPI=1 NAME_SUFFIX=_mpi MPI_HOME=/usr/local/openmpi/4.1.8 CUDA_HOME=/usr/local/cuda

```

## 3. 运行测试
创建hostfile.txt文件,以2机16卡为例：
```
gpu-node-1 slots=8
gpu-node-2 slots=8
```
在任意一台机器上执行
```
export LD_LIBRARY_PATH=/usr/local/openmpi/4.1.8/lib:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export NCCL_DEBUG=""
/usr/local/openmpi/4.1.8/bin/mpirun \
    --hostfile hostfile.txt \
    -np 16 \
    --allow-run-as-root \
    --mca pml ucx \
    --mca btl ^openib \
    -x LD_LIBRARY_PATH  \
    -x NCCL_SOCKET_IFNAME=bond0 \
    -x NCCL_NVLS_ENABLE=0 \
    -x NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7" \
    -x NCCL_IB_GID_INDEX=3 \
    ./nccl-tests-2.17.9/build/all_reduce_perf_mpi \
    -b 26214400 -e 26214400 -n 1000 -w 10 -g 1
```
完整测试脚本参考test_nccl.sh, 测试结果保存在`result.json`中.