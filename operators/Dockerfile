FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

WORKDIR /workspace/test
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y python3-pip
RUN pip install packaging

# 更新 apt 并安装基本软件包
RUN apt-get update && apt-get install -y \
    git \
    python3.9 \
    python3.9-distutils \
    python3.9-venv \
    python3.9-dev \
    vim \
    wget \
    openmpi-bin \
    libopenmpi-dev \
    libopenblas-dev \
    mpich \
    g++  \
    libgl1-mesa-glx  \
    libglib2.0-0

# 设置 python 指向 python3.9
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN pip install --upgrade pip setuptools

# 下载和安装 CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.5/cmake-3.14.5-Linux-x86_64.tar.gz && \
    tar -xzvf cmake-3.14.5-Linux-x86_64.tar.gz && \
    find cmake-3.14.5-Linux-x86_64 -type f -not -path '*/man/*' -exec cp -a {} /usr/local/ \; && \
    rm -rf cmake-3.14.5-Linux-x86_64 && \
    rm cmake-3.14.5-Linux-x86_64.tar.gz

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# 安装固定版本的依赖库
RUN pip install pytz==2023.3  --no-deps
RUN pip install requests==2.28.2  --no-deps
RUN pip install rich==13.4.2  --no-deps
RUN pip install tqdm==4.65.0  --no-deps

# 更新 setuptools 和安装 PyTorch 及相关库，兼容 CUDA 11.8
RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# 安装 mmcv-full 和 mmclassification 特定版本
RUN pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# 安装 mmengine
RUN pip install mmengine
RUN pip install openmim
RUN mim install mmdet
RUN mim install mmcv-full
RUN pip install mmsegmentation
RUN mim install mmcls

# 设置 LD_LIBRARY_PATH 和 CUDA_HOME 等环境变量
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME /usr/local/cuda-11.8
ENV PATH $PATH:$CUDA_HOME/bin

# 安装 NVIDIA DALI
#RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/ nvidia-dali-cuda110

# 安装 NVIDIA DALI TensorFlow插件
#RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/ nvidia-dali-tf-plugin-cuda110

# 克隆并安装NVIDIA Apex
RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    python setup.py install
# 设置容器启动时默认命令
CMD ["/bin/bash"]
