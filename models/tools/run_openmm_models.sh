#!/bin/bash

work_dir=$1
partition=$2
mmconfig=$3


function check_mmmodel_version()
{
    path=$1
    version=$2
    pushd $path
    gitbr=$(git describe --tags)
    echo "current branch: $gitbr"
    echo "target branch: $version"

    if [ "$gitbr" == "$version" ];
    then
        echo "$path intact"
    else
        echo "resinstalling "
        git checkout $version
        SETUPTOOLS_USE_DISTUTILS=stdlib srun -p ${partition} -n 1 -N 1 pip install -e . --user
    fi
    popd
}

function check_dir_exist()
{
    dir=$1
    if [ ! -d $dir ]; then
        echo  "FATAL: $dir does not exist"
        exit 1
    fi
}

function check_file()
{
    file=$1
    if [ ! -e $file ]; then
        echo "FATAL: $file does not exist, maybe download failures!"
        exit 1
    fi
}

function check_repos()
{
    cwd=`pwd`
    if [ ! $MMCV_PATH ];then
        echo "dowloading repo ..."
        MMCV_PATH=${cwd}/../../../mmcv
        export MMCV_PATH=$MMCV_PATH
        if [ ! -d $MMCV_PATH ]; then
            git clone https://github.com/open-mmlab/mmcv.git $MMCV_PATH
        fi
    fi
    if [ ! $MMCLS_PATH ];then
        echo "dowloading repo ..."
        MMCLS_PATH=${cwd}/../../../mmclassification
        export MMCLS_PATH=$MMCLS_PATH
        if [ ! -d $MMCLS_PATH ]; then
            git clone https://github.com/open-mmlab/mmclassification.git $MMCLS_PATH
        fi

    fi
    if [ ! $MMDET_PATH ];then
        echo "dowloading repo ..."
        MMDET_PATH=${cwd}/../../../mmdetection
        export MMDET_PATH=$MMDET_PATH
        if [ ! -d $MMDET_PATH ]; then
            git clone https://github.com/open-mmlab/mmdetection.git $MMDET_PATH
        fi
    fi
    if [ ! $MMSEG_PATH ];then
        echo "dowloading repo ..."
        MMSEG_PATH=${cwd}/../../../mmsegmentation
        export MMSEG_PATH=$MMSEG_PATH
        if [ ! -d $MMSEG_PATH ]; then
            git clone https://github.com/open-mmlab/mmsegmentation.git $MMSEG_PATH
        fi
    fi
    check_dir_exist $MMCV_PATH
    check_dir_exist $MMCLS_PATH
    check_dir_exist $MMDET_PATH
    check_dir_exist $MMSEG_PATH
}

function check_download_file()
{
    dst=$1
    link=$2
    if [ ! -e $dst ]; then
        wget $link -P ~/.cache/torch/hub/checkpoints/
    fi
}

function download_pretrain()
{
    check_download_file ~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth "https://download.pytorch.org/models/resnet50-19c8e357.pth"
    check_download_file ~/.cache/torch/hub/checkpoints/swin_tiny_patch4_window7_224.pth "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
    check_file ~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
    check_file ~/.cache/torch/hub/checkpoints/swin_tiny_patch4_window7_224.pth
}


function check_mmcv_version()
{
    if [ ! $MMCV_PATH ]
    then
        echo "MMCV_PATH is not set!"
        exit
    fi
    version=$1
    echo "target version: $version"
    pushd $MMCV_PATH
    gitbr=$(git name-rev --name-only HEAD)
    arrIN=(${gitbr//// })
    gitbr=${arrIN[1]}
    echo "current version: $gitbr"
    if [ "$gitbr" == "$version" ];
    then
        echo "mmcv intact"
    else
        echo "resinstalling mmcv"
        git checkout $version

        # install with GPU!
        SETUPTOOLS_USE_DISTUTILS=stdlib MMCV_WITH_OPS=1 srun -p ${partition} -n 1 -N 1 --gres=gpu:1 pip install -e . --user
    fi
    popd
}

function wait_openmm_task()
{
    # query squeue, make sure 'openmm_models' are finished
    ret=`squeue --user $USER  | grep openmm_models`
    # echo $ret

    while [ "$ret" != "" ]
    do
        echo "waiting for openmm test to finish"
        sleep 10m
        ret=`squeue --user $USER  | grep openmm_models`
    done
}


check_mmmodel_version $MMCLS_PATH v0.23.0

check_mmmodel_version $MMDET_PATH v2.24.0

check_mmmodel_version $MMSEG_PATH v0.23.0

# test mmcls and mmseg


check_mmcv_version v1.5.0

python test_openmm_models.py --perf --accu --cls cls --work-dir $work_dir --partition ${partition} --config $mmconfig
python test_openmm_models.py --perf --accu --cls seg --work-dir $work_dir --partition ${partition} --config $mmconfig

# python test_openmm_models.py --perf --models resnet50 --work-dir $work_dir --partition ${partition} --config $mmconfig


wait_openmm_task
check_mmcv_version v1.3.17
python test_openmm_models.py --perf --accu --cls det --work-dir $work_dir --partition ${partition} --config $mmconfig
