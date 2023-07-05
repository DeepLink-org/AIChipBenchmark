#!/bin/bash

if [ $# -lt 2 ] ; then
echo "USAGE: $0 slurm_partition work_dir config_file"
exit 1;
fi

partition=$1
work_dir=$2
mmconfig=${3:-'model_configs.json'}

work_dir=$(realpath "$work_dir")
mmconfig=$(realpath "$mmconfig")

function makedirs()
{
    result_json=${work_dir}/model_results.json

    export result_json
    echo $result_json

    mkdir -p $work_dir

    mm_dir=${work_dir}/mmlogs
    mkdir -p $mm_dir
}

cwd=`pwd`


function wait_for_resource()
{
    # query squeue, make sure 'model_test' are finished
    ret=`squeue --user $USER  | grep model_test`

    while [ "$ret" != "" ]
    do
        echo "waiting for openmm test to finish"
        sleep 1m
        ret=`squeue --user $USER  | grep model_test`
    done
}


function test_inceptionv3()
{
    pushd ${cwd}/../imagenet_example

    echo `pwd`
    sbatch -p ${partition} -n 16 --ntasks-per-node 8 --gres=gpu:8 sbatch_run.sh 1 $result_json
    sbatch -p ${partition} -n 8 --ntasks-per-node 8 --gres=gpu:8 sbatch_run.sh 1 $result_json
    sbatch -p ${partition} -n 4 --ntasks-per-node 4 --gres=gpu:4 sbatch_run.sh 1 $result_json

    sbatch -p ${partition} -n 1 --ntasks-per-node 1 --gres=gpu:1 sbatch_run.sh 1 $result_json
    sbatch -p ${partition} -n 16 --ntasks-per-node 8 --gres=gpu:8 sbatch_run.sh 0 $result_json
    popd

}



function test_openmmmodels()
{
    # source install_openmm_repos.sh
    sh run_openmm_models.sh $mm_dir ${partition} $mmconfig
}

function get_openmm_results()
{
    sleep 10m
    # query squeue, make sure 'openmm_models' are finished
    ret=`squeue --user $USER  | grep openmm`
    # echo $ret

    while [ "$ret" != "" ]
    do
        echo "waiting for openmm test to finish"
        sleep 10m
        ret=`squeue --user $USER  | grep openmm`
    done

    if [ "$ret" == "" ]
    then
        echo "openmm jobs are finished"
        echo $ret
        sleep 1m
        python parse_openmm_results.py --log_root $mm_dir --output $result_json --config $mmconfig
    fi
}

function test_unet()
{
    pushd $cwd
    sh slurm_run_nnunet.sh ${partition} $result_json
    popd
}

# start test
makedirs


test_unet

test_inceptionv3


pushd $cwd

test_openmmmodels

get_openmm_results


popd
