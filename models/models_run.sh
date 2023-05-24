#!/bin/bash

# ngpus=(1 2 4 8 16)
ngpus=(1 2 4 8)
work_dir='perf_workdir'
mkdir ${work_dir}/infolog
mkdir ${work_dir}/testlog

if [ 'mmcls' == $1 ];then
    frame=$1
    models=(
        "densenet/densenet121_4xb256_in1k.py"
        "efficientnet/efficientnet-b2_8xb32_in1k.py"
        "mobilenet_v2/mobilenet-v2_8xb32_in1k.py"
        "resnet/resnet50_8xb32_in1k.py"
        "seresnet/seresnet50_8xb32_in1k.py"
        "shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py"
        "swin_transformer/swin-large_16xb64_in1k.py"
        "vgg/vgg16_8xb32_in1k.py"
    )
elif [ 'mmdet' == $1 ];then
    frame=$1
    models=(
       "cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py"
       "centernet/centernet_resnet18_140e_coco.py"
       "faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
       "fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_1x_coco.py"
       "mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py"
       "retinanet/retinanet_r50_fpn_1x_coco.py"
       "solo/decoupled_solo_r50_fpn_1x_coco.py"
       "ssd/ssd300_coco.py"
       "swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py"
       "yolo/yolov3_d53_320_273e_coco.py"
    )
elif [ 'mmseg' == $1 ];then
    frame=$1
    models=(
        "apcnet/apcnet_r50-d8_512x1024_40k_cityscapes.py"
        "deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py"
        "fcn/fcn_r50-d8_512x1024_40k_cityscapes.py"
        "pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py"
    )
fi

for model in ${models[@]};do
    for n in ${ngpus[@]};do
        ngpu_per_node=$(($n>8?8:$n))
        let nnode=(n+7)/8
        ls ${frame}/configs/$model
        set -x
        now=$(date +"%Y%m%d_%H%M%S")
        #srun --exclusive --mpi=pmi2 -n$n -p partition --gres=gpu:$ngpu_per_node --ntasks-per-node=$ngpu_per_node -N $nnode python -u ${frame}/tools/train.py ${frame}/configs/$model --launcher slurm --work-dir=$work_dir/${frame}/${model%/*}_${n} 2>&1 >  infolog/${frame}_${model%/*}_${n}.log | tee testlog/${frame}_${model%/*}_${n}.log
        sh ${frame}/tools/dist_train.sh ${frame}/configs/$model $n --work-dir=$work_dir/${frame}/${model%/*}_${n} 2>&1 >  ${work_dir}/infolog/${frame}_${model%/*}_${n}.log | tee ${work_dir}/testlog/${frame}_${model%/*}_${n}.log.${now}
        sleep 10
    done
done
