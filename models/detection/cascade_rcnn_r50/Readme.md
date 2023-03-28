# env
- mmdetection:
    - v2.24.0
- mmcv:
    - mmcv-full 1.3.17

# 精度测试
```
GPUS=8 GPUS_PER_NODE=8 SRUN_ARGS="--exclusive" sh ./tools/slurm_train.sh caif_dev cascade_rcnn ./configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py cases/cascade_rcnn_r50_fpn_1x_coco_acc_gpus8
```


计算每个iter的平均时间
```
srun -p caif_dev --gres=gpu:1 python tools/analysis_tools/analyze_logs.py cal_train_time ./cases/cascade_rcnn_r50_fpn_1x_coco_acc_gpus8/20220511_103549.log.json
```

# 测试结果
精度|
:---:|
epoch=12, box_AP 40.5

