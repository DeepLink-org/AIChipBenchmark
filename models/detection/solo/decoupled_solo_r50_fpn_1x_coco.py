dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        '.data/coco/':
        'openmmlab:s3://openmmlab/datasets/detection/coco/',
        'data/coco/':
        'openmmlab:s3://openmmlab/datasets/detection/coco/'
    }))
train_pipeline = [
    dict(type='LoadImageFromFile',
         file_client_args=dict(
             backend='petrel',
             path_mapping=dict({
                 '.data/coco/':
                 'openmmlab:s3://openmmlab/datasets/detection/coco/',
                 'data/coco/':
                 'openmmlab:s3://openmmlab/datasets/detection/coco/'
             }))),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile',
         file_client_args=dict(
             backend='petrel',
             path_mapping=dict({
                 '.data/coco/':
                 'openmmlab:s3://openmmlab/datasets/detection/coco/',
                 'data/coco/':
                 'openmmlab:s3://openmmlab/datasets/detection/coco/'
             }))),
    dict(type='MultiScaleFlipAug',
         img_scale=(1333, 800),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize',
                  mean=[123.675, 116.28, 103.53],
                  std=[58.395, 57.12, 57.375],
                  to_rgb=True),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img'])
         ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_train2017.json',
        img_prefix='data/coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile',
                 file_client_args=dict(
                     backend='petrel',
                     path_mapping=dict({
                         '.data/coco/':
                         'openmmlab:s3://openmmlab/datasets/detection/coco/',
                         'data/coco/':
                         'openmmlab:s3://openmmlab/datasets/detection/coco/'
                     }))),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect',
                 keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ],
        file_client_args=dict(
            backend='petrel',
            path_mapping=dict({
                '.data/coco/':
                'openmmlab:s3://openmmlab/datasets/detection/coco/',
                'data/coco/':
                'openmmlab:s3://openmmlab/datasets/detection/coco/'
            }))),
    val=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile',
                 file_client_args=dict(
                     backend='petrel',
                     path_mapping=dict({
                         '.data/coco/':
                         'openmmlab:s3://openmmlab/datasets/detection/coco/',
                         'data/coco/':
                         'openmmlab:s3://openmmlab/datasets/detection/coco/'
                     }))),
            dict(type='MultiScaleFlipAug',
                 img_scale=(1333, 800),
                 flip=False,
                 transforms=[
                     dict(type='Resize', keep_ratio=True),
                     dict(type='RandomFlip'),
                     dict(type='Normalize',
                          mean=[123.675, 116.28, 103.53],
                          std=[58.395, 57.12, 57.375],
                          to_rgb=True),
                     dict(type='Pad', size_divisor=32),
                     dict(type='ImageToTensor', keys=['img']),
                     dict(type='Collect', keys=['img'])
                 ])
        ],
        file_client_args=dict(
            backend='petrel',
            path_mapping=dict({
                '.data/coco/':
                'openmmlab:s3://openmmlab/datasets/detection/coco/',
                'data/coco/':
                'openmmlab:s3://openmmlab/datasets/detection/coco/'
            }))),
    test=dict(
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile',
                 file_client_args=dict(
                     backend='petrel',
                     path_mapping=dict({
                         '.data/coco/':
                         'openmmlab:s3://openmmlab/datasets/detection/coco/',
                         'data/coco/':
                         'openmmlab:s3://openmmlab/datasets/detection/coco/'
                     }))),
            dict(type='MultiScaleFlipAug',
                 img_scale=(1333, 800),
                 flip=False,
                 transforms=[
                     dict(type='Resize', keep_ratio=True),
                     dict(type='RandomFlip'),
                     dict(type='Normalize',
                          mean=[123.675, 116.28, 103.53],
                          std=[58.395, 57.12, 57.375],
                          to_rgb=True),
                     dict(type='Pad', size_divisor=32),
                     dict(type='ImageToTensor', keys=['img']),
                     dict(type='Collect', keys=['img'])
                 ])
        ],
        file_client_args=dict(
            backend='petrel',
            path_mapping=dict({
                '.data/coco/':
                'openmmlab:s3://openmmlab/datasets/detection/coco/',
                'data/coco/':
                'openmmlab:s3://openmmlab/datasets/detection/coco/'
            }))))
evaluation = dict(metric=['bbox', 'segm'])
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=0.001,
                 step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=12)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
model = dict(type='SOLO',
             backbone=dict(type='ResNet',
                           depth=50,
                           num_stages=4,
                           out_indices=(0, 1, 2, 3),
                           frozen_stages=1,
                           init_cfg=dict(type='Pretrained',
                                         checkpoint='torchvision://resnet50'),
                           style='pytorch'),
             neck=dict(type='FPN',
                       in_channels=[256, 512, 1024, 2048],
                       out_channels=256,
                       start_level=0,
                       num_outs=5),
             mask_head=dict(type='DecoupledSOLOHead',
                            num_classes=80,
                            in_channels=256,
                            stacked_convs=7,
                            feat_channels=256,
                            strides=[8, 8, 16, 32, 32],
                            scale_ranges=((1, 96), (48, 192), (96, 384),
                                          (192, 768), (384, 2048)),
                            pos_scale=0.2,
                            num_grids=[40, 36, 24, 16, 12],
                            cls_down_index=0,
                            loss_mask=dict(type='DiceLoss',
                                           use_sigmoid=True,
                                           loss_weight=3.0,
                                           activate=False),
                            loss_cls=dict(type='FocalLoss',
                                          use_sigmoid=True,
                                          gamma=2.0,
                                          alpha=0.25,
                                          loss_weight=1.0),
                            norm_cfg=dict(type='GN',
                                          num_groups=32,
                                          requires_grad=True)),
             test_cfg=dict(nms_pre=500,
                           score_thr=0.1,
                           mask_thr=0.5,
                           filter_thr=0.05,
                           kernel='gaussian',
                           sigma=2.0,
                           max_per_img=100))
work_dir = 'cases/decoupled_solo_r50_fpn_1x_coco_acc_gpus8'
auto_resume = False
gpu_ids = range(0, 8)
