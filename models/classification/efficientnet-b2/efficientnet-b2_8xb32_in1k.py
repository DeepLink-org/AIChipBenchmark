model = dict(type='ImageClassifier',
             backbone=dict(type='EfficientNet', arch='b2'),
             neck=dict(type='GlobalAveragePooling'),
             head=dict(type='LinearClsHead',
                       num_classes=1000,
                       in_channels=1408,
                       loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                       topk=(1, 5)))
dataset_type = 'ImageNet'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        '.data/imagenet/':
        'openmmlab:s3://openmmlab/datasets/classification/imagenet/',
        'data/imagenet/':
        'openmmlab:s3://openmmlab/datasets/classification/imagenet/'
    }))
train_pipeline = [
    dict(type='LoadImageFromFile',
         file_client_args=dict(
             backend='petrel',
             path_mapping=dict({
                 '.data/imagenet/':
                 'openmmlab:s3://openmmlab/datasets/classification/imagenet/',
                 'data/imagenet/':
                 'openmmlab:s3://openmmlab/datasets/classification/imagenet/'
             }))),
    dict(type='RandomResizedCrop',
         size=260,
         efficientnet_style=True,
         interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile',
         file_client_args=dict(
             backend='petrel',
             path_mapping=dict({
                 '.data/imagenet/':
                 'openmmlab:s3://openmmlab/datasets/classification/imagenet/',
                 'data/imagenet/':
                 'openmmlab:s3://openmmlab/datasets/classification/imagenet/'
             }))),
    dict(type='CenterCrop',
         crop_size=260,
         efficientnet_style=True,
         interpolation='bicubic'),
    dict(type='Normalize',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='ImageNet',
        data_prefix='data/imagenet/train',
        ann_file='data/imagenet/meta/train.txt',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(
                    backend='petrel',
                    path_mapping=dict({
                        '.data/imagenet/':
                        'openmmlab:s3://openmmlab/datasets/classification/imagenet/',
                        'data/imagenet/':
                        'openmmlab:s3://openmmlab/datasets/classification/imagenet/'
                    }))),
            dict(type='RandomResizedCrop',
                 size=260,
                 efficientnet_style=True,
                 interpolation='bicubic'),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(type='Normalize',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='ImageNet',
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(
                    backend='petrel',
                    path_mapping=dict({
                        '.data/imagenet/':
                        'openmmlab:s3://openmmlab/datasets/classification/imagenet/',
                        'data/imagenet/':
                        'openmmlab:s3://openmmlab/datasets/classification/imagenet/'
                    }))),
            dict(type='CenterCrop',
                 crop_size=260,
                 efficientnet_style=True,
                 interpolation='bicubic'),
            dict(type='Normalize',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='ImageNet',
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(
                    backend='petrel',
                    path_mapping=dict({
                        '.data/imagenet/':
                        'openmmlab:s3://openmmlab/datasets/classification/imagenet/',
                        'data/imagenet/':
                        'openmmlab:s3://openmmlab/datasets/classification/imagenet/'
                    }))),
            dict(type='CenterCrop',
                 crop_size=260,
                 efficientnet_style=True,
                 interpolation='bicubic'),
            dict(type='Normalize',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=1, metric='accuracy')
optimizer = dict(type='RMSprop',
                 lr=0.01,
                 momentum=0.9,
                 weight_decay=1e-05,
                 alpha=0.9,
                 eps=0.01)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing',
                 min_lr=0,
                 warmup='linear',
                 warmup_iters=16,
                 warmup_ratio=0.0001,
                 warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=400)
checkpoint_config = dict(interval=100)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = 'effiout/accu-rmpsprop'
gpu_ids = range(0, 16)
