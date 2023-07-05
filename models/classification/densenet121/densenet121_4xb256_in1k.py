model = dict(type='ImageClassifier',
             backbone=dict(type='DenseNet', arch='121'),
             neck=dict(type='GlobalAveragePooling'),
             head=dict(type='LinearClsHead',
                       num_classes=1000,
                       in_channels=1024,
                       loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
dataset_type = 'ImageNet'
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        '.data/imagenet/':
        'openmmlab:s3://openmmlab/datasets/classification/imagenet/',
        'data/imagenet/':
        'openmmlab:s3://openmmlab/datasets/classification/imagenet/'
    }))
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
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
    dict(type='RandomResizedCrop', size=224),
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
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=256,
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
            dict(type='RandomResizedCrop', size=224),
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
            dict(type='Resize', size=(256, -1)),
            dict(type='CenterCrop', crop_size=224),
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
            dict(type='Resize', size=(256, -1)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Normalize',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=1, metric='accuracy')
optimizer = dict(type='SGD',
                 lr=0.8,
                 momentum=0.9,
                 weight_decay=0.0001,
                 nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=2500,
                 warmup_ratio=0.25,
                 step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=90)
checkpoint_config = dict(interval=30)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = 'denseout/accu_newlr'
gpu_ids = range(0, 8)
