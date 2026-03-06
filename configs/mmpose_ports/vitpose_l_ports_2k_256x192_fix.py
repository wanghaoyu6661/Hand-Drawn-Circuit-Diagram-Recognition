# /root/autodl-tmp/mmpose/configs/ports/vitpose_l_ports_2k_256x192_fix.py

_base_ = [
    '../../third_party/mmpose/configs/_base_/default_runtime.py'
]

data_root = '/root/autodl-tmp/vitpose_ports_dataset/ports_2k'
data_mode = 'topdown'
dataset_type = 'CocoDataset'

dataset_meta = dict(
    dataset_name='ports_2k',
    joint_weights=[1.0, 1.0],   # ✅ 2 个
    keypoint_info=dict({
        0: dict(id=0, name='k0', color=[255,0,0], swap='', type=''),
        1: dict(id=1, name='k1', color=[0,255,0], swap='', type=''),
    }),
    sigmas=[0.025, 0.025],      # ✅ 2 个
    skeleton_info=dict()
)


default_scope = 'mmpose'
log_level = 'INFO'

custom_imports = dict(
    imports=[
        'mmpose.engine.optim_wrappers.layer_decay_optim_wrapper',
        'mmpretrain.models',
    ],
    allow_failed_imports=False
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=1, save_best='coco/AP', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False),
)

custom_hooks = [dict(type='SyncBuffersHook')]

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
    ),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='large',
        img_size=(256, 192),
        patch_size=16,
        with_cls_token=False,
        qkv_bias=True,
        drop_path_rate=0.5,
        out_type='featmap',
        patch_cfg=dict(padding=2),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=1024,
        out_channels=2,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(type='UDPHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2),
    ),
    test_cfg=dict(flip_test=False),
)

load_from = '/root/autodl-tmp/weights/vitpose/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth'

codec = dict(type='UDPHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomHalfBody', prob=0.0),
    dict(type='RandomBBoxTransform', scale_factor=(0.75, 1.25), rotate_factor=40, shift_factor=0.0),
    dict(type='TopdownAffine', input_size=(192, 256), use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256), use_udp=True),
    dict(type='PackPoseInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=dataset_meta,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=dataset_meta,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'),
        pipeline=val_pipeline,
        test_mode=True,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=f'{data_root}/annotations/val.json')
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    # warmup: 前 500 iter 线性升到正常 lr
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    # 50 epoch 内衰减两次（比 cosine 更“稳妥可控”）
    dict(
        type='MultiStepLR',
        begin=0,
        end=30,
        by_epoch=True,
        milestones=[20, 27],
        gamma=0.1
    )
]

train_cfg = dict(
    _delete_=True,          # ✅ 关键：删除 base 里继承过来的 by_epoch 等旧字段
    type='EpochBasedTrainLoop',
    max_epochs=30,
    val_interval=1
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

auto_scale_lr = dict(base_batch_size=512)
