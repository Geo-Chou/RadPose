import math

default_scope = 'mmwave3d'
custom_imports = dict(imports=['mmwave3d'], allow_failed_imports=False)

backend_args = None
dataset_type = 'MMWaveDataset'
data_root = 'data/sit'
num_frames = 5
max_epochs = 50
point_cloud_range = [0.12, (30/180)*math.pi, -(30/180)*math.pi, 6.0, (150/180)*math.pi, (30/180)*math.pi]
angle_voxel_size = math.radians(1.0)

model = dict(
    type='MmKeypoints',
    data_preprocessor=dict(
        type='MmWaveDataPreprocessor',
        point_cloud_range=point_cloud_range,
        voxel_size_r=0.12,
        voxel_angle_size_deg=angle_voxel_size,
        expected_frames=num_frames,
        angles_in_degrees=False,
        voxel_pool_kernel=(2, 3, 3),
        voxel_pool_stride=(2, 3, 3),
        voxel_pool_type='avg',
        voxel_downsample_stages=2,
        non_blocking=True),
    transformer=dict(
        type='MmWaveTransformerBackbone',
        token_dim=1,
        pos_dim=4,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=1024,
        dropout=0.1,
        norm_first=True),
    head=dict(
        type='MmWavePoseHead',
        in_dim=256,
        num_person=4,
        num_joints=17,
        mlp_hidden_dim=512,
        mlp_depth=2,
        loss_mpjpe=dict(type='MPJPELoss', loss_weight=1.0),
        loss_human_count=dict(
            type='HumanCountLoss', loss_weight=0.5)))

train_pipeline = [
    dict(
        type='LoadMmPointsFromFile',
        load_dim=5,
        use_dim=[0, 1, 2],
        backend_args=backend_args),
    dict(type='PointSample', num_points=350000),
    dict(type='LoadAnnotations3D', with_keypoints_3d=True, eval = False),
    dict(keys=('points', 'gt_keypoints_3d'), type='Pack3DInputs')
]

test_pipeline = [
    dict(
        type='LoadMmPointsFromFile',
        load_dim=5,
        use_dim=[0, 1, 2],
        backend_args=backend_args),
    dict(type='PointSample', num_points=350000),
    dict(type='LoadAnnotations3D', with_keypoints_3d=True, eval = True),
    dict(keys=('points', 'gt_keypoints_3d'), type='Pack3DInputs')
]

train_dataloader = dict(
    batch_size=24,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/annotations',
        pipeline=train_pipeline,
        num_frames=num_frames,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=24,
    num_workers=6,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/annotations',
        pipeline=test_pipeline,
        num_frames=num_frames,
        test_mode=True,
        load_eval_anns=True,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(type='MPJPE')
test_evaluator = val_evaluator

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_epochs*100, val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=3e-4, weight_decay=1e-2),
#     clip_grad=dict(max_norm=5.0, norm_type=2))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=6e-5, weight_decay=1e-2),
    clip_grad=dict(max_norm=5.0, norm_type=2))


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs*100,
        by_epoch=False,
        milestones=[25*100, 40*100],
        gamma=0.1)
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=10,
        log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=50,
        by_epoch=False,
        max_keep_ckpts=10,
        save_best='MPJPE',
        rule='less',
        save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(by_epoch=False, window_size=10)
log_level = 'INFO'
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Visualizer', vis_backends=vis_backends, name='visualizer')

auto_scale_lr = dict(enable=False, base_batch_size=2)
randomness = dict(seed=0, deterministic=False)
work_dir = './work_dirs/mmwave_transformer_6'
load_from = None
resume = False
find_unused_parameters = False
