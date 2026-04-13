import math

default_scope = 'mmwave3d'
custom_imports = dict(imports=['mmwave3d'], allow_failed_imports=False)
work_dir = './work_dirs/mmwave_transformer_set_prediction'

backend_args = None
dataset_type = 'MMWaveDataset'
data_root = 'data/'
num_frames = 5
max_epochs = 80
point_cloud_range = [0.12, (30/180)*math.pi, -(30/180)*math.pi, 6.0, (150/180)*math.pi, (30/180)*math.pi]
angle_voxel_size = math.radians(1.0)
data_norm = True
world_size=4


randomness = dict(
    seed=3407,
    # deterministic=True,
    diff_rank_seed=False
)

model = dict(
    type='MmKeypoints',
    data_preprocessor=dict(
        type='MmWaveDataPreprocessor',
        point_cloud_range=point_cloud_range,
        data_norm=data_norm,
        voxel_size_r=0.12,
        voxel_angle_size_deg=angle_voxel_size,
        expected_frames=num_frames,
        angles_in_degrees=False,
        voxel_pool_kernel=[(3, 2, 2),(3, 2, 2)],
        voxel_pool_stride=[(2, 2, 2), (2, 2, 2)],
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
        ffn_dim=512,
        dropout=0.1,
        norm_first=True),
    head=dict(
        type='MmWavePoseHeadV2',
        in_dim=256,
        num_person=4,
        num_joints=17,
        mlp_hidden_dim=512,
        mlp_depth=2,
        data_norm=data_norm,
        point_cloud_range=point_cloud_range,
        use_set_prediction=True,
        loss_coord=dict(
            type='SetPredictionLossV2',
            coord_weight=1.0,
            no_object_weight=0.5,
            conf_weight=1.0,
            loss_weight=1.0,
            # Sinkhorn soft matching
            use_sinkhorn=True,  # True: Sinkhorn, False: Hungarian
            sinkhorn_iters=100,  # Sinkhorn iter
            sinkhorn_temperature=0.1)))  # Temperature parameter, lower is harder

train_pipeline = [
    dict(
        type='LoadMmPointsFromFile',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        use_amplitude = True,
        use_phase = True,
        normalize_amplitude = True,
        normalize_phase = False,
        backend_args=backend_args),
    dict(type='PointSample', num_points=380000),
    dict(type='LoadAnnotations3D', with_keypoints_3d=True, eval = False),
    dict(keys=('points', 'gt_keypoints_3d'), type='Pack3DInputs')
]

test_pipeline = [
    dict(
        type='LoadMmPointsFromFile',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        use_amplitude = True,
        use_phase = True,
        normalize_amplitude = True,
        normalize_phase = False,
        backend_args=backend_args),
    dict(type='PointSample', num_points=380000),
    dict(type='LoadAnnotations3D', with_keypoints_3d=True, eval = True),
    dict(keys=('points', 'gt_keypoints_3d'), type='Pack3DInputs')
]

train_dataloader = dict(
    batch_size=24,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/annotations',
        pipeline=train_pipeline,
        num_frames=num_frames,
        frame_interval = 1,
        metainfo_file='metainfo.json',
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=24,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/annotations',
        pipeline=test_pipeline,
        num_frames=num_frames,
        test_mode=True,
        frame_interval = 1,
        metainfo_file='metainfo.json',
        load_eval_anns=True,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(type='MPJPE')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=max_epochs,
    val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=3e-4, weight_decay=1e-3),
#     clip_grad=dict(max_norm=10.0, norm_type=2))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    # BUG FIX
    optimizer=dict(type='AdamW', lr=5e-4, weight_decay=1e-5),
    clip_grad=dict(max_norm=10.0, norm_type=2),
    # accumulative_counts=4,
    ) 

param_scheduler = [
    # dict(
    #     type='LinearLR',
    #     start_factor=0.01,
    #     begin=0,
    #     end=500,
    #     by_epoch=False),
    dict(
        type='CosineAnnealingLR',
        begin=500,
        T_max=50*221,
        end=80*221,
        by_epoch=False,
        eta_min=1e-6)
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=10,
        log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        by_epoch=True,
        max_keep_ckpts=10,
        save_best='MPJPE',
        rule='less',
        save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(by_epoch=True, window_size=10)
log_level = 'INFO'
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Visualizer', vis_backends=vis_backends, name='visualizer')

auto_scale_lr = dict(enable=False, base_batch_size=2)

load_from = None
resume = False
find_unused_parameters = False
