"""
Configuration for Multi-Person (2+ persons) 2.5D Heatmap-based Pose Estimation.

This config demonstrates:
- Support for 2+ persons using Hungarian matching algorithm
- Requires scipy: pip install scipy
- Optimized parameters for multi-person scenarios
"""

import math

default_scope = 'mmwave3d'
custom_imports = dict(imports=['mmwave3d'], allow_failed_imports=False)
work_dir = './work_dirs/mmwave_heatmap_multiperson'

backend_args = None
dataset_type = 'MMWaveDataset'
data_root = 'data/'  # Update to your multi-person dataset
num_frames = 5
max_epochs = 50
iters_per_epoch = 627
max_iters = max_epochs * iters_per_epoch
point_cloud_range = [0.12, (30/180)*math.pi, -(30/180)*math.pi, 6.0, (150/180)*math.pi, (30/180)*math.pi]
angle_voxel_size = math.radians(1.0)
data_norm = True
world_size = 4

# Heatmap parameters
range_bins = 60  # Increased resolution for better multi-person separation
angle_bins = 75
sigma_r = 0.36  # Smaller sigma for sharper peaks (better separation)
sigma_a = math.radians(6)  # ~4 degrees

randomness = dict(
    seed=3407,
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
        # Disable fixed pooling - use learnable downsampling in backbone instead
        voxel_downsample_stages=0,
        non_blocking=True
    ),
    transformer=dict(
        type='MmWaveTransformerBackbone',
        token_dim=1,
        pos_dim=4,
        embed_dim=384,  # Increased capacity for multi-person
        num_layers=6,    # Deeper network
        num_heads=6,
        ffn_dim=768,
        dropout=0.1,
        temporal_layers=2,
        norm_first=True,
        # Learnable downsampling: 2 stages of 3D conv+bn+relu+pool
        learnable_downsample_stages=2,
        learnable_downsample_channels=[16, 32],  # Progressive channel increase
        learnable_downsample_kernel=3,  # 3x3x3 conv kernel
        learnable_downsample_stride=2,  # 2x2x2 pooling stride
        learnable_downsample_pool='max'  # Max pooling for better feature preservation
    ),
    head=dict(
        type='MmWaveHeatmapHead',
        in_dim=384,
        num_person=2, 
        num_joints=17,
        mlp_hidden_dim=768,
        mlp_depth=2,  # Deeper MLP
        range_bins=range_bins,
        angle_bins=angle_bins,
        point_cloud_range=point_cloud_range,
        sigma_r=sigma_r,
        sigma_a=sigma_a,
        loss_heatmap=dict(
            type='Heatmap2DLoss',
            alpha=1.0,
            beta=2.0,
            reduction='mean',
            loss_weight=10
        ),
        loss_elevation=dict(
            type='ElevationLoss',
            threshold=0.15,  # Higher threshold for multi-person
            reduction='mean',
            loss_weight=0.1  # Lower weight for elevation
        ),
        coord_loss_weight=2.0,
        coord_loss_beta=0.1
    )
)

# Training pipeline
train_pipeline = [
    dict(
        type='LoadMmPointsFromFile',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        use_amplitude=True,
        use_phase=True,
        normalize_amplitude=True,
        normalize_phase=False,
        backend_args=backend_args
    ),
    # dict(
    #     type='TemporalPhaseFiltering',
    #     amp_threshold=0.1,      # Much lower threshold
    #     phase_delta=1.5,        # Allow large phase variation
    #     phase_voxel=1.5,        # Much larger voxel
    #     amp_relax=0.1,          # Very relaxed amplitude
    #     static_delta=1.5,       # Allow more static points
    #     static_persist=1,       # Minimum persistence
    #     motion_decay=0.3,       # Less aggressive decay
    #     motion_keep=0.5,        # Keep more motion points
    #     history_size=3,
    #     use_phase=False
    # ),
    dict(
        type='UniformSampling',
        num_points_per_frame=360000,  # More points for multi-person
        sampling_method='random'
    ),
    dict(type='LoadAnnotations3D', with_keypoints_3d=True, eval=False),
    dict(keys=('points', 'gt_keypoints_3d'), type='Pack3DInputs')
]

test_pipeline = [
    dict(
        type='LoadMmPointsFromFile',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        use_amplitude=True,
        use_phase=True,
        normalize_amplitude=True,
        normalize_phase=False,
        backend_args=backend_args
    ),
    # dict(
    #     type='TemporalPhaseFiltering',
    #     amp_threshold=0.1,
    #     phase_delta=1.5,
    #     phase_voxel=1.5,
    #     amp_relax=0.1,
    #     static_delta=1.5,
    #     static_persist=1,
    #     motion_decay=0.3,
    #     motion_keep=0.5,
    #     history_size=3,
    #     use_phase=False
    # ),
    dict(
        type='UniformSampling',
        num_points_per_frame=360000,
        sampling_method='random'
    ),
    dict(type='LoadAnnotations3D', with_keypoints_3d=True, eval=True),
    dict(keys=('points', 'gt_keypoints_3d'), type='Pack3DInputs')
]

# Dataloaders - reduced batch size for memory
train_dataloader = dict(
    batch_size=8,  # Smaller batch for debugging
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/annotations',
        pipeline=train_pipeline,
        num_frames=num_frames,
        frame_interval=1,
        metainfo_file='metainfo.json',
        backend_args=backend_args
    )
)

val_dataloader = dict(
    batch_size=8,
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
        frame_interval=1,
        metainfo_file='metainfo.json',
        load_eval_anns=True,
        backend_args=backend_args
    )
)

test_dataloader = val_dataloader

# Evaluation metrics
val_evaluator = dict(
    type='HeatmapMPJPE',
    decode_from_heatmap=False
)
test_evaluator = val_evaluator

# Training schedule
# train_cfg = dict(
#     type='EpochBasedTrainLoop',
#     max_epochs=max_epochs,
#     val_interval=5
# )
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=iters_per_epoch,
    val_begin=iters_per_epoch
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer - lower learning rate for stability
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-3),
    clip_grad=dict(max_norm=10.0, norm_type=2),
    # accumulative_counts=8
)

# Learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        begin=0,
        end=100,
        by_epoch=False
    ),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=13080,
        end=13080,
        by_epoch=False,
        eta_min=1e-7
    )
]

# Hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=10,
        log_metric_by_epoch=False
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5304,  # 2 epochs -> 5304 iters
        by_epoch=False,
        max_keep_ckpts=3,
        save_best='heatmap/MPJPE',
        rule='less',
        save_last=True
    ),
    sampler_seed=dict(type='DistSamplerSeedHook')
)

custom_hooks = [
    # Phase 1: Loss Weight Warmup
    # Disable coord loss for first 3000 iters (~5 epochs) to let heatmaps converge
    dict(
        type='LossWeightWarmupHook',
        loss_name='coord_loss_weight',
        warmup_iters=3000,      # ~5 epochs
        init_weight=0.0,
        final_weight=2.0        # Target coord loss weight after warmup
    ),
]

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

# Logging
log_processor = dict(by_epoch=False, window_size=10)
log_level = 'INFO'
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Visualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Misc
auto_scale_lr = dict(enable=False, base_batch_size=4)
load_from = None
resume = False
find_unused_parameters = False
