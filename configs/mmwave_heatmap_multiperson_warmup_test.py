"""
Configuration for Multi-Person 2.5D Heatmap with Progressive Training Strategy.

Key Features:
- Coord Loss Warmup: Initially disable coord loss, enable after heatmaps converge
- Layer-wise Learning Rate Decay: Different LR for backbone vs head
- Progressive training strategy for better convergence

Hooks are registered in mmwave3d.hooks module.
"""

import math

# ============================================================================
# Base Configuration
# ============================================================================

default_scope = 'mmwave3d'
custom_imports = dict(imports=['mmwave3d'], allow_failed_imports=False)
work_dir = './work_dirs/mmwave_heatmap_multiperson_warmup_true_run'

backend_args = None
dataset_type = 'MMWaveDataset'
data_root = 'data/'
num_frames = 5
max_epochs = 50
iters_per_epoch = 1326
max_iters = max_epochs * iters_per_epoch
point_cloud_range = [0.12, (30/180)*math.pi, -(30/180)*math.pi, 6.0, (150/180)*math.pi, (30/180)*math.pi]
angle_voxel_size = math.radians(1.0)
data_norm = True
world_size = 4

# Heatmap parameters
range_bins = 60
angle_bins = 90
sigma_r = 0.24
sigma_a = math.radians(6)

randomness = dict(
    seed=3407,
    diff_rank_seed=False
)

# ============================================================================
# Model Architecture
# ============================================================================

model = dict(
    type='MmKeypoints',
    data_preprocessor=dict(
        type='SparseVoxelization',
        point_cloud_range=point_cloud_range,
        data_norm=data_norm,
        voxel_size_r=0.12,  # Increased from 0.12 to reduce voxel count (half resolution)
        voxel_angle_size_deg=math.radians(1.0),  # Increased from 1.0 to reduce voxel count
        expected_frames=num_frames,
        angles_in_degrees=False,
        non_blocking=True
        # Removed pooling params - handled by SpConv in transformer
    ),
    transformer=dict(
        type='MmWaveSpConvTransformerBackbone',
        input_token_dim=3,  # amp + phase + coord features
        pos_dim=4,
        embed_dim=256,
        num_layers=4,
        num_heads=4,
        ffn_dim=512,
        dropout=0.1,
        spconv_channels=(64, 64),  # Sparse conv channels before transformer
        spconv_kernel_size=3,
        enable_learnable_downsample=True,
        downsample_tokens=2048,
        downsample_method='attention',
        temporal_layers=1,  # Reduced from 2 to save memory
        norm_first=True
    ),
    head=dict(
        type='MmWaveHeatmapHead',
        in_dim=256,
        num_person=2, 
        num_joints=17,
        num_frames=num_frames,
        mlp_hidden_dim=512,
        mlp_depth=2,
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
            threshold=0.15,
            reduction='mean',
            loss_weight=0.1
        ),
        coord_loss_weight=5.0,  # Will be controlled by LossWeightWarmupHook
        coord_loss_beta=0.1
    )
)

# ============================================================================
# Data Pipeline
# ============================================================================

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
    dict(
        type='TemporalPhaseFiltering',
        amp_threshold=0.1,
        phase_delta=1.5,
        phase_voxel=1.5,
        amp_relax=0.01,
        static_delta=0.5,
        static_persist=1,
        motion_decay=0.1,
        motion_keep=0.1,
        history_size=3,
        use_phase=False
    ),
    dict(
        type='UniformSampling',
        num_points_per_frame=20000,  # Further reduced: 5 frames = 50k total points
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
    dict(
        type='TemporalPhaseFiltering',
        amp_threshold=0.1,
        phase_delta=1.5,
        phase_voxel=1.5,
        amp_relax=0.01,
        static_delta=0.5,
        static_persist=1,
        motion_decay=0.1,
        motion_keep=0.1,
        history_size=3,
        use_phase=False
    ),
    dict(type='LoadAnnotations3D', with_keypoints_3d=True, eval=True),
    dict(keys=('points', 'gt_keypoints_3d'), type='Pack3DInputs')
]

# ============================================================================
# Dataloaders
# ============================================================================

train_dataloader = dict(
    batch_size=4,
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

# ============================================================================
# Evaluation
# ============================================================================

val_evaluator = [
    dict(type='HeatmapMPJPE'),
    dict(type='MatchedPredictionDump', 
         out_dir='work_dirs/matched_predictions_0117/')
]
test_evaluator = val_evaluator

# ============================================================================
# Training Schedule
# ============================================================================

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=iters_per_epoch,
    val_begin= iters_per_epoch
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================================
# Optimizer & Scheduler
# ============================================================================

# Base LR - will be scaled by LayerDecayLRHook for different modules
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-3),
    clip_grad=dict(max_norm=10.0, norm_type=2),
    accumulative_counts=2  # Gradient accumulation to simulate larger batch size
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
        T_max=int(max_iters/2),
        end=int(max_iters/2),
        by_epoch=False,
        eta_min=1e-7
    )
]

# ============================================================================
# Hooks Configuration
# ============================================================================

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
        interval=iters_per_epoch * 2,  # Save every 2 epochs
        by_epoch=False,
        max_keep_ckpts=3,
        save_best='heatmap/MPJPE',
        rule='less',
        save_last=True
    ),
    sampler_seed=dict(type='DistSamplerSeedHook')
)

# Custom training strategy hooks
custom_hooks = [
    # Phase 1: Loss Weight Warmup
    # Disable coord loss for first 3000 iters (~5 epochs) to let heatmaps converge
    dict(
        type='LossWeightWarmupHook',
        loss_name='coord_loss_weight',
        warmup_iters=5 * iters_per_epoch,      # ~5 epochs
        init_weight=0.0,
        final_weight=2.0        # Target coord loss weight after warmup
    ),
    
    # Phase 2: Layer-wise Learning Rate Decay
    # Assign different LR to different modules for stable training
    dict(
        type='LayerDecayLRHook',
        backbone_lr_mult=1,      # Backbone: 10% of base LR (conservative)
        transformer_lr_mult=1,   # Transformer: 50% of base LR
        decay_start_iter=15000,    # Start progressive decay after ~24 epochs
        decay_factor=1           # Reduce backbone LR to 1% by end
    )
]

# ============================================================================
# Environment & Logging
# ============================================================================

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

log_processor = dict(by_epoch=False, window_size=10)
log_level = 'INFO'
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Visualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# ============================================================================
# Misc
# ============================================================================

auto_scale_lr = dict(enable=False, base_batch_size=8)
load_from = None
resume = False
find_unused_parameters = False
