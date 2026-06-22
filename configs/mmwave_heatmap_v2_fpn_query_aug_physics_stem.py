"""
mmWave heatmap config with a physics-aware anisotropic sparse stem.

This keeps the original `mmwave_heatmap_v2_fpn_query_aug` pipeline unchanged
except for the local feature extraction stage inside the transformer backbone.
"""

import math

default_scope = 'mmwave3d'
custom_imports = dict(imports=['mmwave3d'], allow_failed_imports=False)
work_dir = './work_dirs/mmpose_outdoor_heatmap'

backend_args = None
dataset_type = 'MMWaveDataset'
data_root = 'data_outdoor/'
num_frames = 5
max_epochs = 50
iters_per_epoch = 213
max_iters = max_epochs * iters_per_epoch
point_cloud_range = [0.12, (30 / 180) * math.pi, -(30 / 180) * math.pi,
                     6.0, (150 / 180) * math.pi, (30 / 180) * math.pi]
angle_voxel_size = math.radians(1.0)
data_norm = True
world_size = 4

range_bins = 150
angle_bins = 180
sigma_r = 0.12
sigma_a = math.radians(1.25)

randomness = dict(
    seed=3407,
    diff_rank_seed=False
)

model = dict(
    type='MmKeypoints',
    data_preprocessor=dict(
        type='SparseVoxelization',
        point_cloud_range=point_cloud_range,
        data_norm=data_norm,
        voxel_size_r=0.06,
        voxel_angle_size_deg=math.radians(0.5),
        expected_frames=num_frames,
        angles_in_degrees=False,
        max_tokens_per_sample=5120,
        non_blocking=True,
    ),
    transformer=dict(
        type='PhysicsAwareFPNFrameQueryTransformer',
        input_token_dim=3,
        pos_dim=4,
        embed_dim=512,
        num_layers=6,
        num_heads=4,
        ffn_dim=512,
        dropout=0.1,
        fpn_channels=(128, 256, 512),
        spconv_channels=(64, 64),
        spconv_kernel_size=3,
        temporal_layers=2,
        num_query_tokens=576,
        downsample_tokens=2048,
        downsample_method='attention',
        non_center_downsample_tokens=512,
        norm_first=True,
        pos_freq_bands=(1.0, 2.0, 4.0, 8.0, 16.0),
        use_isotropic_branch=True,
    ),
    head=dict(
        type='CNNHeatmapHead',
        in_dim=512,             # Matches transformer embed_dim
        num_person=1,
        num_joints=16,
        num_frames=num_frames,
        # CNN Decoder configuration
        use_matching = False,
        initial_spatial_size=24,
        decoder_channels=(512, 256, 128),  # Progressive upsampling
        # Heatmap resolution
        range_bins=range_bins,
        angle_bins=angle_bins,
        point_cloud_range=point_cloud_range,
        sigma_r=sigma_r,
        sigma_a=sigma_a,
        predict_uncertainty=True,
        log_sigma_min=-7.0,
        log_sigma_max=7.0,
        # Losses
        loss_heatmap=dict(
            type='Heatmap2DLoss',
            alpha=1.0,
            beta=2.0,
            reduction='mean',
            loss_weight= 100
        ),
        loss_elevation=dict(
            type='ElevationLoss',
            threshold=0.15,
            reduction='mean',
            loss_weight=1.5
        ),
        loss_skeleton_angle=dict(
            type='SkeletonAngleLoss',
            reduction='mean',
            loss_weight=0.3,
            spherical_input=True,
            use_arccos=True
        ),
        loss_skeleton_length=dict(
            type='SkeletonLengthLoss',
            reduction='mean',
            loss_weight=3.5,
            spherical_input=True
        ),
        coord_loss_weight=5.0,
        coord_loss_beta=0.1,
        decode_inference_with_argmax=True
    )
)


train_pipeline = [
    dict(
        type='LoadMmPointsFromFile',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        use_amplitude=True,
        use_phase=True, 
        normalize_amplitude=True,
        normalize_phase=True,
        backend_args=backend_args
    ),
    dict(
        type='TemporalPhaseFiltering',
        amp_threshold=0.5,
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
    # dict(
    #     type='MmWaveJitterAugmentation',
    #     range_jitter=(0.01, 0.03),
    #     angle_jitter_deg=(0.2, 0.6),
    #     phase_noise_deg=None,
    #     amp_noise_pct=(0.01, 0.075),
    #     amp_scale=(0.85, 1.15),
    #     angles_in_degrees=False,
    #     phase_in_degrees=True,
    #     phase_normalized=True,
    #     phase_range_degrees=360.0,
    #     clip_to_original=True,
    #     apply_prob=0.8
    # ),
    # dict(
    #     type='MmWaveRangeShift',
    #     range_shift=(0.06, 0.36),
    #     apply_prob=0.5,
    #     point_cloud_range=point_cloud_range,
    #     clip_to_range=True,
    #     shared_across_frames=True
    # ),
    # dict(
    #     type='MmWaveYawRotation',
    #     yaw_deg=(0.0, 8.0),
    #     angles_in_degrees=False,
    #     apply_prob=0.5,
    #     point_cloud_range=point_cloud_range,
    #     clip_to_original=True,
    #     shared_across_frames=True
    # ),
    dict(
        type='UniformSampling',
        sampling_ratio = 0.75,  
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
        normalize_phase=True,
        backend_args=backend_args
    ),
    dict(
        type='TemporalPhaseFiltering',
        amp_threshold=0.5,
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


train_dataloader = dict(
    batch_size=16,      # Reduced due to increased model size
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
    batch_size=16,
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


val_evaluator = [
    dict(
        type='HeatmapMPJPE',
        decode_from_heatmap=True,  # Use soft-argmax decoding like training
        softargmax_temperature=3.0,  # Sharper than standard (1.0), but smoother than hard argmax
        point_cloud_range=point_cloud_range,
        analysis_out_dir='work_dirs/heatmap_mpjpe_backbone',
        low_mpjpe_thr=0.06,
        high_mpjpe_thr=0.10,
    ),
    dict(
        type='PCKMetric',
        thresholds=(0.12, 0.15),
    ),
    # dict(
    #     type='MatchedPredictionDump', 
    #     out_dir='work_dirs/0219_predictions_wo_ik/',
    #     use_ik_output=False,      # False=heatmap\u8f93\u51fa, True=IK\u8f93\u51fa
    #     include_gt=True,          # \u662f\u5426\u5305\u542bGT
    #     include_errors=False       # \u662f\u5426\u5305\u542bper-joint\u8bef\u5dee
    # ),
]
test_evaluator = val_evaluator


train_cfg = dict(
    type='EpochBasedTrainLoop',
    # max_iters=max_iters,
    # val_interval=iters_per_epoch,
    # val_begin=iters_per_epoch
    max_epochs=max_epochs,
    val_interval=5,
    val_begin=5
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ============================================================================
# Optimizer & Scheduler
# ============================================================================

optim_wrapper = dict(
    type='OptimWrapper',
    # loss_scale='dynamic',
    # dtype='bfloat16',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-3),
    clip_grad=dict(max_norm=10.0, norm_type=2),
    accumulative_counts=4  # Increased gradient accumulation
)

# Learning rate scheduler
param_scheduler = [
    # dict(
    #     type='LinearLR',
    #     start_factor=0.1,
    #     begin=0,
    #     end=200,  # Longer warmup for complex model
    #     by_epoch=False
    # ),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=max_iters,
        end=max_iters,
        by_epoch=False,
        eta_min=1e-7
    )
]


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=100,
        log_metric_by_epoch=False
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        by_epoch=True,
        max_keep_ckpts=5,
        save_best='heatmap/MPJPE',
        rule='less',
        save_last=True
    ),
    sampler_seed=dict(type='DistSamplerSeedHook')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl', timeout=1800)
)

log_processor = dict(by_epoch=False, window_size=100)
log_level = 'INFO'
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Visualizer',
    vis_backends=vis_backends,
    name='visualizer'
)


auto_scale_lr = dict(enable=False, base_batch_size=8)
load_from = None
resume = False
find_unused_parameters = True
