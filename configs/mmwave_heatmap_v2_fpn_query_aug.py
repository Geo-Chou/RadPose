"""
Advanced Multi-Person 2.5D Heatmap Configuration with:
1. CNN Decoder Head (eliminating information bottleneck)
2. FPN + Frame Query Transformer (multi-scale + temporal fusion)
3. Frame-level temporal attention (not position-wise)
4. Center frame query for target prediction

Key Improvements:
- CNN decoder preserves spatial structure
- Multi-scale features via FPN
- Frame query explicitly targets center frame
- Full cross-frame temporal modeling
"""

import math

default_scope = 'mmwave3d'
custom_imports = dict(imports=['mmwave3d'], allow_failed_imports=False)
work_dir = './work_dirs/mmwave_heatmap_fpn_single_final'

backend_args = None
dataset_type = 'MMWaveDataset'
data_root = 'data/'
num_frames = 5
max_epochs = 50
iters_per_epoch = 2395
max_iters = max_epochs * iters_per_epoch
point_cloud_range = [0.12, (30/180)*math.pi, -(30/180)*math.pi, 6.0, (150/180)*math.pi, (30/180)*math.pi]
angle_voxel_size = math.radians(1.0)
data_norm = True
world_size = 4

# Heatmap parameters - Higher resolution for better accuracy
range_bins = 150  # Increased from 60: 0.049m per bin (5cm resolution)
angle_bins = 180  # Increased from 90: 0.67° per bin
sigma_r = 0.08   # Adjusted for higher resolution
sigma_a = math.radians(1.)  # Adjusted for higher resolution

randomness = dict(
    seed=3407,
    diff_rank_seed=False
)

model = dict(
    type='MmKeypoints',
    data_preprocessor=dict(
        type='SparseVoxelization',  # Sparse format for efficient SpConv
        point_cloud_range=point_cloud_range,
        data_norm=data_norm,
        voxel_size_r=0.06,
        voxel_angle_size_deg=math.radians(0.5),
        expected_frames=num_frames,
        angles_in_degrees=False,
        max_tokens_per_sample=20480,  # Reduced from 20000: 4096×5frames=20480 total voxels (manageable for SpConv) #8192
        non_blocking=True,
    ),
    transformer=dict(
        type='FPNFrameQueryTransformer',
        input_token_dim=3,
        pos_dim=4,
        embed_dim=512,          # Increased from 128
        num_layers=6,           # Increased from 4
        num_heads=4,            # Increased from 4
        ffn_dim=512,          # Increased from 512
        dropout=0.1,
        # FPN configuration
        fpn_channels=(128, 256, 512),  # Multi-scale channels
        spconv_channels=(64, 64),
        spconv_kernel_size=3,
        # Temporal configuration
        temporal_layers=2,      # More temporal layers
        num_query_tokens=576,  #576
        # Downsampling
        downsample_tokens=1536,
        downsample_method='attention',
        non_center_downsample_tokens=384,
        norm_first=True,
        pos_freq_bands=(1.0, 2.0, 4.0, 8.0, 16.0)  # More frequency bands
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
        amp_threshold=0.05,
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
        amp_threshold=0.05,
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
    batch_size=6,      # Reduced due to increased model size
    num_workers=6,
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
    batch_size=6,
    num_workers=6,
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
        point_cloud_range=point_cloud_range
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

# Custom training strategy hooks
# custom_hooks = [
#     # Loss weight warmup
#     dict(
#         type='LossWeightWarmupHook',
#         loss_warmups=[
#         dict(
#             loss_name='loss_heatmap',
#             warmup_iters=1 * iters_per_epoch,
#             init_weight=10.0,
#             final_weight=100.0
#         ),
#         dict(
#             loss_name='coord_loss_weight',
#             warmup_iters=1 * iters_per_epoch,
#             init_weight=0.0,
#             final_weight=5.0
#         ),
#         dict(
#             loss_name='loss_elevation',
#             warmup_iters=1 * iters_per_epoch,
#             init_weight=1,
#             final_weight=1.5
#         ),
#         dict(
#             loss_name='loss_skeleton_angle',
#             warmup_iters=1 * iters_per_epoch,
#             init_weight=0.05,
#             final_weight=0.3
#         ),
#         dict(
#             loss_name='loss_skeleton_length',
#             warmup_iters=1 * iters_per_epoch,
#             init_weight=0.1,
#             final_weight=3.5
#         ),
#         ]
#     ),
    
#     # Layer-wise LR decay
#     # dict(
#     #     type='LayerDecayLRHook',
#     #     backbone_lr_mult=1,
#     #     transformer_lr_mult=1,
#     #     decay_start_iter=None,
#     #     decay_factor=1
#     # )
# ]

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
load_from = '/workspace/mmpose/work_dirs/mmwave_heatmap_fpn_single_final/epoch_19.pth'
resume = False
find_unused_parameters = True


# Right lower leg(0,1): mean=0.424867 var=0.013803
# Right Upper leg(1,2): mean=0.370201 var=0.012932
# Right hip joint link(2,6): mean=0.109646 var=0.004461
# Left lower leg(5,4): mean=0.428162 var=0.015312
# Left Upper leg(3,4): mean=0.379860 var=0.018944
# Left hip joint link(3,6): mean=0.112304 var=0.007067
# lower torso(6,7): mean=0.465852 var=0.007204
# Right forearm(10,11): mean=0.270382 var=0.101753
# Right upper arm(11,12): mean=0.284976 var=0.009755
# Right shoulder(12,7): mean=0.169907 var=0.006977
# Left forearm(14,15): mean=0.263449 var=0.100065
# Left upper arm(13,14): mean=0.282122 var=0.008362
# Left shoulder(13,7): mean=0.170101 var=0.006763
# Neck(7,8): mean=0.103890 var=0.002520
# Head(8,9): mean=0.307599 var=0.020157
