import math

default_scope = "mmwave3d"
custom_imports = dict(imports=["mmwave3d"], allow_failed_imports=False)
work_dir = "./work_dirs/radpose"

backend_args = None
dataset_type = "MMWaveDataset"
data_root = "data/seen/"
num_frames = 5

num_gpus = 4
batch_size = 26
effective_batch_size = batch_size * num_gpus
num_train_samples = 120174
iters_per_epoch = -(-num_train_samples // effective_batch_size)
max_epochs = 2
max_iters = max_epochs * iters_per_epoch

point_cloud_range = [
    0.12,
    (30 / 180) * math.pi,
    -(30 / 180) * math.pi,
    6.0,
    (150 / 180) * math.pi,
    (30 / 180) * math.pi,
]
angle_voxel_size = math.radians(1.0)
data_norm = True
world_size = num_gpus

range_bins = 150
angle_bins = 180
sigma_r = 0.09
sigma_a = math.radians(1.0)

model = dict(
    type="MmKeypoints",
    data_preprocessor=dict(
        type="SparseVoxelization",
        point_cloud_range=point_cloud_range,
        data_norm=data_norm,
        voxel_size_r=0.06,
        voxel_angle_size_deg=math.radians(0.5),
        expected_frames=num_frames,
        angles_in_degrees=False,
        max_tokens_per_sample=7680,
        non_blocking=True,
        vectorized_batch=True,
        overflow_summary_tokens=512,
        overflow_coarsen_factor=4,
    ),
    transformer=dict(
        type="PhysicsAwareFPNFrameQueryTransformer",
        input_token_dim=3,
        pos_dim=4,
        embed_dim=512,
        num_layers=6,
        num_heads=4,
        ffn_dim=512,
        dropout=0.05,
        fpn_channels=(128, 256, 512),
        spconv_channels=(64, 64),
        spconv_kernel_size=3,
        temporal_layers=2,
        num_query_tokens=576,
        downsample_tokens=2048,
        downsample_method="attention",
        non_center_downsample_tokens=512,
        norm_first=True,
        pos_freq_bands=(1.0, 2.0, 4.0, 8.0, 16.0),
        use_isotropic_branch=True,
    ),
    head=dict(
        type="HeatmapIKHead",
        heatmap_head=dict(
            type="CNNHeatmapHead",
            in_dim=512,
            num_person=1,
            num_joints=16,
            num_frames=num_frames,
            use_matching=False,
            initial_spatial_size=24,
            decoder_channels=(512, 256, 128),
            range_bins=range_bins,
            angle_bins=angle_bins,
            point_cloud_range=point_cloud_range,
            sigma_r=sigma_r,
            sigma_a=sigma_a,
            predict_uncertainty=True,
            log_sigma_min=-7.0,
            log_sigma_max=7.0,
            loss_heatmap=dict(
                type="Heatmap2DLoss",
                alpha=1.0,
                beta=2.0,
                reduction="mean",
                loss_weight=100.0,
            ),
            loss_elevation=dict(
                type="ElevationLoss", threshold=0.15, reduction="mean", loss_weight=1.5
            ),
            loss_skeleton_angle=dict(
                type="SkeletonAngleLoss",
                reduction="mean",
                loss_weight=0.3,
                spherical_input=True,
                use_arccos=True,
            ),
            loss_skeleton_length=dict(
                type="SkeletonLengthLoss",
                reduction="mean",
                loss_weight=3.5,
                spherical_input=True,
            ),
            coord_loss_weight=5.0,
            coord_loss_beta=0.1,
            decode_inference_with_argmax=True,
            use_elevation_heatmap=True,
            elev_bins=90,
            sigma_el=math.radians(1.0),
            re_heatmap_weight=1.0,
            range_consistency_weight=0.1,
        ),
        ik_head=dict(
            type="IKHead",
            in_dim=512,
            num_person=1,
            num_joints=16,
            root_index=6,
            skeleton_edges=[
                (0, 1),
                (1, 2),
                (2, 6),
                (5, 4),
                (4, 3),
                (3, 6),
                (7, 6),
                (10, 11),
                (11, 12),
                (12, 7),
                (15, 14),
                (14, 13),
                (13, 7),
                (8, 7),
                (9, 8),
            ],
            bone_length_means=[
                (0, 1, 0.402669),
                (1, 2, 0.376000),
                (2, 6, 0.096862),
                (5, 4, 0.404981),
                (4, 3, 0.380240),
                (3, 6, 0.098863),
                (7, 6, 0.455995),
                (10, 11, 0.233903),
                (11, 12, 0.264479),
                (12, 7, 0.158192),
                (15, 14, 0.237198),
                (14, 13, 0.270383),
                (13, 7, 0.157439),
                (8, 7, 0.101654),
                (9, 8, 0.304740),
            ],
            gcn_hidden=512,
            gcn_layers=3,
            joint_attn_num_heads=4,
            joint_attn_dropout=0.1,
            loss_ik_coord=dict(type="MPJPELoss", reduction="mean", loss_weight=3.0),
            loss_ik_bias_weight=0.0,
            loss_ik_bias_beta=0.1,
            loss_ik_angle_weight=0.2,
            loss_ik_angle_type="geodesic",
            loss_ik_angle_beta=0.1,
            loss_fk_consistency_weight=0.3,
            loss_fk_consistency_beta=0.1,
            loss_graph_reg_weight=0.01,
            loss_bone_length_weight=0.5,
            bone_length_range=0.15,
            use_confidence=True,
            use_confidence_weighting=False,
            point_cloud_range=point_cloud_range,
            softargmax_temperature=3.0,
            use_depth_weighting=True,
            depth_weight_scale=0.5,
            fk_residual_weight=0.5,
            fk_residual_beta=0.1,
            loss_fk_residual_reg_weight=0.05,
            loss_root_pos_weight=0.5,
            loss_root_residual_reg_weight=0.05,
            root_residual_weight=0.5,
        ),
    ),
)


train_pipeline = [
    dict(
        type="LoadMmPointsFromFile",
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        use_amplitude=True,
        use_phase=True,
        normalize_amplitude=True,
        normalize_phase=True,
        backend_args=backend_args,
    ),
    dict(
        type="TemporalPhaseFiltering",
        amp_threshold=0.06,
        phase_delta=1.5,
        phase_voxel=1.5,
        amp_relax=0.01,
        static_delta=0.5,
        static_persist=1,
        motion_decay=0.1,
        motion_keep=0.1,
        history_size=3,
        use_phase=False,
    ),
    dict(type="UniformSampling", sampling_ratio=0.75, sampling_method="random"),
    dict(type="LoadAnnotations3D", with_keypoints_3d=True, eval=False),
    dict(keys=("points", "gt_keypoints_3d"), type="Pack3DInputs"),
]

test_pipeline = [
    dict(
        type="LoadMmPointsFromFile",
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        use_amplitude=True,
        use_phase=True,
        normalize_amplitude=True,
        normalize_phase=True,
        backend_args=backend_args,
    ),
    dict(
        type="TemporalPhaseFiltering",
        amp_threshold=0.06,
        phase_delta=1.5,
        phase_voxel=1.5,
        amp_relax=0.01,
        static_delta=0.5,
        static_persist=1,
        motion_decay=0.1,
        motion_keep=0.1,
        history_size=3,
        use_phase=False,
    ),
    dict(type="LoadAnnotations3D", with_keypoints_3d=True, eval=True),
    dict(keys=("points", "gt_keypoints_3d"), type="Pack3DInputs"),
]


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="train/annotations",
        pipeline=train_pipeline,
        num_frames=num_frames,
        frame_interval=1,
        metainfo_file="metainfo.json",
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="val/annotations",
        pipeline=test_pipeline,
        num_frames=num_frames,
        test_mode=True,
        frame_interval=1,
        metainfo_file="metainfo.json",
        load_eval_anns=True,
        backend_args=backend_args,
    ),
)

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root="data/unseen/",
        ann_file="test/annotations",
        pipeline=test_pipeline,
        num_frames=num_frames,
        test_mode=True,
        split="test",
        frame_interval=1,
        metainfo_file="metainfo.json",
        load_eval_anns=True,
        backend_args=backend_args,
    ),
)


val_evaluator = [
    dict(
        type="HeatmapMPJPE",
        decode_from_heatmap=True,
        softargmax_temperature=3.0,
        point_cloud_range=point_cloud_range,
    ),
    dict(
        type="IKMPJPE",
        point_cloud_range=point_cloud_range,
        low_mpjpe_thr=0.10,
        high_mpjpe_thr=0.20,
    ),
    dict(
        type="IKPCKMetric",
        thresholds=(0.10,),
    ),
]
test_evaluator = val_evaluator


train_cfg = dict(
    type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1, val_begin=1
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")


optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-3),
    paramwise_cfg=dict(
        custom_keys={
            "transformer": dict(lr_mult=0.1),
            "head.heatmap_head": dict(lr_mult=0.1),
            "head.ik_head": dict(lr_mult=1.0),
        }
    ),
    clip_grad=dict(max_norm=10.0, norm_type=2),
    accumulative_counts=4,
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, begin=0, end=200, by_epoch=False),
    dict(
        type="CosineAnnealingLR",
        begin=0,
        T_max=max_iters,
        end=max_iters,
        by_epoch=False,
        eta_min=1e-7,
    ),
]


default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=100, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        by_epoch=True,
        max_keep_ckpts=5,
        save_best="ik/MPJPE",
        rule="less",
        save_last=True,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)


custom_hooks = [
    dict(
        type="EarlyStoppingHook",
        monitor="ik/MPJPE",
        rule="less",
        min_delta=0.001,
        patience=4,
    ),
]

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl", timeout=18000),
)

log_processor = dict(by_epoch=False, window_size=100)
log_level = "INFO"
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="Visualizer", vis_backends=vis_backends, name="visualizer")


auto_scale_lr = dict(enable=False, base_batch_size=192)


load_from = None
resume = False
find_unused_parameters = True
