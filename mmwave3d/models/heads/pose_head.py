from typing import Dict, List, Optional, Sequence, Tuple
import time

import torch
import torch.nn as nn

from mmwave3d.registry import MODELS
from mmengine.logging import MMLogger
from mmwave3d.utils import spherical_to_cartesian

@MODELS.register_module()
class MmWavePoseHead(nn.Module):
    """MLP head predicting 4-person 3D keypoints (17 per person) & scores."""

    def __init__(self,
                 in_dim: int,
                 num_person: int = 4,
                 num_joints: int = 17,
                 mlp_hidden_dim: int = 512,
                 mlp_depth: int = 2,
                 data_norm: bool = False,
                 point_cloud_range: Optional[Sequence[float]] = None,
                 loss_mpjpe: Optional[Dict] = dict(type='MPJPELoss', loss_weight=1.0),
                 loss_human_count: Optional[Dict] = dict(
                     type='HumanCountLoss', threshold=0.5, min_joints=9,
                     loss_weight=1.0)) -> None:
        super().__init__()
        self.num_person = num_person
        self.num_joints = num_joints
        self.loss_mpjpe = MODELS.build(loss_mpjpe) if loss_mpjpe else None
        self.loss_human_count = (
            MODELS.build(loss_human_count) if loss_human_count else None)
        self.data_norm = data_norm
        if self.data_norm:
            if point_cloud_range is None or len(point_cloud_range) != 6:
                raise ValueError('point_cloud_range must have 6 values when data_norm=True')
            range_min = torch.tensor(point_cloud_range[:3], dtype=torch.float32)
            range_max = torch.tensor(point_cloud_range[3:], dtype=torch.float32)
            self.register_buffer('range_min', range_min, persistent=False)
            self.register_buffer('range_max', range_max, persistent=False)
            self.register_buffer('range_span', range_max - range_min, persistent=False)
        else:
            self.register_buffer('range_min', torch.zeros(3), persistent=False)
            self.register_buffer('range_max', torch.zeros(3), persistent=False)
            self.register_buffer('range_span', torch.ones(3), persistent=False)

        mlp_layers = []
        dim = in_dim
        for _ in range(mlp_depth):
            mlp_layers += [nn.Linear(dim, mlp_hidden_dim), nn.ReLU(inplace=True)]
            dim = mlp_hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        self.keypoint_head = nn.Linear(dim, num_person * num_joints * 3)
        self.score_head = nn.Linear(dim, num_person * num_joints)

    def forward(self,
                inputs: Dict[str, torch.Tensor],
                data_samples: Optional[List] = None) -> Dict[str, torch.Tensor]:
        # torch.cuda.synchronize()
        # start_time = time.time()
        feats = inputs['features']  # (B, N, C) from transformer
        pooled = feats.mean(dim=1)
        feat_emb = self.mlp(pooled)
        keypoints = self.keypoint_head(feat_emb).view(
            feats.size(0), self.num_person, self.num_joints, 3)
        if self.data_norm:
            keypoints = keypoints.sigmoid()
            keypoints = self._denormalize_coords(keypoints)
        scores = torch.sigmoid(self.score_head(feat_emb).view(
            feats.size(0), self.num_person, self.num_joints))
        outputs = dict(keypoints=keypoints, scores=scores)
        # torch.cuda.synchronize()
        # head_time = time.time()-start_time
        # logger = MMLogger.get_current_instance()
        # if logger is not None:
        #             logger.info(
        #                 'head timings - head: %.4fs',
        #                 head_time)

        if data_samples is not None:
            # torch.cuda.synchronize()
            # start_time = time.time()
            gt_kpts, mask = self._stack_gt(
                data_samples, keypoints.device, keypoints.size(0))
            # torch.cuda.synchronize()
            # if logger is not None:
            #         logger.info(
            #             'head gt timings - gt: %.4fs',
            #             time.time()-start_time)
            # torch.cuda.synchronize()
            # start_time= time.time()
            if gt_kpts is not None:
                loss_dict = self.loss(keypoints, scores, gt_kpts, mask)
                outputs.update(loss_dict)
                # torch.cuda.synchronize()
                # if logger is not None:
                #     logger.info(
                #         'head loss timings - loss: %.4fs',
                #         time.time()-start_time)
        return outputs

    def _stack_gt(self, data_samples: List, device: torch.device,
                  batch_size: int
                  ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not data_samples or batch_size == 0:
            return None, None
        bsz = batch_size
        gt_batch = torch.zeros(
            (bsz, self.num_person, self.num_joints, 3), device=device)
        mask_batch = torch.zeros(
            (bsz, self.num_person, self.num_joints), dtype=torch.bool, device=device)
        processed = 0
        for idx, sample in enumerate(data_samples):
            if idx >= batch_size:
                break
            if not hasattr(sample, 'gt_keypoints_3d'):
                continue
            gt = sample.gt_keypoints_3d
            if gt is None:
                continue
            gt = gt.to(device)
            if gt.dim() == 2:
                gt = gt.unsqueeze(0)
            if gt.shape[-1] > 3:
                coords = gt[..., :3]
                extra = gt[..., 3:]
            else:
                coords = gt
                extra = None
            persons = min(coords.shape[0], self.num_person)
            if persons == 0:
                continue
            gt_batch[idx, :persons] = coords[:persons]
            mask = coords[:persons].abs().sum(dim=-1) > 0
            if extra is not None:
                if extra.shape[-1] == 1:
                    extra_mask = extra[:persons, :, 0] > 0
                else:
                    extra_mask = extra[:persons].abs().sum(dim=-1) > 0
                mask = mask | extra_mask
            mask_batch[idx, :persons] = mask
            processed += 1
        if mask_batch.sum() == 0 or processed == 0:
            return None, None
        return gt_batch, mask_batch

    def _denormalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        span = self.range_span.to(device=coords.device, dtype=coords.dtype)
        minv = self.range_min.to(device=coords.device, dtype=coords.dtype)
        return coords * span + minv

    def loss(self, pred_kpts: torch.Tensor, pred_scores: torch.Tensor,
             gt_kpts: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = dict()
        # torch.cuda.synchronize()
        # start_time = time.time()
        if self.loss_mpjpe is not None:
            losses['loss_mpjpe'] = self.loss_mpjpe(pred_kpts, gt_kpts, mask)
            # torch.cuda.synchronize()
            # mpjpe_t = time.time()
            # mpjpe_time = mpjpe_t-start_time
        if self.loss_human_count is not None:
            losses['loss_human_count'] = self.loss_human_count(
                pred_scores, mask.float())
        #     torch.cuda.synchronize()
        #     human_t = time.time()
        #     human_time = human_t-mpjpe_t
        # logger = MMLogger.get_current_instance()
        # if logger is not None:
        #             logger.info(
        #                 'loss timings - mpjpe: %.4fs.  human: %.4fs',
        #                 mpjpe_time, human_time)
        return losses


@MODELS.register_module()
class MmWaveSinglePoseHead(nn.Module):
    """Compact head for single-person 3D keypoint regression."""

    def __init__(self,
                 in_dim: int,
                 num_person: int = 1,
                 num_joints: int = 17,
                 mlp_hidden_dim: int = 512,
                 mlp_depth: int = 2,
                 data_norm: bool = False,
                 point_cloud_range: Optional[Sequence[float]] = None,
                 loss_mpjpe: Optional[Dict] = None,
                #  dict(type='CartesianL2Loss', loss_weight=1.0),
                 use_radius_loss: bool = False,
                 radius_loss: Optional[Dict] = None) -> None:
        super().__init__()
        if num_person != 1:
            raise ValueError('MmWaveSinglePoseHead only supports num_person=1')
        self.num_joints = num_joints
        self.loss_mpjpe = MODELS.build(loss_mpjpe) if loss_mpjpe else None
        self.use_radius_loss = use_radius_loss
        if self.use_radius_loss:
            if radius_loss is None:
                radius_loss = dict(
                    type='KeypointRadiusOffsetLoss',
                    radius=0.2,
                    loss_weight=1.0)
            self.radius_loss = MODELS.build(radius_loss)
        self.data_norm = data_norm
        if self.data_norm:
            if point_cloud_range is None or len(point_cloud_range) != 6:
                raise ValueError('point_cloud_range must have 6 values when data_norm=True')
            range_min = torch.tensor(point_cloud_range[:3], dtype=torch.float32)
            range_max = torch.tensor(point_cloud_range[3:], dtype=torch.float32)
            self.register_buffer('range_min', range_min, persistent=False)
            self.register_buffer('range_max', range_max, persistent=False)
            self.register_buffer('range_span', range_max - range_min, persistent=False)
        else:
            self.register_buffer('range_min', torch.zeros(3), persistent=False)
            self.register_buffer('range_max', torch.zeros(3), persistent=False)
            self.register_buffer('range_span', torch.ones(3), persistent=False)

        mlp_layers = []
        dim = in_dim
        for _ in range(mlp_depth):
            mlp_layers += [nn.Linear(dim, mlp_hidden_dim), nn.ReLU(inplace=True)]
            dim = mlp_hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        self.keypoint_head = nn.Linear(dim, num_joints * 3)
        if self.use_radius_loss:
            # Predict per-token offsets to joints for radius-based supervision.
            self.offset_head = nn.Linear(in_dim, num_joints * 3)

    def forward(self,
                inputs: Dict[str, torch.Tensor],
                data_samples: Optional[List] = None) -> Dict[str, torch.Tensor]:
        feats = inputs['features']  # (B, N, C) from transformer
        pooled = feats.mean(dim=1)
        feat_emb = self.mlp(pooled)
        keypoints = self.keypoint_head(feat_emb).view(
            feats.size(0), self.num_joints, 3)
        if self.data_norm:
            keypoints = keypoints.sigmoid()
            keypoints = self._denormalize_coords(keypoints)
        outputs = dict(keypoints=keypoints)
        
        # Compute per-token offsets if radius loss is enabled
        # This must always run to avoid unused parameters error
        if self.use_radius_loss and hasattr(self, 'offset_head'):
            offsets = self.offset_head(feats).view(
                feats.size(0), feats.size(1), self.num_joints, 3)
            outputs['offsets'] = offsets  # Store for later use

        if data_samples is not None:
            gt_kpts, mask = self._stack_gt(
                data_samples, keypoints.device, keypoints.size(0))
            
            if gt_kpts is not None:
                loss_dict = {}
                
                # Standard MPJPE loss
                if self.loss_mpjpe is not None:
                    loss_dict['loss_mpjpe'] = self.loss_mpjpe(keypoints, gt_kpts, mask)
                
                # Radius-based loss
                if self.use_radius_loss and 'offsets' in outputs:
                    if 'positions' not in inputs:
                        raise ValueError('positions required in inputs for radius loss')
                    
                    # positions: (B, N, 4) with (r_norm, theta_norm, phi_norm, frame_id)
                    # data_norm=True，[0,1]，
                    pos = inputs['positions']  # (B, N, 4)
                    points_sph_norm = pos[..., :3]  # (B, N, 3) - 
                    
                    if self.data_norm:
                        # ：[0,1]
                        span = self.range_span.to(device=points_sph_norm.device, dtype=points_sph_norm.dtype)
                        minv = self.range_min.to(device=points_sph_norm.device, dtype=points_sph_norm.dtype)
                        points_sph = points_sph_norm * span + minv  # (B, N, 3) - 
                    else:
                        points_sph = points_sph_norm
                    
                    # 
                    points_xyz = spherical_to_cartesian(points_sph)  # (B, N, 3)
                    
                    # GT joints (r, az, el)，
                    # GT3(r, az, el)
                    gt_xyz = spherical_to_cartesian(gt_kpts[..., :3])  # (B, J, 3)
                    
                    # Compute radius loss using pre-computed offsets
                    loss_radius = self.radius_loss(points_xyz, outputs['offsets'], gt_xyz, mask)
                    loss_dict['loss_kp_radius'] = loss_radius
                
                outputs.update(loss_dict)
            else:
                raise ValueError('MmWaveSinglePoseHead requires GT keypoints')
        
        return outputs

    def _stack_gt(self, data_samples: List, device: torch.device,
                  batch_size: int
                  ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not data_samples or batch_size == 0:
            return None, None
        sample_names = []
        bsz = batch_size
        gt_batch = torch.zeros(
            (bsz, self.num_joints, 3), device=device)
        mask_batch = torch.zeros(
            (bsz, self.num_joints), dtype=torch.bool, device=device)
        processed = 0
        for idx, sample in enumerate(data_samples):
            if idx >= batch_size:
                break
            if hasattr(sample, 'sample_name'):
                sample_names.append(sample.sample_name)
            if not hasattr(sample, 'gt_keypoints_3d'):
                continue
            gt = sample.gt_keypoints_3d
            if gt is None:
                continue
            gt = gt.to(device)
            if gt.shape[-1] > 3:
                coords = gt[..., :3]
                extra = gt[..., 3:]
            else:
                coords = gt
                extra = None
            if coords.dim() == 3:
                coords = coords[0]
                if extra is not None:
                    extra = extra[0]
            elif coords.dim() != 2:
                continue
            if coords.numel() == 0:
                continue
            if coords.shape[-1] > 3:
                coords = coords[..., :3]
            if extra is not None and extra.dim() == 3:
                extra = extra[0]
            mask = coords.abs().sum(dim=-1) > 0
            if extra is not None:
                if extra.shape == coords.shape:
                    extra_mask = extra.abs().sum(dim=-1) > 0
                    mask = mask | extra_mask
                elif extra.dim() == 2 and extra.shape[-1] == 1:
                    mask = mask | (extra[:, 0] > 0)
                elif extra.dim() == 2 and extra.shape[-1] > 1:
                    mask = mask | (extra.abs().sum(dim=-1) > 0)
            num_valid = min(coords.shape[0], self.num_joints)
            gt_batch[idx, :num_valid] = coords[:num_valid]
            mask_batch[idx, :num_valid] = mask[:num_valid]
            processed += 1
        if mask_batch.sum() == 0 or processed == 0:
            logger = MMLogger.get_current_instance()
            if logger is not None:
                logger.warning(
                    '[MmWaveSinglePoseHead] Empty GT batch. samples=%s processed=%d',
                    sample_names, processed)
            return None, None
        return gt_batch, mask_batch

    def _denormalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        span = self.range_span.to(device=coords.device, dtype=coords.dtype)
        minv = self.range_min.to(device=coords.device, dtype=coords.dtype)
        return coords * span + minv

    def loss(self, pred_kpts: torch.Tensor,
             gt_kpts: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = dict()
        # torch.cuda.synchronize()
        # start_time = time.time()
        if self.loss_mpjpe is not None:
            losses['loss_mpjpe'] = self.loss_mpjpe(pred_kpts, gt_kpts, mask)
            # torch.cuda.synchronize()
            # mpjpe_t = time.time()
            # mpjpe_time = mpjpe_t-start_time
        #     torch.cuda.synchronize()
        #     human_t = time.time()
        #     human_time = human_t-mpjpe_t
        # logger = MMLogger.get_current_instance()
        # if logger is not None:
        #             logger.info(
        #                 'loss timings - mpjpe: %.4fs.  human: %.4fs',
        #                 mpjpe_time, human_time)
        return losses
