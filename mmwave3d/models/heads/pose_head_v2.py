
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from mmwave3d.registry import MODELS
from mmengine.logging import MMLogger
from mmwave3d.utils import spherical_to_cartesian


@MODELS.register_module()
class MmWavePoseHeadV2(nn.Module):
    """
    MLP head support set prediction
    
    Outputs:
    - keypoints: (B, num_person, num_joints, 3)
    - joint_scores: (B, num_person, num_joints) - per-joint confidence
    - person_scores: (B, num_person) - per-person confidence (whether a person is present)
    """

    def __init__(self,
                 in_dim: int,
                 num_person: int = 4,
                 num_joints: int = 17,
                 mlp_hidden_dim: int = 512,
                 mlp_depth: int = 2,
                 data_norm: bool = False,
                 point_cloud_range: Optional[Sequence[float]] = None,
                 loss_coord: Optional[Dict] = None,
                 loss_conf: Optional[Dict] = None,
                 use_set_prediction: bool = True,
                 use_radius_loss: bool = False,
                 radius_loss: Optional[Dict] = None) -> None:
        super().__init__()
        self.num_person = num_person
        self.num_joints = num_joints
        self.use_set_prediction = use_set_prediction
        self.use_radius_loss = use_radius_loss
        if self.use_radius_loss and num_person != 1:
            raise ValueError('KeypointRadiusOffsetLoss currently only supports num_person=1.')
        
        # Build losses
        if use_set_prediction:
            if loss_coord is None:
                loss_coord = dict(
                    type='SetPredictionLossV2',
                    coord_weight=1.0,
                    no_object_weight=0.5,
                    loss_weight=1.0)
            self.loss_fn = MODELS.build(loss_coord)
        else:
            if loss_coord is None:
                loss_coord = dict(type='MPJPELoss', loss_weight=1.0)
            self.loss_coord = MODELS.build(loss_coord)
            if loss_conf:
                self.loss_conf = MODELS.build(loss_conf)
            else:
                self.loss_conf = None
        
        self.data_norm = data_norm
        if self.data_norm:
            if point_cloud_range is None or len(point_cloud_range) != 6:
                raise ValueError('point_cloud_range must have 6 values')
            range_min = torch.tensor(point_cloud_range[:3], dtype=torch.float32)
            range_max = torch.tensor(point_cloud_range[3:], dtype=torch.float32)
            self.register_buffer('range_min', range_min, persistent=False)
            self.register_buffer('range_max', range_max, persistent=False)
            self.register_buffer('range_span', range_max - range_min, persistent=False)
        else:
            self.register_buffer('range_min', torch.zeros(3), persistent=False)
            self.register_buffer('range_max', torch.zeros(3), persistent=False)
            self.register_buffer('range_span', torch.ones(3), persistent=False)

        # Shared MLP
        mlp_layers = []
        dim = in_dim
        for _ in range(mlp_depth):
            mlp_layers += [nn.Linear(dim, mlp_hidden_dim), nn.ReLU(inplace=True)]
            dim = mlp_hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Output heads
        self.keypoint_head = nn.Linear(dim, num_person * num_joints * 3)
        
        # BUG FIX
        if self.data_norm:
            nn.init.uniform_(self.keypoint_head.weight, -0.01, 0.01)
            nn.init.zeros_(self.keypoint_head.bias)
        
        # Only create joint_score_head if not using set prediction
        # In set prediction, person_score_head is used instead
        if not use_set_prediction:
            self.joint_score_head = nn.Linear(dim, num_person * num_joints)
        
        if use_set_prediction:
            self.person_score_head = nn.Linear(dim, num_person)
            nn.init.uniform_(self.person_score_head.weight, -0.1, 0.1)
            nn.init.zeros_(self.person_score_head.bias)
        
        # Optional offset branch for radius-constrained loss (per-token offsets)
        if self.use_radius_loss:
            if radius_loss is None:
                radius_loss = dict(
                    type='KeypointRadiusOffsetLoss',
                    radius=0.2,
                    loss_weight=1.0)
            self.radius_loss = MODELS.build(radius_loss)
            self.offset_head = nn.Linear(in_dim, num_joints * 3)

    def forward(self,
                inputs: Dict[str, torch.Tensor],
                data_samples: Optional[List] = None) -> Dict[str, torch.Tensor]:
        feats = inputs['features']  # (B, N, C)
        pooled = feats.mean(dim=1)
        feat_emb = self.mlp(pooled)
        
        # Keypoints
        keypoints_raw = self.keypoint_head(feat_emb).view(
            feats.size(0), self.num_person, self.num_joints, 3)
        
        if self.data_norm:
            keypoints = keypoints_raw.sigmoid()
            keypoints = self._denormalize_coords(keypoints)
        
        outputs = dict(keypoints=keypoints)
        
        if self.use_set_prediction:
            # Set prediction mode: use person_scores only
            person_scores = self.person_score_head(feat_emb)  # (B, num_person), logits
            outputs['person_scores'] = person_scores
        else:
            # Original mode: use joint_scores
            joint_scores = self.joint_score_head(feat_emb).view(
                feats.size(0), self.num_person, self.num_joints)
            outputs['scores'] = torch.sigmoid(joint_scores)
        
        if data_samples is not None:
            gt_kpts, mask = self._stack_gt(
                data_samples, keypoints.device, keypoints.size(0))
            loss_dict = dict()
            if gt_kpts is not None:
                if self.use_set_prediction:
                    person_scores = outputs.get('person_scores')
                    loss_dict = self.loss(keypoints, person_scores, gt_kpts, mask)
                else:
                    scores = outputs.get('scores')
                    loss_dict = self.loss(keypoints, scores, gt_kpts, mask)
            else:
                # No GT: keep graph connected to parameters to satisfy DDP.
                zero_tap = keypoints.sum()
                if self.use_set_prediction:
                    zero_tap = zero_tap + outputs.get('person_scores', torch.tensor(0., device=keypoints.device)).sum()
                else:
                    zero_tap = zero_tap + outputs.get('scores', torch.tensor(0., device=keypoints.device)).sum()
                loss_dict = dict(loss_dummy=zero_tap * 0.0)
            
            # Optional keypoint-centered radius loss on per-token offsets
            if self.use_radius_loss and hasattr(self, 'offset_head'):
                offsets = self.offset_head(feats).view(
                    feats.size(0), feats.size(1), self.num_joints, 3)
                if 'positions' in inputs:
                    pos = inputs['positions']  # (B, N, 4) with (r,a,e,t)
                    points_sph = pos[..., :3]
                    points_xyz = spherical_to_cartesian(points_sph)
                    gt_single = gt_kpts[:, 0] if (gt_kpts is not None and gt_kpts.dim() == 4) else (
                        gt_kpts if gt_kpts is not None else None)
                    if gt_single is not None:
                        gt_cart = spherical_to_cartesian(gt_single)
                        loss_radius = self.radius_loss(points_xyz, offsets, gt_cart)
                    else:
                        loss_radius = offsets.sum() * 0.0
                else:
                    loss_radius = offsets.sum() * 0.0
                loss_dict['loss_kp_radius'] = loss_radius
            if loss_dict:
                outputs.update(loss_dict)
        
        return outputs

    def _stack_gt(self, data_samples: List, device: torch.device,
                  batch_size: int
                  ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not data_samples or batch_size == 0:
            return None, None
        sample_names = []
        bsz = batch_size
        gt_batch = torch.zeros(
            (bsz, self.num_person, self.num_joints, 3), device=device)
        mask_batch = torch.zeros(
            (bsz, self.num_person, self.num_joints), dtype=torch.bool, device=device)
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
            logger = MMLogger.get_current_instance()
            if logger is not None:
                logger.warning(
                    '[MmWavePoseHeadV2] Empty GT batch. samples=%s processed=%d',
                    sample_names, processed)
            return None, None
        return gt_batch, mask_batch

    def _denormalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        span = self.range_span.to(device=coords.device, dtype=coords.dtype)
        minv = self.range_min.to(device=coords.device, dtype=coords.dtype)
        return coords * span + minv

    def loss(self, pred_kpts: torch.Tensor, 
             scores: torch.Tensor,
             gt_kpts: torch.Tensor, 
             mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        if self.use_set_prediction:
            # set prediction loss: scores are person_scores (logits)
            assert scores is not None
            losses = self.loss_fn(pred_kpts, scores, gt_kpts, mask)
            return losses
        else:
            # original loss: scores are joint_scores (probabilities)
            losses = dict()
            losses['loss_mpjpe'] = self.loss_coord(pred_kpts, gt_kpts, mask)
            if self.loss_conf is not None:
                losses['loss_human_count'] = self.loss_conf(scores, mask.float())
            return losses
