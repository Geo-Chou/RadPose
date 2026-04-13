from itertools import permutations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmwave3d.registry import MODELS
from mmwave3d.utils import spherical_to_cartesian


@MODELS.register_module()
class MPJPELoss(nn.Module):
    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        assert reduction in ['mean', 'sum']
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Replace NaNs/Infs before scoring, otherwise torch.norm will return NaN
        # and poison the full batch.
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        pred = spherical_to_cartesian(pred)
        target = spherical_to_cartesian(target)
        mask = mask.to(dtype=torch.bool)
        if weights is not None:
            weights = weights.to(pred.device)
            if weights.shape[:3] != pred.shape[:3]:
                raise RuntimeError(
                    f'weights shape {weights.shape} does not match pred {pred.shape}'
                )
        total_loss = pred.new_zeros(())
        valid_samples = 0
        for b in range(pred.size(0)):
            weight_b = weights[b] if weights is not None else None
            sample_loss = self._match_sample(pred[b], target[b], mask[b], weight_b)
            if sample_loss is None:
                continue
            total_loss = total_loss + sample_loss
            valid_samples += 1
        if self.reduction == 'mean':
            total_loss = total_loss / max(valid_samples, 1)
        return total_loss * self.loss_weight

    def _match_sample(self, pred_person: torch.Tensor,
                      gt_person: torch.Tensor,
                      mask_person: torch.Tensor,
                      weights_person: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        gt_indices = [i for i in range(gt_person.size(0))
                      if mask_person[i].any()]
        if len(gt_indices) == 0:
            return None
        num_pred = pred_person.size(0)
        gt_count = len(gt_indices)
        cost = pred_person.new_zeros((num_pred, gt_count))
        for pi in range(num_pred):
            for gi, gidx in enumerate(gt_indices):
                joint_mask = mask_person[gidx]
                if joint_mask.any():
                    diff = pred_person[pi, joint_mask] - gt_person[gidx, joint_mask]
                    if weights_person is not None:
                        w = weights_person[pi, joint_mask].to(diff.dtype)
                        denom = w.sum().clamp_min(1e-6)
                        cost[pi, gi] = (diff.norm(dim=-1) * w).sum() / denom
                    else:
                        cost[pi, gi] = diff.norm(dim=-1).mean()
                else:
                    cost[pi, gi] = pred_person.new_tensor(1e6)
        pred_ids = list(range(num_pred))
        best_perm = None
        best_val = None
        for perm in permutations(pred_ids, gt_count):
            total = sum(cost[perm[k], k] for k in range(gt_count))
            total_val = float(total.item())
            if best_val is None or total_val < best_val:
                best_val = total_val
                best_perm = perm
        sample_loss = pred_person.new_zeros(())
        for gi, pi in enumerate(best_perm):
            gt_idx = gt_indices[gi]
            joint_mask = mask_person[gt_idx]
            diff = pred_person[pi, joint_mask] - gt_person[gt_idx, joint_mask]
            if weights_person is not None:
                w = weights_person[pi, joint_mask].to(diff.dtype)
                denom = w.sum().clamp_min(1e-6)
                sample_loss = sample_loss + (diff.norm(dim=-1) * w).sum() / denom
            else:
                sample_loss = sample_loss + diff.norm(dim=-1).mean()
        return sample_loss / gt_count


@MODELS.register_module()
class HeteroMPJPELoss(nn.Module):
    """Heteroscedastic MPJPE loss with per-joint log-sigma."""

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 log_sigma_min: float = -7.0,
                 log_sigma_max: float = 7.0) -> None:
        super().__init__()
        assert reduction in ['mean', 'sum']
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.log_sigma_min = float(log_sigma_min)
        self.log_sigma_max = float(log_sigma_max)

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor,
                log_sigma: torch.Tensor) -> torch.Tensor:
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        pred = spherical_to_cartesian(pred)
        target = spherical_to_cartesian(target)
        mask = mask.to(dtype=torch.bool)

        if log_sigma.shape[:3] != pred.shape[:3]:
            raise RuntimeError(
                f'log_sigma shape {log_sigma.shape} does not match pred {pred.shape}'
            )
        log_sigma = log_sigma.clamp(self.log_sigma_min, self.log_sigma_max)
        total_loss = pred.new_zeros(())
        valid_samples = 0
        for b in range(pred.size(0)):
            sample_loss = self._match_sample(
                pred[b], target[b], mask[b], log_sigma[b])
            if sample_loss is None:
                continue
            total_loss = total_loss + sample_loss
            valid_samples += 1
        if self.reduction == 'mean':
            total_loss = total_loss / max(valid_samples, 1)
        return total_loss * self.loss_weight

    def _match_sample(self, pred_person: torch.Tensor,
                      gt_person: torch.Tensor,
                      mask_person: torch.Tensor,
                      log_sigma_person: torch.Tensor) -> Optional[torch.Tensor]:
        gt_indices = [i for i in range(gt_person.size(0))
                      if mask_person[i].any()]
        if len(gt_indices) == 0:
            return None
        num_pred = pred_person.size(0)
        gt_count = len(gt_indices)
        cost = pred_person.new_zeros((num_pred, gt_count))
        for pi in range(num_pred):
            for gi, gidx in enumerate(gt_indices):
                joint_mask = mask_person[gidx]
                if joint_mask.any():
                    diff = pred_person[pi, joint_mask] - gt_person[gidx, joint_mask]
                    log_sigma = log_sigma_person[pi, joint_mask].to(diff.dtype)
                    sigma2 = (log_sigma.exp() ** 2).clamp_min(1e-6)
                    err2 = (diff ** 2).sum(dim=-1)
                    loss = 0.5 * err2 / sigma2 + log_sigma
                    cost[pi, gi] = loss.mean()
                else:
                    cost[pi, gi] = pred_person.new_tensor(1e6)
        pred_ids = list(range(num_pred))
        best_perm = None
        best_val = None
        for perm in permutations(pred_ids, gt_count):
            total = sum(cost[perm[k], k] for k in range(gt_count))
            total_val = float(total.item())
            if best_val is None or total_val < best_val:
                best_val = total_val
                best_perm = perm
        sample_loss = pred_person.new_zeros(())
        for gi, pi in enumerate(best_perm):
            gt_idx = gt_indices[gi]
            joint_mask = mask_person[gt_idx]
            diff = pred_person[pi, joint_mask] - gt_person[gt_idx, joint_mask]
            log_sigma = log_sigma_person[pi, joint_mask].to(diff.dtype)
            sigma2 = (log_sigma.exp() ** 2).clamp_min(1e-6)
            err2 = (diff ** 2).sum(dim=-1)
            loss = 0.5 * err2 / sigma2 + log_sigma
            sample_loss = sample_loss + loss.mean()
        return sample_loss / gt_count


@MODELS.register_module()
class CartesianL2Loss(nn.Module):
    """Cartesian-space L2 loss for single-person joints."""

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        assert reduction in ['mean', 'sum']
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        pred = spherical_to_cartesian(pred)
        target = spherical_to_cartesian(target)
        mask = mask.to(dtype=torch.bool)
        diff = pred - target
        joint_error = diff.norm(dim=-1)
        valid = mask.to(dtype=joint_error.dtype)
        loss = joint_error * valid
        if self.reduction == 'mean':
            denom = valid.sum().clamp_min(1.0)
            loss = loss.sum() / denom
        else:
            loss = loss.sum()
        return loss * self.loss_weight


@MODELS.register_module()
class HumanCountLoss(nn.Module):
    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        assert reduction in ['mean', 'sum']
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred_scores: torch.Tensor,
                gt_mask: torch.Tensor) -> torch.Tensor:
        # Sanitize predictions to avoid NaNs/Infs that break BCE on CUDA.
        scores = torch.nan_to_num(
            pred_scores, nan=0.5, posinf=1.0, neginf=0.0)
        scores = scores.clamp_(0.0, 1.0)
        # Aggregate joint confidences into a per-person probability.
        person_prob = scores.mean(dim=-1)
        gt_person = (gt_mask.sum(dim=-1) >= 1).float()
        prob = person_prob.clamp(min=1e-6, max=1 - 1e-6)
        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy(
                prob.to(dtype=torch.float32),
                gt_person.to(dtype=torch.float32),
                reduction=self.reduction)
        return loss.to(dtype=pred_scores.dtype) * self.loss_weight


@MODELS.register_module()
class KeypointRadiusOffsetLoss(nn.Module):
    """Keypoint-centered radius loss on per-point offsets.

    Only points within a radius r of a GT joint contribute to that joint's loss.
    Inputs:
        points_xyz: (B, N, 3) Cartesian coords of points.
        pred_offsets: (B, N, J, 3) offsets from each point to each joint.
        gt_joints: (B, J, 3) GT joint positions in Cartesian.
        mask: (B, J) bool mask indicating valid joints.
    """

    def __init__(self,
                 radius: float = 0.2,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        assert reduction in ['mean', 'sum']
        self.radius = radius
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                points_xyz: torch.Tensor,
                pred_offsets: torch.Tensor,
                gt_joints: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            points_xyz: (B, N, 3) point positions in Cartesian
            pred_offsets: (B, N, J, 3) predicted offsets from points to joints
            gt_joints: (B, J, 3) ground truth joint positions in Cartesian
            mask: (B, J) bool mask for valid joints
        """
        points_xyz = torch.nan_to_num(points_xyz, nan=0.0, posinf=0.0, neginf=0.0)
        pred_offsets = torch.nan_to_num(pred_offsets, nan=0.0, posinf=0.0, neginf=0.0)
        gt_joints = torch.nan_to_num(gt_joints, nan=0.0, posinf=0.0, neginf=0.0)

        bsz, num_points, num_joints, _ = pred_offsets.shape

        loss_accum = pred_offsets.new_zeros(())
        valid_count = 0
        total_nearby_points = 0  # 
        
        # ：
        # if torch.rand(1).item() < 5:  # 5%
        #     import sys
        #     pts_sample = points_xyz[0]  # 
        #     jts_sample = gt_joints[0]
        #     print(f"[DEBUG] Points XYZ range: x=[{pts_sample[:, 0].min():.2f}, {pts_sample[:, 0].max():.2f}], "
        #           f"y=[{pts_sample[:, 1].min():.2f}, {pts_sample[:, 1].max():.2f}], "
        #           f"z=[{pts_sample[:, 2].min():.2f}, {pts_sample[:, 2].max():.2f}]",
        #           file=sys.stderr, flush=True)
        #     print(f"[DEBUG] Joints XYZ range: x=[{jts_sample[:, 0].min():.2f}, {jts_sample[:, 0].max():.2f}], "
        #           f"y=[{jts_sample[:, 1].min():.2f}, {jts_sample[:, 1].max():.2f}], "
        #           f"z=[{jts_sample[:, 2].min():.2f}, {jts_sample[:, 2].max():.2f}]",
        #           file=sys.stderr, flush=True)

        for b in range(bsz):
            pts = points_xyz[b]  # (N, 3)
            joints = gt_joints[b]  # (J, 3)
            pred = pred_offsets[b]  # (N, J, 3)
            
            # Determine which joints are valid
            if mask is not None:
                valid_joints = mask[b]  # (J,)
            else:
                # Assume all joints with non-zero coords are valid
                valid_joints = joints.abs().sum(dim=-1) > 0  # (J,)
            
            if not valid_joints.any():
                continue
            
            # Only process valid joints
            for j in range(num_joints):
                if not valid_joints[j]:
                    continue
                
                joint_pos = joints[j]  # (3,)
                # Distance from each point to this joint
                dist = torch.norm(pts - joint_pos.unsqueeze(0), dim=-1)  # (N,)
                nearby_mask = dist < self.radius  # (N,)
                
                # DEBUG
                # if torch.rand(1).item() < 0.01 and j == 0:  
                #     import sys
                #     min_dist = dist.min().item()
                #     print(f"[DEBUG] Joint {j} pos: {joint_pos.cpu().numpy()}, "
                #           f"min_dist: {min_dist:.3f}m, radius: {self.radius}m, "
                #           f"nearby: {nearby_mask.sum().item()}/{len(dist)}",
                #           file=sys.stderr, flush=True)
                
                if not nearby_mask.any():
                    continue
                
                total_nearby_points += nearby_mask.sum().item()  # DEBUG
                
                # Target offset: joint - point
                target_offset = joint_pos.unsqueeze(0) - pts  # (N, 3)
                pred_offset = pred[:, j, :]  # (N, 3)
                
                # Only compute loss for nearby points
                nearby_pred = pred_offset[nearby_mask]  # (M, 3)
                nearby_target = target_offset[nearby_mask]  # (M, 3)
                
                joint_loss = F.mse_loss(nearby_pred, nearby_target, reduction='mean')
                loss_accum = loss_accum + joint_loss
                valid_count += 1

        if valid_count == 0:
            # DEBUG
            import sys
            print(f"[WARNING] No points within radius {self.radius}m of any joint! "
                  f"Points shape: {points_xyz.shape}, Joints shape: {gt_joints.shape}",
                  file=sys.stderr, flush=True)
            dummy_loss = (pred_offsets ** 2).sum() * 0.0
            return dummy_loss + self.loss_weight * 0.0
        
        if self.reduction == 'mean':
            loss_accum = loss_accum / valid_count
        
        # Statistics logging
        if torch.rand(1).item() < 0.01: 
            import sys
            print(f"[KeypointRadiusLoss] valid_joints: {valid_count}, "
                  f"nearby_points: {total_nearby_points}, loss: {loss_accum.item():.4f}",
                  file=sys.stderr, flush=True)
        
        return loss_accum * self.loss_weight
