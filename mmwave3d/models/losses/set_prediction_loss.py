
from itertools import permutations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmwave3d.registry import MODELS
from mmwave3d.utils import spherical_to_cartesian
from .sinkhorn import sinkhorn_algorithm


@MODELS.register_module()
class SetPredictionLoss(nn.Module):
    """
    Set Prediction Loss

    1. "no-object" cretieria
    2. no-object: push r value to no_object_r
    3. learn how many persons are present implicitly
    """
    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 no_object_r: float = 10.0,  
                 no_object_weight: float = 0.5) -> None:
        super().__init__()
        assert reduction in ['mean', 'sum']
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_object_r = no_object_r
        self.no_object_weight = no_object_weight

    def forward(self,
                pred: torch.Tensor,  # (B, num_pred, num_joints, 3)
                target: torch.Tensor,  # (B, num_gt, num_joints, 3)
                mask: torch.Tensor) -> torch.Tensor:  # (B, num_gt, num_joints)
        
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        
        pred_cart = spherical_to_cartesian(pred)
        target_cart = spherical_to_cartesian(target)
        mask = mask.to(dtype=torch.bool)
        
        total_loss = pred.new_zeros(())
        valid_samples = 0
        
        for b in range(pred.size(0)):
            sample_loss = self._match_sample(
                pred_cart[b], target_cart[b], mask[b], pred[b])
            if sample_loss is None:
                continue
            total_loss = total_loss + sample_loss
            valid_samples += 1
        
        if self.reduction == 'mean':
            total_loss = total_loss / max(valid_samples, 1)
        return total_loss * self.loss_weight

    def _match_sample(self, 
                      pred_person: torch.Tensor,  # (num_pred, num_joints, 3)
                      gt_person: torch.Tensor,    # (num_gt, num_joints, 3)
                      mask_person: torch.Tensor,  # (num_gt, num_joints)
                      pred_spherical: torch.Tensor) -> Optional[torch.Tensor]:
        # right now, only support batch size = 1
        gt_indices = [i for i in range(gt_person.size(0))
                      if mask_person[i].any()]
        
        if len(gt_indices) == 0:
            # no gt, no object
            return self._no_object_loss(pred_spherical)
        
        num_pred = pred_person.size(0)
        gt_count = len(gt_indices)
        
        # cost matrix
        cost = pred_person.new_zeros((num_pred, gt_count))
        for pi in range(num_pred):
            for gi, gidx in enumerate(gt_indices):
                joint_mask = mask_person[gidx]
                if joint_mask.any():
                    diff = pred_person[pi, joint_mask] - gt_person[gidx, joint_mask]
                    cost[pi, gi] = diff.norm(dim=-1).mean()
                else:
                    cost[pi, gi] = pred_person.new_tensor(1e6)
        
        # Hungarian matching using scipy (faster and more stable)
        cost_np = cost.cpu().detach().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        best_perm = row_ind.tolist()
        
        # compute loss
        matched_loss = pred_person.new_zeros(())
        matched_slots = set()
        for gi, pi in enumerate(best_perm):
            gt_idx = gt_indices[gi]
            joint_mask = mask_person[gt_idx]
            diff = pred_person[pi, joint_mask] - gt_person[gt_idx, joint_mask]
            matched_loss = matched_loss + diff.norm(dim=-1).mean()
            matched_slots.add(pi)
        matched_loss = matched_loss / gt_count
        
        # unmatched slot no-object loss
        unmatched_slots = [i for i in range(num_pred) if i not in matched_slots]
        if len(unmatched_slots) > 0:
            no_obj_loss = self._no_object_loss(
                pred_spherical[unmatched_slots])
            total_loss = matched_loss + self.no_object_weight * no_obj_loss
        else:
            total_loss = matched_loss
        
        return total_loss
    
    def _no_object_loss(self, pred_spherical: torch.Tensor) -> torch.Tensor:
        """
        no-object loss: Push unmatched slots far away (r = no_object_r)
        
        We want unmatched slots to be confident that no one is present.
        Approach: Target a large radial distance that's outside observation range.
        After Cartesian conversion, this creates large coordinate values.
        """
        # pred_spherical shape: (num_slots, num_joints, 3) or (num_joints, 3)
        # Convert to Cartesian space for physically meaningful loss
        pred_cart = spherical_to_cartesian(pred_spherical)
        
        # Target a "no object" location: far away in Cartesian space
        # Using r=no_object_r, az=0, el=0 as target
        target_spherical = torch.zeros_like(pred_spherical)
        target_spherical[..., 0] = self.no_object_r  # r
        # az and el remain 0
        target_cart = spherical_to_cartesian(target_spherical)
        
        # Compute distance in Cartesian space
        diff = pred_cart - target_cart
        loss = diff.norm(dim=-1).mean()
        return loss


@MODELS.register_module()
class SetPredictionLossV2(nn.Module):
    """
    V2: use two classification confidence
    
    Coordinate system handling:
    - Input coords are in spherical (r, azimuth, elevation)
    - R, azimuth, elevation have VASTLY different physical ranges:
      * R: [0.12m, 6.0m] - physical distance
      * Azimuth: [-π/6, 5π/6] - angles in radians
      * Elevation: [-π/6, π/6] - angles in radians
    - These cannot be directly compared in spherical space due to scale mismatch
    - Solution: Convert to Cartesian (x, y, z) and compute L2 distance
    - After conversion, all dimensions are in meters with comparable scales
    
    Loss components:
    1. matched slot: coordinate loss + confidence=1 BCE loss
    2. unmatched slot: confidence=0 BCE loss
    """
    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 no_object_weight: float = 0.5,
                 coord_weight: float = 1.0,
                 conf_weight: float = 1.0,
                 use_sinkhorn: bool = False,
                 sinkhorn_iters: int = 100,
                 sinkhorn_temperature: float = 0.1) -> None:
        super().__init__()
        assert reduction in ['mean', 'sum']
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.no_object_weight = no_object_weight
        self.coord_weight = coord_weight
        self.conf_weight = conf_weight
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_temperature = sinkhorn_temperature

    def forward(self,
                pred: torch.Tensor,  # (B, num_pred, num_joints, 3)
                pred_scores: torch.Tensor,  # (B, num_pred) - per-person confidence
                target: torch.Tensor,  # (B, num_gt, num_joints, 3)
                mask: torch.Tensor) -> dict:  # (B, num_gt, num_joints)
        
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        
        pred_cart = spherical_to_cartesian(pred)
        target_cart = spherical_to_cartesian(target)
        mask = mask.to(dtype=torch.bool)
        
        total_coord_loss = pred.new_zeros(())
        total_conf_loss = pred.new_zeros(())
        valid_samples = 0
        
        for b in range(pred.size(0)):
            coord_loss, conf_loss = self._match_sample(
                pred_cart[b], pred_scores[b], target_cart[b], mask[b])
            if coord_loss is None:
                continue
            total_coord_loss = total_coord_loss + coord_loss
            total_conf_loss = total_conf_loss + conf_loss
            valid_samples += 1
        
        if self.reduction == 'mean':
            total_coord_loss = total_coord_loss / max(valid_samples, 1)
            total_conf_loss = total_conf_loss / max(valid_samples, 1)
        
        return {
            'loss_coord': total_coord_loss * self.coord_weight * self.loss_weight,
            'loss_conf': total_conf_loss * self.conf_weight * self.loss_weight
        }

    def _match_sample(self, 
                      pred_person: torch.Tensor,
                      pred_conf: torch.Tensor,  # (num_pred,)
                      gt_person: torch.Tensor,
                      mask_person: torch.Tensor):
        
        gt_indices = [i for i in range(gt_person.size(0))
                      if mask_person[i].any()]
        
        num_pred = pred_person.size(0)
        
        if len(gt_indices) == 0:
            # confidence=0
            target_conf = torch.zeros_like(pred_conf)
            conf_loss = F.binary_cross_entropy_with_logits(
                pred_conf, target_conf, reduction='mean')
            return pred_person.new_zeros(()), conf_loss
        
        gt_count = len(gt_indices)
        
        if self.use_sinkhorn:
            # Sinkhorn soft matching
            return self._sinkhorn_match(pred_person, pred_conf, gt_person, 
                                       mask_person, gt_indices)
        else:
            # Hungarian hard matching
            return self._hungarian_match(pred_person, pred_conf, gt_person,
                                        mask_person, gt_indices)
    
    def _hungarian_match(self,
                        pred_person: torch.Tensor,
                        pred_conf: torch.Tensor,
                        gt_person: torch.Tensor,
                        mask_person: torch.Tensor,
                        gt_indices: list):
        """Hungarian hard matching"""
        num_pred = pred_person.size(0)
        gt_count = len(gt_indices)
        
        # Build cost matrix with both coordinate distance and confidence
        cost = pred_person.new_zeros((num_pred, gt_count))
        for pi in range(num_pred):
            for gi, gidx in enumerate(gt_indices):
                joint_mask = mask_person[gidx]
                if joint_mask.any():
                    diff = pred_person[pi, joint_mask] - gt_person[gidx, joint_mask]
                    coord_cost = diff.norm(dim=-1).mean()
                    
                    cost[pi, gi] = coord_cost
                else:
                    cost[pi, gi] = 1e6
        
        # Hungarian matching using scipy (O(n³) instead of O(n!))
        cost_np = cost.cpu().detach().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)
        
        # Coordinate loss
        coord_loss = pred_person.new_zeros(())
        matched_slots = set()
        
        for pred_idx, gt_idx_in_cost in zip(row_ind, col_ind):
            gt_idx = gt_indices[gt_idx_in_cost]
            joint_mask = mask_person[gt_idx]
            diff = pred_person[pred_idx, joint_mask] - gt_person[gt_idx, joint_mask]
            coord_loss = coord_loss + diff.norm(dim=-1).mean()
            matched_slots.add(pred_idx)
        coord_loss = coord_loss / gt_count
        
        # Confidence loss (hard targets: 0 or 1)
        target_conf = torch.zeros_like(pred_conf)
        for pi in matched_slots:
            target_conf[pi] = 1.0
        
        conf_loss = F.binary_cross_entropy_with_logits(
            pred_conf, target_conf, reduction='mean')
        
        return coord_loss, conf_loss
    
    def _sinkhorn_match(self,
                       pred_person: torch.Tensor,
                       pred_conf: torch.Tensor,
                       gt_person: torch.Tensor,
                       mask_person: torch.Tensor,
                       gt_indices: list):
        """Sinkhorn  soft matching"""
        num_pred = pred_person.size(0)
        gt_count = len(gt_indices)
        
        # Build cost matrix
        cost = pred_person.new_zeros((num_pred, gt_count))
        for pi in range(num_pred):
            for gi, gidx in enumerate(gt_indices):
                joint_mask = mask_person[gidx]
                if joint_mask.any():
                    diff = pred_person[pi, joint_mask] - gt_person[gidx, joint_mask]
                    coord_cost = diff.norm(dim=-1).mean()
                    
                    # Confidence-guided matching
                    conf_penalty = -0.1 * pred_conf[pi].detach().sigmoid()
                    
                    cost[pi, gi] = coord_cost + conf_penalty
                else:
                    cost[pi, gi] = 1e6
        
        # Sinkhorn algorithm to get soft assignment
        assignment = sinkhorn_algorithm(
            cost, 
            num_iters=self.sinkhorn_iters,
            temperature=self.sinkhorn_temperature)
        
        # Use soft assignment to compute weighted coordinate loss
        coord_loss = pred_person.new_zeros(())
        for pi in range(num_pred):
            for gi, gidx in enumerate(gt_indices):
                weight = assignment[pi, gi]
                joint_mask = mask_person[gidx]
                if joint_mask.any() and weight > 1e-6:
                    diff = pred_person[pi, joint_mask] - gt_person[gidx, joint_mask]
                    coord_loss = coord_loss + weight * diff.norm(dim=-1).mean()
        
        # Normalize - 使用tensor操作保持梯度流，不要使用.item()
        # BUG FIX: 之前使用assignment.sum().item()会断开梯度链
        coord_loss = coord_loss / torch.clamp(assignment.sum(), min=1.0)
        
        # Confidence loss: Use soft assignment as soft target
        # target_conf[i] = sum_j assignment[i,j]
        # This provides a continuous gradient signal
        target_conf = assignment.sum(dim=1)  # (num_pred,)
        conf_loss = F.binary_cross_entropy_with_logits(
            pred_conf, target_conf, reduction='mean')
        
        return coord_loss, conf_loss
