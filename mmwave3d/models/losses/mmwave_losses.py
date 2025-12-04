from itertools import permutations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmwave3d.registry import MODELS


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
                mask: torch.Tensor) -> torch.Tensor:
        # Replace NaNs/Infs before scoring, otherwise torch.norm will return NaN
        # and poison the full batch.
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        mask = mask.to(dtype=torch.bool)
        total_loss = pred.new_zeros(())
        valid_samples = 0
        for b in range(pred.size(0)):
            sample_loss = self._match_sample(pred[b], target[b], mask[b])
            if sample_loss is None:
                continue
            total_loss = total_loss + sample_loss
            valid_samples += 1
        if self.reduction == 'mean':
            total_loss = total_loss / max(valid_samples, 1)
        return total_loss * self.loss_weight

    def _match_sample(self, pred_person: torch.Tensor,
                      gt_person: torch.Tensor,
                      mask_person: torch.Tensor) -> Optional[torch.Tensor]:
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
            sample_loss = sample_loss + diff.norm(dim=-1).mean()
        return sample_loss / gt_count


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
        loss = F.binary_cross_entropy(
            person_prob.clamp(min=1e-6, max=1 - 1e-6),
            gt_person,
            reduction=self.reduction)
        return loss * self.loss_weight
