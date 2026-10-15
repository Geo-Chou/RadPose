from itertools import permutations
from typing import Optional

import torch
import torch.nn as nn

from mmwave3d.registry import MODELS
from mmwave3d.utils import spherical_to_cartesian


@MODELS.register_module()
class MPJPELoss(nn.Module):
    def __init__(self, reduction: str = "mean", loss_weight: float = 1.0) -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        pred = spherical_to_cartesian(pred)
        target = spherical_to_cartesian(target)
        mask = mask.to(dtype=torch.bool)
        if weights is not None:
            weights = weights.to(pred.device)
            if weights.shape[:3] != pred.shape[:3]:
                raise RuntimeError(
                    f"weights shape {weights.shape} does not match pred {pred.shape}"
                )
        total_loss = pred.new_zeros(())
        valid_samples = 0
        for batch_index in range(pred.size(0)):
            sample_weights = weights[batch_index] if weights is not None else None
            sample_loss = self._match_sample(
                pred[batch_index],
                target[batch_index],
                mask[batch_index],
                sample_weights,
            )
            if sample_loss is not None:
                total_loss = total_loss + sample_loss
                valid_samples += 1
        if self.reduction == "mean":
            total_loss = total_loss / max(valid_samples, 1)
        return total_loss * self.loss_weight

    def _match_sample(
        self,
        pred_person: torch.Tensor,
        gt_person: torch.Tensor,
        mask_person: torch.Tensor,
        weights_person: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        gt_indices = [
            index for index in range(gt_person.size(0)) if mask_person[index].any()
        ]
        if not gt_indices:
            return None
        num_pred = pred_person.size(0)
        gt_count = len(gt_indices)
        cost = pred_person.new_zeros((num_pred, gt_count))
        for pred_index in range(num_pred):
            for local_gt_index, gt_index in enumerate(gt_indices):
                joint_mask = mask_person[gt_index]
                if not joint_mask.any():
                    cost[pred_index, local_gt_index] = 1e6
                    continue
                difference = (
                    pred_person[pred_index, joint_mask]
                    - gt_person[gt_index, joint_mask]
                )
                if weights_person is None:
                    cost[pred_index, local_gt_index] = difference.norm(dim=-1).mean()
                else:
                    joint_weights = weights_person[pred_index, joint_mask].to(
                        difference.dtype
                    )
                    cost[pred_index, local_gt_index] = (
                        difference.norm(dim=-1) * joint_weights
                    ).sum() / joint_weights.sum().clamp_min(1e-6)
        best_permutation = min(
            permutations(range(num_pred), gt_count),
            key=lambda permutation: float(
                sum(cost[permutation[i], i] for i in range(gt_count)).item()
            ),
        )
        sample_loss = pred_person.new_zeros(())
        for local_gt_index, pred_index in enumerate(best_permutation):
            gt_index = gt_indices[local_gt_index]
            joint_mask = mask_person[gt_index]
            difference = (
                pred_person[pred_index, joint_mask] - gt_person[gt_index, joint_mask]
            )
            if weights_person is None:
                sample_loss = sample_loss + difference.norm(dim=-1).mean()
            else:
                joint_weights = weights_person[pred_index, joint_mask].to(
                    difference.dtype
                )
                sample_loss = sample_loss + (
                    difference.norm(dim=-1) * joint_weights
                ).sum() / joint_weights.sum().clamp_min(1e-6)
        return sample_loss / gt_count
