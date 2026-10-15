from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmwave3d.registry import MODELS
from mmwave3d.utils import spherical_to_cartesian

SKELETON_EDGES_17 = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (3, 5),
    (5, 7),
    (7, 9),
    (2, 4),
    (4, 6),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (11, 12),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

SKELETON_EDGES_16 = [
    (0, 1),
    (1, 2),
    (2, 6),
    (5, 4),
    (4, 3),
    (3, 6),
    (6, 7),
    (10, 11),
    (11, 12),
    (12, 7),
    (15, 14),
    (14, 13),
    (13, 7),
    (7, 8),
    (8, 9),
]


def _edge_index(device: torch.device) -> torch.Tensor:
    return torch.tensor(SKELETON_EDGES_16, device=device, dtype=torch.long)


@MODELS.register_module()
class SkeletonLengthLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        spherical_input: bool = True,
        loss_type: str = "l1",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert reduction in ["mean", "sum"]
        assert loss_type in ["l1", "smooth_l1"]
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.spherical_input = spherical_input
        self.loss_type = loss_type
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.spherical_input:
            pred = spherical_to_cartesian(pred)
            target = spherical_to_cartesian(target)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        max(self.eps, torch.finfo(pred.dtype).eps)

        edges = _edge_index(pred.device)
        j1, j2 = edges[:, 0], edges[:, 1]

        pred_vec = pred[..., j2, :] - pred[..., j1, :]
        target_vec = target[..., j2, :] - target[..., j1, :]

        pred_len = pred_vec.norm(dim=-1)
        target_len = target_vec.norm(dim=-1)

        if self.loss_type == "smooth_l1":
            per_edge = F.smooth_l1_loss(pred_len, target_len, reduction="none")
        else:
            per_edge = (pred_len - target_len).abs()

        if mask is not None:
            mask = mask.to(dtype=torch.bool)
            edge_mask = (mask[..., j1] & mask[..., j2]).to(dtype=per_edge.dtype)
            per_edge = per_edge * edge_mask
            denom = edge_mask.sum().clamp_min(1.0)
        else:
            denom = per_edge.new_tensor(per_edge.numel())

        if self.reduction == "mean":
            loss = per_edge.sum() / denom
        else:
            loss = per_edge.sum()
        return loss * self.loss_weight


@MODELS.register_module()
class SkeletonAngleLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        spherical_input: bool = True,
        use_arccos: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert reduction in ["mean", "sum"]
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.spherical_input = spherical_input
        self.use_arccos = use_arccos
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.spherical_input:
            pred = spherical_to_cartesian(pred)
            target = spherical_to_cartesian(target)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        edges = _edge_index(pred.device)
        j1, j2 = edges[:, 0], edges[:, 1]

        pred_vec = pred[..., j2, :] - pred[..., j1, :]
        target_vec = target[..., j2, :] - target[..., j1, :]

        pred_vec = pred_vec.float()
        target_vec = target_vec.float()
        pred_len = pred_vec.norm(dim=-1).clamp_min(self.eps)
        target_len = target_vec.norm(dim=-1).clamp_min(self.eps)

        cos_sim = (pred_vec * target_vec).sum(dim=-1) / (pred_len * target_len)
        cos_sim = cos_sim.clamp(-1.0 + self.eps, 1.0 - self.eps)
        if self.use_arccos:
            per_edge = torch.acos(cos_sim)
        else:
            per_edge = 1.0 - cos_sim

        if mask is not None:
            mask = mask.to(dtype=torch.bool)
            edge_mask = (mask[..., j1] & mask[..., j2]).to(dtype=per_edge.dtype)
            per_edge = per_edge * edge_mask
            denom = edge_mask.sum().clamp_min(1.0)
        else:
            denom = per_edge.new_tensor(per_edge.numel())

        if self.reduction == "mean":
            loss = per_edge.sum() / denom
        else:
            loss = per_edge.sum()
        return loss * self.loss_weight
