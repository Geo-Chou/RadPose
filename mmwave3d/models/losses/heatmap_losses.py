from typing import List

import torch
import torch.nn as nn

try:
    from scipy.optimize import linear_sum_assignment

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from mmwave3d.registry import MODELS


def compute_heatmap_cost_matrix(
    pred_heatmaps: torch.Tensor,
    target_heatmaps: torch.Tensor,
    joint_masks: torch.Tensor,
) -> torch.Tensor:
    P, J, H, W = pred_heatmaps.shape
    num_gt = target_heatmaps.size(0)
    with torch.no_grad():
        joint_masks = joint_masks.to(pred_heatmaps.device).float()

        diff2 = (pred_heatmaps.unsqueeze(1) - target_heatmaps.unsqueeze(0)) ** 2
        joint_mask = joint_masks.view(1, num_gt, J, 1, 1)
        diff2 = diff2 * joint_mask
        denom = joint_mask.sum(dim=2).clamp(min=1.0) * H * W
        denom = denom.squeeze(-1).squeeze(-1)
        cost_matrix = diff2.sum(dim=(2, 3, 4)) / denom
    return cost_matrix


def find_best_assignment(cost_matrix: torch.Tensor, num_gt: int) -> List[int]:
    P = cost_matrix.size(0)

    if HAS_SCIPY:
        cost_np = cost_matrix.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        assignment = [-1] * num_gt
        used_preds = set()

        for p_idx, g_idx in zip(row_ind, col_ind):
            if g_idx < num_gt:
                assignment[g_idx] = int(p_idx)
                used_preds.add(int(p_idx))

        for gi in range(num_gt):
            if assignment[gi] == -1:
                best_p = -1
                min_c = float("inf")
                for p in range(P):
                    if p not in used_preds:
                        c = cost_matrix[p, gi].item()
                        if c < min_c:
                            min_c = c
                            best_p = p
                if best_p != -1:
                    assignment[gi] = best_p
                    used_preds.add(best_p)
                else:
                    assignment[gi] = 0
        return assignment

    assignment = []
    used_preds = set()
    for gi in range(num_gt):
        best_p = -1
        min_c = float("inf")
        for p in range(P):
            if p in used_preds:
                continue
            if cost_matrix[p, gi] < min_c:
                min_c = cost_matrix[p, gi]
                best_p = p
        if best_p != -1:
            assignment.append(best_p)
            used_preds.add(best_p)
        else:
            assignment.append(0)
    return assignment


@MODELS.register_module()
class Heatmap2DLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 4.0,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred = pred.float()
        target = target.float()
        eps = 1e-8
        pred = pred.clamp(eps, 1 - eps)

        pos_loss = (
            -self.alpha * torch.pow(1 - pred, self.beta) * target * torch.log(pred)
        )
        neg_weight = 1.0 if self.alpha >= 1.0 else (1.0 - self.alpha)
        neg_loss = (
            -neg_weight
            * torch.pow(pred, self.beta)
            * (1 - target)
            * torch.log(1 - pred)
        )

        pos_mask = (target > 0).float()
        neg_mask = 1.0 - pos_mask

        pos_term = pos_loss * pos_mask
        neg_term = neg_loss * neg_mask

        if self.reduction == "mean":
            pos_count = pos_mask.sum().clamp(min=1.0)
            neg_count = neg_mask.sum().clamp(min=1.0)
            pos_mean = pos_term.sum() / pos_count
            neg_mean = neg_term.sum() / neg_count
            return pos_mean + neg_mean
        elif self.reduction == "sum":
            return pos_term.sum() + neg_term.sum()
        else:
            return pos_term + neg_term

    def forward(
        self,
        pred_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
        masks: torch.Tensor,
        perform_matching: bool = True,
    ) -> torch.Tensor:
        B, P, J, H, W = pred_heatmaps.shape
        device = pred_heatmaps.device

        if masks.dim() == 2:
            masks = masks.view(B, P, J)

        if not perform_matching:
            loss = self.focal_loss(pred_heatmaps, target_heatmaps)
            return loss * self.loss_weight

        total_loss = torch.tensor(0.0, device=device)
        valid_samples = 0

        for b in range(B):
            person_valid = masks[b].any(dim=1)
            valid_gt_indices = torch.where(person_valid)[0]
            num_gt = len(valid_gt_indices)

            if num_gt == 0:
                loss = self.focal_loss(pred_heatmaps[b], target_heatmaps[b])
                total_loss += loss
                valid_samples += 1
                continue

            cost_matrix = compute_heatmap_cost_matrix(
                pred_heatmaps[b],
                target_heatmaps[b, valid_gt_indices],
                masks[b, valid_gt_indices],
            )

            best_assignment = find_best_assignment(cost_matrix, num_gt)

            sample_loss = torch.tensor(0.0, device=device)

            matched_preds = set()
            for gt_idx_local, pred_idx in enumerate(best_assignment):
                gt_idx_global = valid_gt_indices[gt_idx_local]
                matched_preds.add(pred_idx)

                loss = self.focal_loss(
                    pred_heatmaps[b : b + 1, pred_idx : pred_idx + 1],
                    target_heatmaps[b : b + 1, gt_idx_global : gt_idx_global + 1],
                )
                sample_loss += loss

            all_preds = set(range(P))
            unmatched_preds = all_preds - matched_preds

            if unmatched_preds:
                unmatched_idx = torch.tensor(list(unmatched_preds), device=device)
                empty_target = torch.zeros_like(pred_heatmaps[b : b + 1, unmatched_idx])

                loss_fp = self.focal_loss(
                    pred_heatmaps[b : b + 1, unmatched_idx], empty_target
                )

                sample_loss += loss_fp * 0.5

            total_loss += sample_loss
            valid_samples += 1

        if valid_samples > 0:
            total_loss = total_loss / valid_samples

        return total_loss * self.loss_weight

    def _find_best_assignment(
        self, cost_matrix: torch.Tensor, num_gt: int
    ) -> List[int]:
        return find_best_assignment(cost_matrix, num_gt)


@MODELS.register_module()
class HeatmapFocalLossNoMatch(Heatmap2DLoss):
    def forward(
        self,
        pred_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        if masks.dim() == 2:
            B, P, J, _, _ = pred_heatmaps.shape
            masks = masks.view(B, P, J)

        joint_mask = masks.view(*masks.shape, 1, 1).float()

        pred_heatmaps = pred_heatmaps.float()
        target_heatmaps = target_heatmaps.float()
        eps = 1e-8
        pred = pred_heatmaps.clamp(eps, 1 - eps)

        pos_mask = (target_heatmaps > 0) & (joint_mask > 0)
        neg_mask = (target_heatmaps <= 0) & (joint_mask > 0)

        pos_loss = (
            -self.alpha
            * torch.pow(1 - pred, self.beta)
            * target_heatmaps
            * torch.log(pred)
        )
        neg_weight = 1.0 if self.alpha >= 1.0 else (1.0 - self.alpha)
        neg_loss = (
            -neg_weight
            * torch.pow(pred, self.beta)
            * (1 - target_heatmaps)
            * torch.log(1 - pred)
        )

        pos_term = pos_loss * pos_mask
        neg_term = neg_loss * neg_mask

        if self.reduction == "mean":
            pos_count = pos_mask.sum().clamp(min=1.0)
            neg_count = neg_mask.sum().clamp(min=1.0)
            pos_mean = pos_term.sum() / pos_count
            neg_mean = neg_term.sum() / neg_count
            loss = pos_mean + neg_mean
        elif self.reduction == "sum":
            loss = pos_term.sum() + neg_term.sum()
        else:
            loss = pos_term + neg_term

        return loss * self.loss_weight


@MODELS.register_module()
class HeatmapMSELoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        pos_weight_k: float = 20.0,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.pos_weight_k = pos_weight_k

    def forward(
        self,
        pred_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
        masks: torch.Tensor,
        perform_matching: bool = True,
    ) -> torch.Tensor:
        B, P, J, H, W = pred_heatmaps.shape
        device = pred_heatmaps.device

        if masks.dim() == 2:
            masks = masks.view(B, P, J)

        if not perform_matching:
            diff = (pred_heatmaps - target_heatmaps) ** 2
            loss = diff.mean()
            return loss * self.loss_weight

        total_loss = torch.tensor(0.0, device=device)
        valid_samples = 0

        for b in range(B):
            person_valid = masks[b].any(dim=1)
            valid_gt_indices = torch.where(person_valid)[0]
            num_gt = len(valid_gt_indices)

            if num_gt == 0:
                loss = (pred_heatmaps[b] ** 2).mean()
                total_loss += loss
                valid_samples += 1
                continue

            cost_matrix = compute_heatmap_cost_matrix(
                pred_heatmaps[b],
                target_heatmaps[b, valid_gt_indices],
                masks[b, valid_gt_indices],
            )

            best_assignment = find_best_assignment(cost_matrix, num_gt)

            sample_loss = torch.tensor(0.0, device=device)
            matched_preds = set()

            for gt_idx_local, pred_idx in enumerate(best_assignment):
                gt_idx_global = valid_gt_indices[gt_idx_local]
                matched_preds.add(pred_idx)

                diff = (
                    pred_heatmaps[b, pred_idx] - target_heatmaps[b, gt_idx_global]
                ) ** 2
                weight = 1.0 + self.pos_weight_k * target_heatmaps[b, gt_idx_global]
                weighted_diff = diff * weight

                sample_loss += weighted_diff.sum() / weight.sum().clamp(min=1.0)

            all_preds = set(range(P))
            unmatched_preds = all_preds - matched_preds
            if unmatched_preds:
                unmatched_idx = torch.tensor(list(unmatched_preds), device=device)
                loss_fp = (pred_heatmaps[b, unmatched_idx] ** 2).mean()
                sample_loss += loss_fp * 0.5

            total_loss += sample_loss
            valid_samples += 1

        if valid_samples > 0:
            total_loss = total_loss / valid_samples

        return total_loss * self.loss_weight

    def _find_best_assignment(
        self, cost_matrix: torch.Tensor, num_gt: int
    ) -> List[int]:
        return find_best_assignment(cost_matrix, num_gt)


@MODELS.register_module()
class ElevationLoss(nn.Module):
    def __init__(
        self, threshold: float = 0.1, reduction: str = "mean", loss_weight: float = 1.0
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred_elevation: torch.Tensor,
        target_elevation: torch.Tensor,
        heatmap_weight: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:

        B, P, J, H, W = pred_elevation.shape
        if masks.dim() == 2:
            masks = masks.view(B, P, J)

        weight_mask = (heatmap_weight > self.threshold).float()

        joint_mask = masks.float().view(B, P, J, 1, 1).expand_as(pred_elevation)
        weight_mask = weight_mask * joint_mask

        diff = torch.abs(pred_elevation - target_elevation)

        weighted_loss = diff * weight_mask

        total_weight = weight_mask.sum().clamp(min=1.0)

        if self.reduction == "mean":
            loss = weighted_loss.sum() / total_weight
        elif self.reduction == "sum":
            loss = weighted_loss.sum()
        else:
            loss = weighted_loss

        return loss * self.loss_weight
