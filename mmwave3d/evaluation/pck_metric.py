from itertools import permutations
from typing import List, Optional, Sequence, Tuple

import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmwave3d.registry import METRICS
from mmwave3d.utils import spherical_to_cartesian


@METRICS.register_module()
class PCKMetric(BaseMetric):
    """Percentage of Correct Keypoints with multi-person matching.

    Distances are computed in Cartesian 3D space (meters). A joint is counted
    as correct when its Euclidean distance is below a threshold.
    """

    def __init__(self,
                 thresholds: Sequence[float] = (0.12, 0.15),
                 prefix: Optional[str] = None) -> None:
        super().__init__(prefix=prefix)
        if not thresholds:
            raise ValueError('thresholds must contain at least one value.')
        self.thresholds = tuple(float(thr) for thr in thresholds)

    @staticmethod
    def _get_field(sample, name):
        if isinstance(sample, dict):
            return sample.get(name, None)
        if hasattr(sample, name):
            return getattr(sample, name)
        if hasattr(sample, 'get'):
            try:
                return sample.get(name, None)
            except Exception:
                return None
        return None

    def process(self, data_batch, data_samples) -> None:
        for data_sample in data_samples:
            pred = self._get_field(data_sample, 'pred_keypoints_3d')
            if pred is None:
                pred = self._get_field(data_sample, 'pred_keypoints')
            if pred is None:
                pred = self._get_field(data_sample, 'keypoints')
            gt = self._get_field(data_sample, 'gt_keypoints_3d')
            if pred is None or gt is None:
                continue
            self.results.append(dict(
                pred_keypoints=pred.detach().cpu(),
                gt_keypoints=gt.detach().cpu()))

    def compute_metrics(self, results: List[dict]) -> dict:
        logger = MMLogger.get_current_instance()
        hit_counts = torch.zeros(len(self.thresholds), dtype=torch.float64)
        total_count = 0
        joint_hit_counts = None
        joint_total_counts = None

        for result in results:
            pred, pred_mask = self._sanitize_joints(
                result['pred_keypoints'], treat_zero_as_invalid=False)
            gt, gt_mask = self._sanitize_joints(
                result['gt_keypoints'], treat_zero_as_invalid=True)

            pred_cart = spherical_to_cartesian(pred)
            gt_cart = spherical_to_cartesian(gt)
            matches = self._match_people(pred_cart, gt_cart, pred_mask, gt_mask)

            for pred_idx, gt_idx in matches:
                joint_mask = pred_mask[pred_idx] & gt_mask[gt_idx]
                if not joint_mask.any():
                    continue
                full_joint_errors = torch.norm(
                    pred_cart[pred_idx] - gt_cart[gt_idx], dim=-1)
                joint_errors = full_joint_errors[joint_mask]
                num_joints = int(full_joint_errors.numel())
                if joint_hit_counts is None:
                    joint_hit_counts = torch.zeros(
                        len(self.thresholds), num_joints, dtype=torch.float64)
                    joint_total_counts = torch.zeros(num_joints, dtype=torch.float64)

                total_count += int(joint_errors.numel())
                joint_total_counts += joint_mask.to(dtype=torch.float64)
                for thr_idx, threshold in enumerate(self.thresholds):
                    hit_counts[thr_idx] += (joint_errors <= threshold).sum().item()
                    joint_hits = ((full_joint_errors <= threshold) & joint_mask).to(dtype=torch.float64)
                    joint_hit_counts[thr_idx] += joint_hits

        metrics = {}
        denom = max(total_count, 1)
        for thr_idx, threshold in enumerate(self.thresholds):
            name = f'PCK@{threshold:.2f}'
            value = float(hit_counts[thr_idx] / denom)
            metrics[name] = value
            logger.info(f'{name}: {value:.4f}')
            if joint_hit_counts is not None and joint_total_counts is not None:
                for joint_idx in range(joint_hit_counts.size(1)):
                    joint_name = f'{name}/joint_{joint_idx}'
                    joint_denom = max(float(joint_total_counts[joint_idx].item()), 1.0)
                    joint_value = float(joint_hit_counts[thr_idx, joint_idx] / joint_denom)
                    metrics[joint_name] = joint_value
                    logger.info(f'{joint_name}: {joint_value:.4f}')
        return metrics

    def _sanitize_joints(self,
                         joints: torch.Tensor,
                         treat_zero_as_invalid: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if joints.dim() == 2:
            joints = joints.unsqueeze(0)

        joints = joints[..., :3]
        finite_mask = ~(torch.isnan(joints).any(dim=-1) | torch.isinf(joints).any(dim=-1))
        if treat_zero_as_invalid:
            non_zero_mask = joints.abs().sum(dim=-1) > 1e-8
            mask = finite_mask & non_zero_mask
        else:
            mask = finite_mask
        sanitized = torch.nan_to_num(joints, nan=0.0, posinf=0.0, neginf=0.0)
        return sanitized, mask

    def _match_people(self,
                      pred: torch.Tensor,
                      gt: torch.Tensor,
                      pred_mask: torch.Tensor,
                      gt_mask: torch.Tensor) -> List[Tuple[int, int]]:
        gt_valid = [idx for idx in range(gt.shape[0]) if gt_mask[idx].any()]
        pred_valid = [idx for idx in range(pred.shape[0]) if pred_mask[idx].any()]
        if not gt_valid or not pred_valid or len(pred_valid) < len(gt_valid):
            return []

        gt_count = len(gt_valid)
        cost = pred.new_full((len(pred_valid), gt_count), float('inf'))
        for pred_local, pred_idx in enumerate(pred_valid):
            for gt_local, gt_idx in enumerate(gt_valid):
                joint_mask = pred_mask[pred_idx] & gt_mask[gt_idx]
                if not joint_mask.any():
                    continue
                diff = pred[pred_idx, joint_mask] - gt[gt_idx, joint_mask]
                cost[pred_local, gt_local] = torch.norm(diff, dim=-1).mean()

        best_perm = None
        best_cost = None
        for perm in permutations(range(len(pred_valid)), gt_count):
            total = sum(cost[perm[k], k] for k in range(gt_count))
            total_value = float(total.item())
            if best_cost is None or total_value < best_cost:
                best_cost = total_value
                best_perm = perm

        if best_perm is None:
            return []
        return [(pred_valid[pred_local], gt_valid[gt_local])
                for gt_local, pred_local in enumerate(best_perm)]


@METRICS.register_module()
class IKPCKMetric(PCKMetric):
    """PCK computed from IK-refined keypoints."""

    def process(self, data_batch, data_samples) -> None:
        for data_sample in data_samples:
            pred = self._get_field(data_sample, 'pred_ik_keypoints_3d')
            if pred is None:
                pred = self._get_field(data_sample, 'ik_keypoints')
            if pred is None:
                pred = self._get_field(data_sample, 'pred_keypoints_3d')
            if pred is None:
                pred = self._get_field(data_sample, 'pred_keypoints')
            if pred is None:
                pred = self._get_field(data_sample, 'keypoints')
            gt = self._get_field(data_sample, 'gt_keypoints_3d')
            if pred is None or gt is None:
                continue
            self.results.append(dict(
                pred_keypoints=pred.detach().cpu(),
                gt_keypoints=gt.detach().cpu()))
