import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmwave3d.registry import METRICS


@METRICS.register_module()
class MPJPE(BaseMetric):
    """Mean Per Joint Position Error with multi-person matching."""

    def __init__(self, prefix: str = None) -> None:
        super().__init__(prefix=prefix)

    def process(self, data_batch, data_samples) -> None:
        for data_sample in data_samples:
            pred = data_sample.get('keypoints', None)
            scores = data_sample.get('scores', None)
            gt = data_sample.get('gt_keypoints_3d', None)
            if pred is None or gt is None:
                continue
            self.results.append(
                dict(pred_joints=pred.cpu(), pred_scores=scores.cpu()
                     if scores is not None else None,
                     gt_joints=gt.cpu()))

    def compute_metrics(self, results):
        logger = MMLogger.get_current_instance()
        total_mpjpe = 0.0
        total_pairs = 0
        for res in results:
            pred = self._sanitize_joints(res['pred_joints'])
            gt = self._sanitize_joints(res['gt_joints'])
            match = self._match_people(pred, gt)
            for p_idx, g_idx in match:
                diff = pred[p_idx] - gt[g_idx]
                mpjpe = torch.norm(diff, dim=-1).mean().item()
                total_mpjpe += mpjpe
                total_pairs += 1
        mpjpe_value = total_mpjpe / max(total_pairs, 1)
        logger.info(f'MPJPE: {mpjpe_value:.4f}')
        return {'MPJPE': mpjpe_value}

    def _match_people(self, pred: torch.Tensor,
                      gt: torch.Tensor) -> list:
        gt_valid = [i for i in range(gt.shape[0]) if gt[i].abs().sum() > 0]
        if not gt_valid:
            return []
        cost = torch.zeros((pred.shape[0], len(gt_valid)))
        for i in range(pred.shape[0]):
            for j, gidx in enumerate(gt_valid):
                diff = pred[i] - gt[gidx]
                cost[i, j] = torch.norm(diff, dim=-1).mean()
        best_pairs = []
        used_pred = set()
        for j, gidx in enumerate(gt_valid):
            best_i = None
            best_val = None
            for i in range(pred.shape[0]):
                if i in used_pred:
                    continue
                val = cost[i, j].item()
                if best_val is None or val < best_val:
                    best_val = val
                    best_i = i
            if best_i is not None:
                used_pred.add(best_i)
                best_pairs.append((best_i, gidx))
        return best_pairs

    def _sanitize_joints(self, joints: torch.Tensor) -> torch.Tensor:
        """Ensure joints are in (num_person, num_joints, 3) format."""
        if joints.dim() == 2:
            joints = joints.unsqueeze(0)
        if joints.shape[-1] > 3:
            joints = joints[..., :3]
        return joints
