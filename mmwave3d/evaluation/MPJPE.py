import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmwave3d.registry import METRICS


METRICS.register_module()
class MPJPE(BaseMetric):
    """Mean Per Joint Position Error (MPJPE) Metric.

    Args:
        prefix (str, optional): The prefix that will be added in front of
            metric names to disambiguate homonymous metrics of different
            evaluators. Defaults to None.
    """

    def __init__(self, prefix=None):
        super().__init__(prefix=prefix)

    def process(self, data_batch, data_samples):
        """Extract predictions + GT from model outputs."""
        for sample in data_samples:
            pred_joints = sample['pred_joints_3d']
            gt_joints = sample['gt_joints_3d']

            self.results.append(
                dict(pred_joints=pred_joints.cpu().numpy(),
                     gt_joints=gt_joints.cpu().numpy()))

    def compute_metrics(self, results):
        """Compute MPJPE from processed results."""
        logger = MMLogger.get_current_instance()
        total_mpjpe = 0.0
        total_count = 0

        for res in results:
            pred_joints = res['pred_joints']
            gt_joints = res['gt_joints']

            assert pred_joints.shape == gt_joints.shape, \
                "Predicted and ground truth joints must have the same shape."