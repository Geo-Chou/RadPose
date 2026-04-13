from itertools import permutations

import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmwave3d.registry import METRICS
from mmwave3d.utils import spherical_to_cartesian


@METRICS.register_module()
class MPJPE(BaseMetric):
    """
    Mean Per Joint Position Error with multi-person matching.
    
    Coordinate System:
    - Input coordinates are in spherical (r, azimuth, elevation)
    - Converts to Cartesian (x, y, z) for distance calculation
    - This is crucial because spherical dimensions have different physical scales:
      * R: [0.12, 6.0] meters (distance)
      * Azimuth/Elevation: [-π, π] radians (angles)
    - After conversion to Cartesian, all dimensions are in meters (comparable scale)
    - Final MPJPE reflects actual 3D position error in meters
    """

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
            pred, _ = self._sanitize_joints(res['pred_joints'])
            gt, gt_mask = self._sanitize_joints(res['gt_joints'])
            
            # Convert from spherical (r, azimuth, elevation) to Cartesian (x, y, z)
            # This is CRITICAL for physical error calculation:
            # - Spherical dimensions have different scales: R in meters [0.12-6], angles in radians [-π,π]
            # - Cartesian conversion ensures all dimensions are in meters with comparable scales
            pred = spherical_to_cartesian(pred)  # (num_persons, num_joints, 3) in meters
            gt = spherical_to_cartesian(gt)      # (num_persons, num_joints, 3) in meters
            
            match = self._match_people(pred, gt, gt_mask)
            for p_idx, g_idx in match:
                # Calculate error in Cartesian space (physical 3D space)
                diff = pred[p_idx] - gt[g_idx]  # (num_joints, 3) in meters
                
                # Compute L2 distance for each joint: sqrt(dx^2 + dy^2 + dz^2)
                joint_errors = torch.norm(diff, dim=-1)  # (num_joints,) in meters
                
                # Average across all joints to get per-person error
                mpjpe = joint_errors.mean().item()  # scalar in meters
                total_mpjpe += mpjpe
                total_pairs += 1
        
        mpjpe_value = total_mpjpe / max(total_pairs, 1)
        logger.info(f'MPJPE: {mpjpe_value:.4f} meters')
        return {'MPJPE': mpjpe_value}

    def _match_people(self, pred: torch.Tensor, gt: torch.Tensor,
                      gt_mask: torch.Tensor) -> list:
        """
        Match predicted persons to ground-truth persons using Hungarian algorithm.
        
        Note: All inputs (pred, gt) should already be in Cartesian coordinates (meters).
        Cost matrix is computed using Euclidean distance in physical 3D space.
        """
        if gt_mask is None:
            gt_mask = gt.abs().sum(dim=-1) > 0
        gt_valid = [i for i in range(gt.shape[0]) if gt_mask[i].any()]
        if not gt_valid or pred.shape[0] == 0:
            return []
        pred_ids = list(range(pred.shape[0]))
        gt_count = len(gt_valid)
        if len(pred_ids) < gt_count:
            return []
        
        # Build cost matrix: cost[i, j] = physical distance between pred[i] and gt[j]
        cost = pred.new_full((len(pred_ids), gt_count), float('inf'))
        for pi, pidx in enumerate(pred_ids):
            for gi, gidx in enumerate(gt_valid):
                joint_mask = gt_mask[gidx]
                if not joint_mask.any():
                    continue
                # All computations in Cartesian (physical) space
                diff = pred[pidx, joint_mask] - gt[gidx, joint_mask]  # (num_valid_joints, 3) in meters
                joint_distances = diff.norm(dim=-1)  # (num_valid_joints,) - L2 distance per joint in meters
                cost[pi, gi] = joint_distances.mean()  # average distance across joints
        
        # Hungarian algorithm: find optimal assignment minimizing total cost
        best_perm = None
        best_val = None
        for perm in permutations(range(len(pred_ids)), gt_count):
            total = sum(cost[perm[k], k] for k in range(gt_count))
            total_val = float(total.item())
            if best_val is None or total_val < best_val:
                best_val = total_val
                best_perm = perm
        if best_perm is None:
            return []
        
        matches = []
        for gi, pi in enumerate(best_perm):
            matches.append((pred_ids[pi], gt_valid[gi]))
        return matches

    def _sanitize_joints(self, joints: torch.Tensor):
        """
        Ensure joints are in (num_person, num_joints, 3) format and build masks.
        
        Coordinates at this point are in spherical format (r, azimuth, elevation).
        They will be converted to Cartesian in compute_metrics() for physical error calculation.
        
        Args:
            joints: Input tensor, could be (num_joints, 3), (num_person, num_joints, 3),
                   or with extra dimensions for confidence/mask
        
        Returns:
            joints: (num_person, num_joints, 3) - first 3 dims only
            mask: (num_person, num_joints) - boolean mask for valid joints
        """
        if joints.dim() == 2:
            joints = joints.unsqueeze(0)  # (1, num_joints, 3+)
        
        # Extract validity mask from first 3 coordinates or extra fields
        mask = joints[..., :3].abs().sum(dim=-1) > 0  # Has non-zero coordinate
        if joints.shape[-1] > 3:
            extra = joints[..., 3:]
            if extra.shape[-1] == 1:
                # Extra field is a single confidence/validity score
                mask = mask | (extra[..., 0] > 0)
            else:
                # Extra fields are additional coordinates/scores
                mask = mask | (extra.abs().sum(dim=-1) > 0)
            joints = joints[..., :3]  # Keep only first 3 dimensions
        
        return joints, mask
