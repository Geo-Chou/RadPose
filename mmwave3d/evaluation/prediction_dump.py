import json
from pathlib import Path
from typing import List, Optional, Tuple
from itertools import permutations

import torch
import numpy as np
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from mmwave3d.registry import METRICS
from mmwave3d.utils import spherical_to_cartesian


@METRICS.register_module()
class PredictionDump(BaseMetric):
    """Dump filtered predictions to JSON files per sample."""

    def __init__(self,
                 out_dir: str,
                 score_thr: float = 0.,
                 min_joints: int = 7,
                 prefix: Optional[str] = None) -> None:
        super().__init__(prefix=prefix)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.score_thr = score_thr
        self.min_joints = min_joints
        self._counter = 0

    def process(self, data_batch, data_samples: List) -> None:
        if not is_main_process():
            return
        for data_sample in data_samples:
            # print(data_sample['sample_name'])
            keypoints = data_sample.get('pred_keypoints_3d', None)
            if keypoints is None:
                keypoints = data_sample.get('pred_keypoints', None)
            if keypoints is None:
                keypoints = data_sample.get('keypoints', None)
            
            # Support both old format ('scores') and new HeadV2 format ('person_scores', 'joint_scores')
            if 'person_scores' in data_sample:
                # HeadV2 format: use person_scores for filtering
                person_scores = data_sample.get('person_scores', None)
                joint_scores = data_sample.get('joint_scores', None)
                if keypoints is None or person_scores is None:
                    continue
                keypoints = keypoints.detach()
                person_scores = person_scores.detach()
                joint_scores = joint_scores.detach() if joint_scores is not None else None
                
                # Filter by person_scores (per-person confidence)
                valid_mask = person_scores > self.score_thr
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            else:
                # Old format: use joint_scores for filtering
                scores = data_sample.get('scores', None)
                print('scores:', scores)
                if keypoints is None or scores is None:
                    continue
                keypoints = keypoints.detach()
                scores = scores.detach()
                
                # Filter by joint_scores: count joints above threshold
                valid_mask = (scores > self.score_thr).sum(dim=-1) > self.min_joints
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            
            if valid_indices.numel() == 0:
                continue
            sample_name = self._resolve_sample_name(data_sample)
            self._dump_sample(sample_name, keypoints, valid_indices)

    def compute_metrics(self, results):
        return {}

    def _resolve_sample_name(self, data_sample) -> Path:
        metainfo = getattr(data_sample, 'metainfo', {})
        name = getattr(data_sample, 'dump_name', None)
        if not name:
            try:
                name = data_sample['sample_name']
            except Exception:
                name = None
        if not name and hasattr(metainfo, 'get'):
            name = metainfo.get('sample_name')
        if name:
            name = str(name)
        paths = metainfo.get('radar_points_paths', None)
        if not name and paths:
            if isinstance(paths, list) and paths:
                center = len(paths) // 2
                path_str = paths[center]
            else:
                path_str = paths
            name = Path(path_str).stem
        if not name:
            name = f'sample_{self._counter}'
            self._counter += 1
        out_path = self.out_dir / f'{name}.json'
        # Ensure unique filename if collisions occur.
        suffix = 1
        while out_path.exists():
            out_path = self.out_dir / f'{name}_{suffix}.json'
            suffix += 1
        return out_path

    def _dump_sample(self,
                     out_path: Path,
                     keypoints: torch.Tensor,
                     valid_indices: torch.Tensor) -> None:
        keypoints_cart = spherical_to_cartesian(keypoints)
        people = []
        for idx in valid_indices.tolist():
            kp_cart = keypoints_cart[idx].detach().cpu()
            kp_sph = keypoints[idx].detach().cpu()
            kp3d = kp_cart
            kpsp = kp_sph
            bbox = self._bbox_from_points(kp_cart)
            people.append(
                dict(
                    keypoints_3d=self._to_list(kp3d),
                    keypoints_sp=self._to_list(kpsp),
                    bbox_3d=bbox))
        with out_path.open('w', encoding='utf-8') as handle:
            json.dump(people, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _bbox_from_points(points: torch.Tensor) -> List[float]:
        if points.numel() == 0:
            return [0.0] * 6
        mins, _ = points.min(dim=0)
        maxs, _ = points.max(dim=0)
        center = (maxs + mins) / 2
        size = maxs - mins
        return [float(center[0]), float(center[1]), float(center[2]),
                float(size[0]), float(size[1]), float(size[2])]

    @staticmethod
    def _to_list(tensor: torch.Tensor) -> List[List[float]]:
        return tensor.tolist()


@METRICS.register_module()
class MatchedPredictionDump(BaseMetric):
    """Dump matched predictions to JSON files per sample.
    
    This evaluator matches predictions to ground truth using Hungarian algorithm
    and outputs only the matched predictions along with their corresponding GT
    and per-joint errors.
    
    Args:
        out_dir: Output directory for JSON files
        use_ik_output: If True, use IK output; if False, use heatmap output
        include_gt: If True, also dump ground truth in the same format
        include_errors: If True, include per-joint errors in output
        prefix: Metric name prefix
    """

    def __init__(self,
                 out_dir: str,
                 use_ik_output: bool = False,
                 include_gt: bool = False,
                 include_errors: bool = False,
                 prefix: Optional[str] = None) -> None:
        super().__init__(prefix=prefix)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.use_ik_output = use_ik_output
        self.include_gt = include_gt
        self.include_errors = include_errors
        self._counter = 0

    def process(self, data_batch, data_samples: List) -> None:
        if not is_main_process():
            return
        
        logger = MMLogger.get_current_instance()
        
        for data_sample in data_samples:
            # Choose which predictions to use based on use_ik_output flag
            if self.use_ik_output:
                # Try to get IK predictions
                pred_keypoints = self._get_field(data_sample, 'pred_ik_keypoints_3d')
                if pred_keypoints is None:
                    pred_keypoints = self._get_field(data_sample, 'ik_keypoints')
                source_name = 'IK'
            else:
                # Get heatmap predictions
                pred_keypoints = self._get_field(data_sample, 'pred_keypoints_3d')
                if pred_keypoints is None:
                    pred_keypoints = self._get_field(data_sample, 'pred_keypoints')
                if pred_keypoints is None:
                    pred_keypoints = self._get_field(data_sample, 'keypoints')
                source_name = 'Heatmap'
            
            # Get ground truth
            gt_keypoints = self._get_field(data_sample, 'gt_keypoints_3d')
            
            if pred_keypoints is None:
                logger.debug(f'Missing {source_name} predictions')
                continue
            
            if gt_keypoints is None:
                logger.debug('Missing gt_keypoints_3d')
                continue
            
            pred_keypoints = pred_keypoints.detach().cpu()
            gt_keypoints = gt_keypoints.detach().cpu()
            
            # Sanitize inputs
            pred, pred_mask = self._sanitize_joints(pred_keypoints, treat_zero_as_invalid=False)
            gt, gt_mask = self._sanitize_joints(gt_keypoints, treat_zero_as_invalid=True)
            
            # Convert to Cartesian for matching and error computation
            pred_cart = spherical_to_cartesian(pred)
            gt_cart = spherical_to_cartesian(gt)
            
            # Match predictions to GT
            matches, cost_matrix = self._match_persons(pred_cart, gt_cart, pred_mask, gt_mask)
            
            # Resolve sample name
            sample_name = self._resolve_sample_name(data_sample)
            
            # Dump matched results in annotation format
            self._dump_matched_sample(
                sample_name, 
                pred, pred_cart, pred_mask,
                gt, gt_cart, gt_mask,
                matches, cost_matrix
            )

    def compute_metrics(self, results):
        return {}

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

    def _sanitize_joints(self,
                         joints: torch.Tensor,
                         treat_zero_as_invalid: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sanitize joint coordinates and create validity mask."""
        if joints.dim() == 2:
            joints = joints.unsqueeze(0)
        
        # Create mask for valid joints
        finite_mask = ~(torch.isnan(joints).any(dim=-1) | torch.isinf(joints).any(dim=-1))
        if treat_zero_as_invalid:
            non_zero_mask = joints.abs().sum(dim=-1) > 1e-8
            mask = finite_mask & non_zero_mask
        else:
            mask = finite_mask
        
        # Replace invalid values with zeros
        sanitized = torch.nan_to_num(joints, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Keep only first 3 dimensions (r, az, el)
        if sanitized.size(-1) > 3:
            sanitized = sanitized[..., :3]
        
        return sanitized, mask

    def _match_persons(self,
                       pred: torch.Tensor,
                       gt: torch.Tensor,
                       pred_mask: torch.Tensor,
                       gt_mask: torch.Tensor) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
        """
        Match predicted persons to GT persons using Hungarian algorithm.
        
        Returns:
            matches: List of (pred_idx, gt_idx) tuples
            cost_matrix: The computed cost matrix
        """
        num_pred = pred.size(0)
        num_gt = gt.size(0)
        
        # Find valid GT persons
        gt_person_valid = gt_mask.any(dim=1)
        valid_gt_indices = torch.where(gt_person_valid)[0].tolist()
        
        if len(valid_gt_indices) == 0 or num_pred == 0:
            return [], torch.zeros(num_pred, num_gt)
        
        num_valid_gt = len(valid_gt_indices)
        cost_matrix = torch.zeros(num_pred, num_valid_gt)
        
        # Compute cost matrix
        for pi in range(num_pred):
            for gi, gt_idx in enumerate(valid_gt_indices):
                joint_mask = gt_mask[gt_idx] & pred_mask[pi]
                if joint_mask.any():
                    diff = pred[pi, joint_mask] - gt[gt_idx, joint_mask]
                    cost_matrix[pi, gi] = torch.norm(diff, dim=-1).mean()
                else:
                    cost_matrix[pi, gi] = 1e6
        
        # Find optimal assignment using Hungarian algorithm
        assignment = self._find_optimal_assignment(cost_matrix, num_valid_gt)
        
        # Build matches list
        matches = []
        for gi, pi in enumerate(assignment):
            gt_idx = valid_gt_indices[gi]
            matches.append((pi, gt_idx))
        
        return matches, cost_matrix

    def _find_optimal_assignment(self, cost_matrix: torch.Tensor, num_gt: int) -> List[int]:
        """Find optimal assignment using Hungarian algorithm."""
        num_pred = cost_matrix.size(0)
        
        if num_gt == 0:
            return []
        
        if HAS_SCIPY and num_gt > 2:
            cost_np = cost_matrix.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            
            assignment = [-1] * num_gt
            for pred_idx, gt_idx in zip(row_ind, col_ind):
                if gt_idx < num_gt:
                    assignment[gt_idx] = int(pred_idx)
            
            # Fill unassigned
            for gi in range(num_gt):
                if assignment[gi] == -1:
                    used = set(assignment) - {-1}
                    best_pred = None
                    best_cost = float('inf')
                    for p in range(num_pred):
                        if p not in used and cost_matrix[p, gi] < best_cost:
                            best_cost = cost_matrix[p, gi]
                            best_pred = p
                    assignment[gi] = best_pred if best_pred is not None else 0
            
            return assignment
        
        # Fallback for small cases
        if num_gt == 1:
            return [cost_matrix[:, 0].argmin().item()]
        
        if num_gt == 2 and num_pred >= 2:
            min_cost = float('inf')
            best_perm = None
            for perm in permutations(range(num_pred), num_gt):
                cost = sum(cost_matrix[perm[i], i] for i in range(num_gt))
                if cost < min_cost:
                    min_cost = cost
                    best_perm = list(perm)
            return best_perm if best_perm else list(range(min(num_pred, num_gt)))
        
        # Greedy for larger cases without scipy
        used_preds = set()
        assignment = []
        for gi in range(num_gt):
            best_pred = None
            best_cost = float('inf')
            for pi in range(num_pred):
                if pi in used_preds:
                    continue
                if cost_matrix[pi, gi] < best_cost:
                    best_cost = cost_matrix[pi, gi]
                    best_pred = pi
            if best_pred is not None:
                assignment.append(best_pred)
                used_preds.add(best_pred)
        return assignment

    def _resolve_sample_name(self, data_sample) -> Path:
        metainfo = getattr(data_sample, 'metainfo', {})
        name = getattr(data_sample, 'dump_name', None)
        if not name:
            try:
                name = data_sample['sample_name']
            except Exception:
                name = None
        if not name and hasattr(metainfo, 'get'):
            name = metainfo.get('sample_name')
        if name:
            name = str(name)
        paths = metainfo.get('radar_points_paths', None)
        if not name and paths:
            if isinstance(paths, list) and paths:
                center = len(paths) // 2
                path_str = paths[center]
            else:
                path_str = paths
            name = Path(path_str).stem
        if not name:
            name = f'sample_{self._counter}'
            self._counter += 1
        out_path = self.out_dir / f'{name}.json'
        return out_path

    def _dump_matched_sample(self,
                             out_path: Path,
                             pred_sph: torch.Tensor,
                             pred_cart: torch.Tensor,
                             pred_mask: torch.Tensor,
                             gt_sph: torch.Tensor,
                             gt_cart: torch.Tensor,
                             gt_mask: torch.Tensor,
                             matches: List[Tuple[int, int]],
                             cost_matrix: torch.Tensor) -> None:
        """Dump matched predictions in annotation format.
        
        Output format matches annotation structure:
        [
            {
                "keypoints_3d": [[x, y, z, 1.0], ...],  # 16 joints in Cartesian
                "keypoints_sp": [[r, az, el, 1.0], ...],  # 16 joints in spherical
                "bbox_3d": [cx, cy, cz, sx, sy, sz],
                "gt_keypoints_3d": [...],  # optional
                "gt_keypoints_sp": [...],  # optional
                "errors": [e0, e1, ..., e15]  # optional per-joint L2 errors
            },
            ...
        ]
        """
        people = []
        for pred_idx, gt_idx in matches:
            # Prediction keypoints
            kp_cart = pred_cart[pred_idx].detach().cpu()
            kp_sph = pred_sph[pred_idx].detach().cpu()
            
            # Format as annotation: add confidence dimension
            # Annotation format: (num_joints, 4) where last dim is confidence (1.0)
            kp3d_with_conf = torch.cat([
                kp_cart,
                torch.ones(kp_cart.size(0), 1)
            ], dim=-1).tolist()
            
            kpsp_with_conf = torch.cat([
                kp_sph,
                torch.ones(kp_sph.size(0), 1)
            ], dim=-1).tolist()
            
            bbox = self._bbox_from_points(kp_cart)
            
            person_dict = {
                "keypoints_3d": kp3d_with_conf,
                "keypoints_sp": kpsp_with_conf,
                "bbox_3d": bbox
            }
            
            # Optionally include ground truth
            if self.include_gt:
                gt_cart_matched = gt_cart[gt_idx].detach().cpu()
                gt_sph_matched = gt_sph[gt_idx].detach().cpu()
                
                gt_kp3d = torch.cat([
                    gt_cart_matched,
                    torch.ones(gt_cart_matched.size(0), 1)
                ], dim=-1).tolist()
                
                gt_kpsp = torch.cat([
                    gt_sph_matched,
                    torch.ones(gt_sph_matched.size(0), 1)
                ], dim=-1).tolist()
                
                person_dict["gt_keypoints_3d"] = gt_kp3d
                person_dict["gt_keypoints_sp"] = gt_kpsp
            
            # Optionally include per-joint errors
            if self.include_errors:
                gt_cart_matched = gt_cart[gt_idx].detach().cpu()
                joint_mask = gt_mask[gt_idx].detach().cpu()
                
                # Compute L2 error for each joint
                errors = []
                for j in range(kp_cart.size(0)):
                    if joint_mask[j]:
                        diff = kp_cart[j] - gt_cart_matched[j]
                        error = torch.norm(diff).item()
                    else:
                        error = -1.0  # Invalid joint
                    errors.append(error)
                
                person_dict["errors"] = errors
                
                # Also add summary statistics
                valid_errors = [e for e in errors if e >= 0]
                if valid_errors:
                    person_dict["mean_error"] = float(np.mean(valid_errors))
                    person_dict["max_error"] = float(np.max(valid_errors))
            
            people.append(person_dict)
        
        with out_path.open('w', encoding='utf-8') as handle:
            json.dump(people, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _bbox_from_points(points: torch.Tensor) -> List[float]:
        if points.numel() == 0:
            return [0.0] * 6
        mins, _ = points.min(dim=0)
        maxs, _ = points.max(dim=0)
        center = (maxs + mins) / 2
        size = maxs - mins
        return [float(center[0]), float(center[1]), float(center[2]),
                float(size[0]), float(size[1]), float(size[2])]
