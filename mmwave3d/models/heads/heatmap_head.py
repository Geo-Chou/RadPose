"""
2.5D Heatmap Head for mmWave pose estimation.

This head outputs:
- RA (Range-Azimuth) heatmaps for joint localization confidence
- Elevation maps for height prediction at each RA location
"""

from typing import Dict, List, Optional, Sequence, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmwave3d.registry import MODELS
from mmwave3d.models.losses.heatmap_losses import compute_heatmap_cost_matrix, find_best_assignment
from mmengine.logging import MMLogger
from mmwave3d.utils import spherical_to_cartesian


@MODELS.register_module()
class MmWaveHeatmapHead(nn.Module):
    """
    2.5D Heatmap Head that predicts:
    1. RA heatmaps: confidence maps on range-azimuth plane for each joint
    2. Elevation maps: elevation values at each RA location
    
    For multi-person support (max 2 people), outputs separate heatmaps per person.
    
    Args:
        in_dim: Input feature dimension
        num_person: Maximum number of persons (default: 2)
        num_joints: Number of keypoints per person (default: 17)
        mlp_hidden_dim: Hidden dimension for MLP
        mlp_depth: Depth of MLP layers
        range_bins: Number of range bins in heatmap
        angle_bins: Number of azimuth bins in heatmap
        point_cloud_range: [r_min, az_min, el_min, r_max, az_max, el_max]
        sigma_r: Gaussian sigma for range dimension (in meters)
        sigma_a: Gaussian sigma for azimuth dimension (in radians)
        loss_heatmap: Heatmap loss configuration
        loss_elevation: Elevation loss configuration
    """

    def __init__(self,
                 in_dim: int,
                 num_person: int = 2,
                 num_joints: int = 17,
                 mlp_hidden_dim: int = 512,
                 mlp_depth: int = 2,
                 range_bins: int = 50,
                 angle_bins: int = 180,
                 point_cloud_range: Sequence[float] = [0.12, -0.524, -0.524, 6.0, 2.618, 0.524],
                 sigma_r: float = 0.15,
                 sigma_a: float = 0.087,  # ~5 degrees
                 loss_heatmap: Optional[Dict] = None,
                 loss_elevation: Optional[Dict] = None,
                 loss_skeleton_angle: Optional[Dict] = None,
                 loss_skeleton_length: Optional[Dict] = None,
                 coord_loss_weight: float = 10.0,
                 coord_loss_beta: float = 0.1,
                 decode_inference_with_argmax: bool = True,
                 predict_uncertainty: bool = False,
                 log_sigma_min: float = -7.0,
                 log_sigma_max: float = 7.0) -> None:
        super().__init__()
        
        self.num_person = num_person
        self.num_joints = num_joints
        self.range_bins = range_bins
        self.angle_bins = angle_bins
        self.sigma_r = sigma_r
        self.sigma_a = sigma_a
        self.coord_loss_weight = coord_loss_weight
        self.coord_loss_beta = coord_loss_beta
        self.decode_inference_with_argmax = decode_inference_with_argmax
        self.predict_uncertainty = predict_uncertainty
        self.log_sigma_min = float(log_sigma_min)
        self.log_sigma_max = float(log_sigma_max)
        
        # Parse point cloud range
        self.r_min, self.az_min, self.el_min = point_cloud_range[:3]
        self.r_max, self.az_max, self.el_max = point_cloud_range[3:]
        
        # Create coordinate grids
        r_coords = torch.linspace(self.r_min, self.r_max, range_bins)
        az_coords = torch.linspace(self.az_min, self.az_max, angle_bins)
        self.register_buffer('r_coords', r_coords, persistent=False)
        self.register_buffer('az_coords', az_coords, persistent=False)
        
        # Build MLP backbone
        mlp_layers = []
        dim = in_dim
        for _ in range(mlp_depth):
            mlp_layers += [nn.Linear(dim, mlp_hidden_dim), nn.ReLU(inplace=True)]
            dim = mlp_hidden_dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Heatmap prediction heads (per person, per joint)
        # Output: (B, num_person * num_joints, range_bins, angle_bins)
        self.heatmap_head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_person * num_joints * range_bins * angle_bins)
        )
        
        # Elevation prediction heads (per person, per joint)
        # Output: (B, num_person * num_joints, range_bins, angle_bins)
        self.elevation_head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_person * num_joints * range_bins * angle_bins)
        )

        self.uncertainty_head = None
        if self.predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_person * num_joints)
            )
        
        # Initialize weights
        self._init_weights()
        
        # Build losses
        if loss_heatmap is None:
            loss_heatmap = dict(type='Heatmap2DLoss', loss_weight=1.0)
        self.loss_heatmap_fn = MODELS.build(loss_heatmap)
        
        if loss_elevation is None:
            loss_elevation = dict(type='ElevationLoss', loss_weight=1.0)
        self.loss_elevation_fn = MODELS.build(loss_elevation)

        self.loss_skeleton_angle_fn = (
            MODELS.build(loss_skeleton_angle) if loss_skeleton_angle else None
        )
        self.loss_skeleton_length_fn = (
            MODELS.build(loss_skeleton_length) if loss_skeleton_length else None
        )
        
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, 
                inputs: Dict[str, torch.Tensor],
                data_samples: Optional[List] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            inputs: Dictionary with 'features' key - (B, N, C) features from backbone
            data_samples: List of data samples with GT (optional, used in training)
            
        Returns:
            Dictionary containing:
            - 'heatmaps': (B, num_person, num_joints, range_bins, angle_bins)
            - 'elevation_maps': (B, num_person, num_joints, range_bins, angle_bins)
            - 'keypoints': (B, num_person, num_joints, 3) - decoded 3D positions
            - (if training with GT) loss_heatmap, loss_elevation
        """
        # Extract and pool features
        feats = inputs['features']  # (B, N, C)
        pooled = feats.mean(dim=1)  # (B, C)
        B = pooled.size(0)
        
        # Verify batch size consistency
        if data_samples is not None and len(data_samples) != B:
            raise RuntimeError(
                f"Batch size mismatch: features has {B} samples but "
                f"data_samples has {len(data_samples)} samples. "
                f"This indicates a data pipeline issue."
            )
        
        # Shared MLP
        x = self.mlp(pooled)
        
        # Predict heatmaps
        heatmap_flat = self.heatmap_head(x)  # (B, num_person*num_joints*H*W)
        heatmaps = heatmap_flat.view(B, self.num_person, self.num_joints, 
                                     self.range_bins, self.angle_bins)
        heatmaps = torch.sigmoid(heatmaps)  # Normalize to [0, 1]

        joint_confidence = self._compute_joint_confidence(heatmaps)
        
        # Predict elevation maps
        elev_flat = self.elevation_head(x)  # (B, num_person*num_joints*H*W)
        elevation_maps = elev_flat.view(B, self.num_person, self.num_joints,
                                       self.range_bins, self.angle_bins)
        # Normalize elevation to [-1, 1] range for stability
        elevation_maps = torch.tanh(elevation_maps)
        
        # Decode keypoints from heatmaps
        keypoints = self.decode_keypoints(
            heatmaps,
            elevation_maps,
            use_argmax=not self.training and self.decode_inference_with_argmax)

        joint_log_sigma = None
        if self.uncertainty_head is not None:
            joint_log_sigma = self.uncertainty_head(x)
            joint_log_sigma = joint_log_sigma.view(B, self.num_person, self.num_joints)
            joint_log_sigma = joint_log_sigma.clamp(self.log_sigma_min, self.log_sigma_max)
        
        outputs = {
            'heatmaps': heatmaps,
            'elevation_maps': elevation_maps,
            'keypoints': keypoints,
            'joint_confidence': joint_confidence
        }
        if joint_log_sigma is not None:
            outputs['joint_log_sigma'] = joint_log_sigma
            outputs['joint_sigma'] = torch.exp(joint_log_sigma)
        
        # Compute loss if training with ground truth
        if data_samples is not None:
            gt_kpts, mask = self._stack_gt(data_samples, keypoints.device)
            if gt_kpts is not None:
                loss_dict = self.loss(outputs, gt_kpts, mask)
                outputs.update(loss_dict)
        
        return outputs

    @staticmethod
    def _compute_joint_confidence(heatmaps: torch.Tensor,
                                  eps: float = 1e-8) -> torch.Tensor:
        """Compute per-joint confidence from heatmap entropy."""
        B, P, J, H, W = heatmaps.shape
        heatmaps_flat = heatmaps.view(B, P, J, H * W)
        heatmap_sum = heatmaps_flat.sum(dim=-1, keepdim=True)
        prob = heatmaps_flat / (heatmap_sum + eps)
        entropy = -(prob * (prob + eps).log()).sum(dim=-1)
        entropy = entropy / math.log(H * W)
        confidence = 1.0 - entropy
        confidence = torch.where(heatmap_sum.squeeze(-1) > eps,
                                 confidence,
                                 confidence.new_zeros(()))
        return confidence.clamp(0.0, 1.0)
    
    def decode_keypoints(self,
                        heatmaps: torch.Tensor,
                        elevation_maps: torch.Tensor,
                        use_argmax: bool = False) -> torch.Tensor:
        """
        Decode 3D keypoints from heatmaps using soft-argmax.
        
        Args:
            heatmaps: (B, num_person, num_joints, H, W)
            elevation_maps: (B, num_person, num_joints, H, W)
            use_argmax: Whether to use argmax instead of soft-argmax
            
        Returns:
            keypoints: (B, num_person, num_joints, 3) in spherical coords (r, az, el)
        """
        B, P, J, H, W = heatmaps.shape
        device = heatmaps.device
        
        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.view(B, P, J, H * W)  # (B, P, J, H*W)
        elevation_flat = elevation_maps.view(B, P, J, H * W)
        
        # Soft-argmax for range (weighted average)
        # r_coords: (H,) -> expand to (1, 1, 1, H, 1)
        r_grid = self.r_coords.view(1, 1, 1, H, 1).expand(B, P, J, H, W)
        r_grid_flat = r_grid.reshape(B, P, J, H * W)
        
        if use_argmax:
            # Pick the peak index to avoid averaging multiple modes
            peak_idx = heatmaps_flat.argmax(dim=-1, keepdim=True)
            r_pred = torch.gather(r_grid_flat, dim=-1, index=peak_idx)
        else:
            # Soft-argmax for range (weighted average)
            heatmap_sum = heatmaps_flat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            r_pred = (r_grid_flat * heatmaps_flat).sum(dim=-1, keepdim=True) / heatmap_sum
        
        # Soft-argmax for azimuth (circular mean with atan2)
        # az_coords: (W,) -> expand to (1, 1, 1, 1, W)
        az_grid = self.az_coords.view(1, 1, 1, 1, W).expand(B, P, J, H, W)
        az_grid_flat = az_grid.reshape(B, P, J, H * W)
        
        if use_argmax:
            az_pred = torch.gather(az_grid_flat, dim=-1, index=peak_idx)
            # Elevation: sample at the peak location
            el_pred = torch.gather(elevation_flat, dim=-1, index=peak_idx)
        else:
            # Circular weighted average using sin/cos
            sin_weighted = (torch.sin(az_grid_flat) * heatmaps_flat).sum(dim=-1)
            cos_weighted = (torch.cos(az_grid_flat) * heatmaps_flat).sum(dim=-1)
            az_pred = torch.atan2(sin_weighted, cos_weighted).unsqueeze(-1)
            # Elevation: weighted average of elevation map
            el_pred = (elevation_flat * heatmaps_flat).sum(dim=-1, keepdim=True) / heatmap_sum
        
        # Denormalize elevation from [-1, 1] to [el_min, el_max]
        el_pred = el_pred * (self.el_max - self.el_min) / 2.0 + (self.el_max + self.el_min) / 2.0
        
        # Concatenate to form (B, P, J, 3)
        keypoints = torch.cat([r_pred, az_pred, el_pred], dim=-1)
        
        return keypoints
    
    def _stack_gt(self, data_samples: List, device: torch.device):
        """
        Stack ground truth keypoints from data samples.
        
        Args:
            data_samples: List of data samples
            device: Target device
            
        Returns:
            gt_keypoints: (B, num_person, num_joints, 3) or None
            mask: (B, num_person, num_joints) or None
        """
        if not data_samples:
            return None, None
        
        gt_list = []
        mask_list = []
        
        for idx, sample in enumerate(data_samples):
            gt_kpts = sample.get('gt_keypoints_3d', None)
            if gt_kpts is None:
                return None, None
            
            # Ensure shape is (num_person, num_joints, 3)
            if gt_kpts.dim() == 2:
                gt_kpts = gt_kpts.unsqueeze(0)  # (1, num_joints, 3)
            
            # Extract only the first 3 dimensions (r, az, el) if more dimensions exist
            if gt_kpts.size(-1) > 3:
                gt_kpts = gt_kpts[..., :3]
            
            # Pad or crop to num_person
            current_num_person = gt_kpts.size(0)
            num_joints = gt_kpts.size(1)
            
            if current_num_person < self.num_person:
                # Need to pad
                pad_size = self.num_person - current_num_person
                pad_kpts = torch.zeros(pad_size, num_joints, 3, 
                                      dtype=gt_kpts.dtype, device=gt_kpts.device)
                gt_kpts = torch.cat([gt_kpts, pad_kpts], dim=0)
            elif current_num_person > self.num_person:
                # Need to crop
                gt_kpts = gt_kpts[:self.num_person]
            
            # Create mask (True for valid joints)
            finite_mask = ~(torch.isnan(gt_kpts).any(dim=-1) | torch.isinf(gt_kpts).any(dim=-1))
            non_zero_mask = gt_kpts.abs().sum(dim=-1) > 1e-8
            mask = finite_mask & non_zero_mask
            
            # Mark padded persons as invalid
            if current_num_person < self.num_person:
                mask[current_num_person:] = False
            
            gt_list.append(gt_kpts)
            mask_list.append(mask)
        
        gt_keypoints = torch.stack(gt_list).to(device)  # (B, P, J, 3)
        mask = torch.stack(mask_list).to(device)  # (B, P, J)
        
        return gt_keypoints, mask
    
    def generate_target_heatmaps(self,
                                 gt_keypoints: torch.Tensor,
                                 gt_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate ground truth heatmaps and elevation maps from 3D keypoints.
        
        Args:
            gt_keypoints: (B, num_person, num_joints, 3) in spherical coords
            gt_masks: (B, num_person, num_joints) - valid joint mask
            
        Returns:
            target_heatmaps: (B, num_person, num_joints, H, W)
            target_elevations: (B, num_person, num_joints, H, W)
        """
        B, P, J, _ = gt_keypoints.shape
        H, W = self.range_bins, self.angle_bins
        device = gt_keypoints.device
        
        target_heatmaps = torch.zeros(B, P, J, H, W, device=device)
        target_elevations = torch.zeros(B, P, J, H, W, device=device)
        
        # Create meshgrid for RA coordinates
        r_grid = self.r_coords.view(H, 1).expand(H, W)  # (H, W)
        az_grid = self.az_coords.view(1, W).expand(H, W)  # (H, W)
        
        for b in range(B):
            for p in range(P):
                for j in range(J):
                    if not gt_masks[b, p, j]:
                        continue
                    
                    r_gt, az_gt, el_gt = gt_keypoints[b, p, j]
                    
                    # Compute Gaussian heatmap
                    # Handle angle wrapping for azimuth
                    delta_az = az_grid - az_gt
                    # Wrap to [-pi, pi]
                    delta_az = torch.atan2(torch.sin(delta_az), torch.cos(delta_az))
                    
                    delta_r = r_grid - r_gt
                    
                    # 2D Gaussian
                    gauss = torch.exp(
                        -(delta_r ** 2) / (2 * self.sigma_r ** 2)
                        -(delta_az ** 2) / (2 * self.sigma_a ** 2)
                    )
                    
                    # Only keep reasonably confident pixels; zero-out low tails
                    mask = gauss > 0.01
                    target_heatmaps[b, p, j] = gauss * mask
                    
                    # Elevation: normalize to [-1, 1]
                    el_normalized = (el_gt - (self.el_max + self.el_min) / 2.0) / ((self.el_max - self.el_min) / 2.0)
                    target_elevations[b, p, j, mask] = el_normalized
        
        return target_heatmaps, target_elevations
    
    def loss(self,
             pred_dict: Dict[str, torch.Tensor],
             gt_keypoints: torch.Tensor,
             gt_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with unified matching.
        
        Key: Perform matching ONCE, apply to all losses (heatmap, elevation, coord).
        This ensures consistency and training stability.
        
        Args:
            pred_dict: Dictionary with 'heatmaps', 'elevation_maps', 'keypoints'
            gt_keypoints: (B, num_person, num_joints, 3)
            gt_masks: (B, num_person, num_joints)
            
        Returns:
            Dictionary of losses
        """
        # Generate target heatmaps
        target_heatmaps, target_elevations = self.generate_target_heatmaps(
            gt_keypoints, gt_masks)
        
        # ========================================================================
        # UNIFIED MATCHING: Compute assignments once for all losses
        # ========================================================================
        assignments = self._compute_assignments(
            pred_dict['keypoints'], gt_keypoints, gt_masks)
        
        # Apply unified matching to all losses
        loss_heatmap = self._compute_matched_heatmap_loss(
            pred_dict['heatmaps'], target_heatmaps, gt_masks, assignments)
        
        loss_elevation = self._compute_matched_elevation_loss(
            pred_dict['elevation_maps'], target_elevations, 
            target_heatmaps, gt_masks, assignments)
        
        loss_coord = self._compute_matched_coord_loss(
            pred_dict['keypoints'], gt_keypoints, gt_masks, assignments)
        
        losses = {
            'loss_heatmap': loss_heatmap,
            'loss_elevation': loss_elevation,
            'loss_coord': loss_coord * self.coord_loss_weight
        }
        
        # Skeleton losses (also use matched assignments)
        if self.loss_skeleton_angle_fn is not None:
            losses['loss_skeleton_angle'] = self._compute_matched_skeleton_loss(
                pred_dict['keypoints'], gt_keypoints, gt_masks, assignments, 'angle')
        
        if self.loss_skeleton_length_fn is not None:
            losses['loss_skeleton_length'] = self._compute_matched_skeleton_loss(
                pred_dict['keypoints'], gt_keypoints, gt_masks, assignments, 'length')
        
        return losses
    
    def _compute_assignments(self, pred_keypoints: torch.Tensor,
                            gt_keypoints: torch.Tensor,
                            gt_masks: torch.Tensor) -> List[List]:
        """
        Compute unified pred-to-gt assignments using keypoint distance.
        
        Returns:
            List of assignments per sample. Each assignment[b] is list of (pred_idx, gt_idx) tuples.
        """
        B, P = pred_keypoints.size(0), pred_keypoints.size(1)
        assignments = []
        
        for b in range(B):
            person_valid = gt_masks[b].any(dim=1)
            valid_gt_indices = torch.where(person_valid)[0].tolist()
            num_gt = len(valid_gt_indices)
            
            if num_gt == 0:
                assignments.append([])
                continue
            
            # Compute cost matrix using keypoint distance
            cost_matrix = self._compute_kpt_cost_matrix(
                pred_keypoints[b], gt_keypoints[b], gt_masks[b], valid_gt_indices)
            
            # Hungarian matching
            assignment = find_best_assignment(cost_matrix, num_gt)
            
            # Convert to (pred_idx, gt_idx) tuples
            assignment_pairs = []
            for gt_local_idx, pred_idx in enumerate(assignment):
                gt_global_idx = valid_gt_indices[gt_local_idx]
                assignment_pairs.append((pred_idx, gt_global_idx))
            
            assignments.append(assignment_pairs)
        
        return assignments
    
    def _compute_matched_heatmap_loss(self, pred_heatmaps: torch.Tensor,
                                     target_heatmaps: torch.Tensor,
                                     gt_masks: torch.Tensor,
                                     assignments: List[List]) -> torch.Tensor:
        """Compute heatmap loss using pre-computed assignments."""
        B, P = pred_heatmaps.size(0), pred_heatmaps.size(1)
        device = pred_heatmaps.device
        
        total_loss = torch.tensor(0.0, device=device)
        valid_samples = 0
        
        for b in range(B):
            if not assignments[b]:
                # No GT: penalize all predictions
                loss = self.loss_heatmap_fn.focal_loss(
                    pred_heatmaps[b], torch.zeros_like(target_heatmaps[b]))
                total_loss += loss
                valid_samples += 1
                continue
            
            sample_loss = torch.tensor(0.0, device=device)
            matched_preds = set()
            
            # Matched pairs
            for pred_idx, gt_idx in assignments[b]:
                matched_preds.add(pred_idx)
                loss = self.loss_heatmap_fn.focal_loss(
                    pred_heatmaps[b:b+1, pred_idx:pred_idx+1],
                    target_heatmaps[b:b+1, gt_idx:gt_idx+1])
                sample_loss += loss
            
            # Unmatched predictions
            unmatched = set(range(P)) - matched_preds
            if unmatched:
                unmatched_idx = torch.tensor(list(unmatched), device=device)
                loss_fp = self.loss_heatmap_fn.focal_loss(
                    pred_heatmaps[b:b+1, unmatched_idx],
                    torch.zeros_like(pred_heatmaps[b:b+1, unmatched_idx]))
                sample_loss += loss_fp * 0.5
            
            total_loss += sample_loss
            valid_samples += 1
        
        loss = (total_loss / valid_samples) if valid_samples > 0 else total_loss
        return loss * self.loss_heatmap_fn.loss_weight
    
    def _compute_matched_elevation_loss(self, pred_elevation: torch.Tensor,
                                       target_elevation: torch.Tensor,
                                       heatmap_weight: torch.Tensor,
                                       gt_masks: torch.Tensor,
                                       assignments: List[List]) -> torch.Tensor:
        """Compute elevation loss using pre-computed assignments."""
        B, P, J, H, W = pred_elevation.shape
        device = pred_elevation.device
        threshold = self.loss_elevation_fn.threshold
        
        total_loss = torch.tensor(0.0, device=device)
        total_weight = torch.tensor(0.0, device=device)
        
        for b in range(B):
            if not assignments[b]:
                continue
            
            for pred_idx, gt_idx in assignments[b]:
                weight_mask = (heatmap_weight[b, gt_idx] > threshold).float()
                joint_mask = gt_masks[b, gt_idx].float().view(J, 1, 1).expand(J, H, W)
                weight_mask = weight_mask * joint_mask
                
                diff = torch.abs(pred_elevation[b, pred_idx] - target_elevation[b, gt_idx])
                weighted_loss = diff * weight_mask
                
                total_loss += weighted_loss.sum()
                total_weight += weight_mask.sum()
        
        loss = total_loss / total_weight.clamp(min=1.0)
        return loss * self.loss_elevation_fn.loss_weight
    
    def _compute_matched_coord_loss(self, pred_keypoints: torch.Tensor,
                                   gt_keypoints: torch.Tensor,
                                   gt_masks: torch.Tensor,
                                   assignments: List[List]) -> torch.Tensor:
        """Compute coordinate loss using pre-computed assignments."""
        B = pred_keypoints.size(0)
        device = pred_keypoints.device
        
        pred_cart = spherical_to_cartesian(pred_keypoints)
        gt_cart = spherical_to_cartesian(gt_keypoints)
        
        total_loss = torch.tensor(0.0, device=device)
        valid_count = 0
        
        for b in range(B):
            if not assignments[b]:
                continue
            
            for pred_idx, gt_idx in assignments[b]:
                joint_mask = gt_masks[b, gt_idx]
                if joint_mask.any():
                    diff = pred_cart[b, pred_idx, joint_mask] - gt_cart[b, gt_idx, joint_mask]
                    loss = F.smooth_l1_loss(diff, torch.zeros_like(diff),
                                          beta=self.coord_loss_beta, reduction='mean')
                    total_loss += loss
                    valid_count += 1
        
        if valid_count == 0:
            return pred_keypoints.new_tensor(0.0)
        
        return total_loss / valid_count
    
    def _compute_matched_skeleton_loss(self, pred_keypoints: torch.Tensor,
                                      gt_keypoints: torch.Tensor,
                                      gt_masks: torch.Tensor,
                                      assignments: List[List],
                                      loss_type: str) -> torch.Tensor:
        """Compute skeleton loss using pre-computed assignments."""
        matched_preds = []
        matched_gts = []
        matched_masks = []
        
        for b in range(len(assignments)):
            for pred_idx, gt_idx in assignments[b]:
                matched_preds.append(pred_keypoints[b, pred_idx])
                matched_gts.append(gt_keypoints[b, gt_idx])
                matched_masks.append(gt_masks[b, gt_idx])
        
        if not matched_preds:
            return pred_keypoints.new_tensor(0.0)
        
        matched_preds = torch.stack(matched_preds).unsqueeze(1)
        matched_gts = torch.stack(matched_gts).unsqueeze(1)
        matched_masks = torch.stack(matched_masks).unsqueeze(1)
        
        if loss_type == 'angle':
            return self.loss_skeleton_angle_fn(matched_preds, matched_gts, matched_masks)
        else:
            return self.loss_skeleton_length_fn(matched_preds, matched_gts, matched_masks)

    def _compute_kpt_cost_matrix(self,
                                 pred_kpts: torch.Tensor,
                                 gt_kpts: torch.Tensor,
                                 gt_masks: torch.Tensor,
                                 valid_gt_indices: List[int]) -> torch.Tensor:
        """
        Compute matching cost matrix using decoded keypoints in XZ plane (meters).
        """
        P, J, _ = pred_kpts.shape
        num_gt = len(valid_gt_indices)
        device = pred_kpts.device

        pred_xz = self._spherical_to_xz(pred_kpts)
        gt_xz = self._spherical_to_xz(gt_kpts)

        cost_matrix = torch.zeros(P, num_gt, device=device)
        for pi in range(P):
            for gi, gt_idx in enumerate(valid_gt_indices):
                joint_mask = gt_masks[gt_idx]
                if joint_mask.any():
                    diff = pred_xz[pi, joint_mask] - gt_xz[gt_idx, joint_mask]
                    cost_matrix[pi, gi] = torch.norm(diff, dim=-1).mean()
                else:
                    cost_matrix[pi, gi] = 1e6
        return cost_matrix

    def _spherical_to_xz(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Convert spherical (r, az, el) to XZ plane (meters), ignoring elevation.
        """
        coords = torch.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        r = coords[..., 0]
        az = coords[..., 1]
        x = r * torch.cos(az)
        z = r * torch.sin(az)
        return torch.stack((x, z), dim=-1)


@MODELS.register_module()
class MmWaveHeatmapSingleHead(MmWaveHeatmapHead):
    """Heatmap head for single-person scenes without matching."""

    def loss(self,
             pred_dict: Dict[str, torch.Tensor],
             gt_keypoints: torch.Tensor,
             gt_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute total loss without person matching.
        """
        target_heatmaps, target_elevations = self.generate_target_heatmaps(
            gt_keypoints, gt_masks)

        loss_heatmap = self.loss_heatmap_fn(
            pred_dict['heatmaps'],
            target_heatmaps,
            gt_masks
        )

        loss_elevation = self.loss_elevation_fn(
            pred_dict['elevation_maps'],
            target_elevations,
            pred_dict['heatmaps'],
            gt_masks
        )

        pred_cart = spherical_to_cartesian(pred_dict['keypoints'])
        gt_cart = spherical_to_cartesian(gt_keypoints)
        kpt_diff = F.smooth_l1_loss(
            pred_cart, gt_cart, beta=self.coord_loss_beta, reduction='none')
        kpt_mask = gt_masks.unsqueeze(-1).float().expand_as(kpt_diff)
        num_valid_kpts = kpt_mask.sum().clamp(min=1.0)
        loss_coord = (kpt_diff * kpt_mask).sum() / num_valid_kpts

        self._maybe_log_stats(gt_masks)

        losses = {
            'loss_heatmap': loss_heatmap,
            'loss_elevation': loss_elevation,
            'loss_coord': loss_coord * self.coord_loss_weight
        }
        if self.loss_skeleton_angle_fn is not None:
            losses['loss_skeleton_angle'] = self.loss_skeleton_angle_fn(
                pred_dict['keypoints'], gt_keypoints, gt_masks)
        if self.loss_skeleton_length_fn is not None:
            losses['loss_skeleton_length'] = self.loss_skeleton_length_fn(
                pred_dict['keypoints'], gt_keypoints, gt_masks)
        return losses
