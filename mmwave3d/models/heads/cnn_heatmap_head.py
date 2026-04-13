"""
CNN-based Heatmap Head with Decoder Architecture for mmWave pose estimation.

This head uses transposed convolutions to decode heatmaps from global features,
avoiding the information bottleneck of pure MLP-based heads.
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
class CNNHeatmapHead(nn.Module):
    """
    CNN-based 2.5D Heatmap Head with decoder architecture.
    
    Architecture:
    1. Global feature → Linear projection → Reshape to spatial map
    2. Progressive upsampling with ConvTranspose2d
    3. Final conv layers for heatmap and elevation prediction
    
    This design addresses the information bottleneck problem by using
    spatial convolutions that naturally preserve and generate 2D structure.
    
    Args:
        in_dim: Input feature dimension (from backbone pooling)
        num_person: Maximum number of persons (default: 2)
        num_joints: Number of keypoints per person (default: 17)
        num_frames: Number of input frames (default: 5)
        initial_spatial_size: Initial spatial size after linear projection (default: 4)
        decoder_channels: Channel progression in decoder (default: [256, 128, 64])
        range_bins: Number of range bins in final heatmap
        angle_bins: Number of azimuth bins in final heatmap
        point_cloud_range: [r_min, az_min, el_min, r_max, az_max, el_max]
        sigma_r: Gaussian sigma for range dimension (in meters)
        sigma_a: Gaussian sigma for azimuth dimension (in radians)
    """

    def __init__(self,
                 in_dim: int,
                 num_person: int = 2,
                 num_joints: int = 17,
                 num_frames: int = 5,
                 use_matching: bool = True,
                 initial_spatial_size: int = 4,
                 decoder_channels: Sequence[int] = (256, 128, 64),
                 range_bins: int = 60,
                 angle_bins: int = 90,
                 point_cloud_range: Sequence[float] = [0.12, -0.524, -0.524, 6.0, 2.618, 0.524],
                 sigma_r: float = 0.15,
                 sigma_a: float = 0.087,
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
        self.num_frames = num_frames
        self.range_bins = range_bins
        self.angle_bins = angle_bins
        self.sigma_r = sigma_r
        self.sigma_a = sigma_a
        self.coord_loss_weight = coord_loss_weight
        self.coord_loss_beta = coord_loss_beta
        self.decode_inference_with_argmax = decode_inference_with_argmax
        self.initial_spatial_size = initial_spatial_size
        self.use_matching = use_matching
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
        
        # Initial projection: global feature → spatial feature map
        initial_channels = decoder_channels[0]
        self.initial_channels = initial_channels
        spatial_dim = initial_spatial_size * initial_spatial_size * initial_channels
        self.initial_proj = nn.Sequential(
            nn.Linear(in_dim, spatial_dim),
            nn.ReLU(inplace=True)
        )
        
        # Build CNN decoder with progressive upsampling
        self.decoder_layers = nn.ModuleList()
        in_ch = initial_channels

        # If spatial inputs use a different channel size than decoder start,
        # project them to match the decoder expectation.
        self.spatial_proj = None
        if in_dim != initial_channels:
            self.spatial_proj = nn.Conv2d(in_dim, initial_channels, kernel_size=1)
        
        for out_ch in decoder_channels[1:]:
            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
            in_ch = out_ch
        
        # Final channels
        final_channels = decoder_channels[-1]
        
        # Prediction heads
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(final_channels, final_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels, num_person * num_joints, kernel_size=1)
        )
        
        self.elevation_head = nn.Sequential(
            nn.Conv2d(final_channels, final_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels, num_person * num_joints, kernel_size=1)
        )

        self.uncertainty_head = None
        if self.predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(final_channels, final_channels),
                nn.ReLU(inplace=True),
                nn.Linear(final_channels, num_person * num_joints)
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
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, 
                inputs: Dict[str, torch.Tensor],
                data_samples: Optional[List] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with CNN decoder.
        
        Args:
            inputs: Dictionary with:
                - 'query_tokens': (B, N_query, C) - preferred, rich spatial features
                - 'features': (B, C) - fallback, global pooled features
            data_samples: List of data samples with GT (optional)
            
        Returns:
            Dictionary containing heatmaps, elevation_maps, keypoints, and losses
        """
        # Prefer query_tokens over global features for richer representation
        if 'query_tokens' in inputs:
            query_tokens = inputs['query_tokens']  # (B, N_query, C)
            B, N_query, C = query_tokens.shape
            # Reshape query tokens to 2D spatial structure
            # Strategy: treat queries as spatial grid sqrt(N_query) x sqrt(N_query)
            spatial_size = int(N_query ** 0.5)
            if spatial_size * spatial_size == N_query:
                # Perfect square, reshape directly
                feats = query_tokens.transpose(1, 2).view(B, C, spatial_size, spatial_size)
            else:
                # Not perfect square, use global pooling as fallback
                feats = query_tokens.mean(dim=1)  # (B, C)
                use_global = True
        else:
            # Fallback to global features
            feats = inputs['features']  # (B, C)
            B = feats.size(0)
            use_global = True
        
        # Verify batch size
        if data_samples is not None and len(data_samples) != B:
            raise RuntimeError(
                f"Batch size mismatch: features has {B} samples but "
                f"data_samples has {len(data_samples)} samples."
            )
        
        # Initial projection to spatial map
        if len(feats.shape) == 2:  # (B, C) - need to project to spatial
            x = self.initial_proj(feats)  # (B, C*H*W)
            x = x.view(B, -1, self.initial_spatial_size, self.initial_spatial_size)
        else:  # (B, C, H, W) - already spatial
            # Adapt to initial spatial size if needed
            if feats.size(2) != self.initial_spatial_size or feats.size(3) != self.initial_spatial_size:
                x = F.interpolate(feats, size=(self.initial_spatial_size, self.initial_spatial_size),
                                mode='bilinear', align_corners=False)
            else:
                x = feats
            if self.spatial_proj is not None and x.size(1) != self.initial_channels:
                x = self.spatial_proj(x)
        
        # Progressive upsampling through decoder
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Interpolate to target resolution if needed
        if x.size(2) != self.range_bins or x.size(3) != self.angle_bins:
            x = F.interpolate(x, size=(self.range_bins, self.angle_bins), 
                            mode='bilinear', align_corners=False)
        
        # Predict heatmaps and elevation
        heatmap_out = self.heatmap_head(x)  # (B, num_person*num_joints, H, W)
        heatmaps = heatmap_out.view(B, self.num_person, self.num_joints, 
                                   self.range_bins, self.angle_bins)
        heatmaps = torch.sigmoid(heatmaps)

        joint_confidence = self._compute_joint_confidence(heatmaps)
        
        elev_out = self.elevation_head(x)  # (B, num_person*num_joints, H, W)
        elevation_maps = elev_out.view(B, self.num_person, self.num_joints,
                                      self.range_bins, self.angle_bins)
        elevation_maps = torch.tanh(elevation_maps)
        
        # Decode keypoints
        keypoints = self.decode_keypoints(
            heatmaps, elevation_maps,
            use_argmax=not self.training and self.decode_inference_with_argmax
        )

        joint_log_sigma = None
        if self.uncertainty_head is not None:
            pooled = x.mean(dim=(2, 3))
            joint_log_sigma = self.uncertainty_head(pooled)
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
        
        # Compute losses if training
        if data_samples is not None:
            gt_kpts, mask = self._stack_gt(data_samples, keypoints.device)
            if gt_kpts is not None:
                loss_dict = self.loss(outputs, gt_kpts, mask)
                outputs.update(loss_dict)
        
        return outputs

    @staticmethod
    def _compute_joint_confidence(heatmaps: torch.Tensor,
                                  eps: float = 1e-8) -> torch.Tensor:
        """Compute per-joint confidence from heatmap entropy.

        Args:
            heatmaps: (B, P, J, H, W), values in [0, 1]
            eps: Numerical stability epsilon.

        Returns:
            confidence: (B, P, J) in [0, 1]
        """
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
        """Decode 3D keypoints from heatmaps using soft-argmax."""
        B, P, J, H, W = heatmaps.shape
        device = heatmaps.device
        
        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.view(B, P, J, H * W)
        elevation_flat = elevation_maps.view(B, P, J, H * W)
        
        # Coordinate grids
        r_grid = self.r_coords.view(1, 1, 1, H, 1).expand(B, P, J, H, W)
        r_grid_flat = r_grid.reshape(B, P, J, H * W)
        
        az_grid = self.az_coords.view(1, 1, 1, 1, W).expand(B, P, J, H, W)
        az_grid_flat = az_grid.reshape(B, P, J, H * W)
        
        if use_argmax:
            peak_idx = heatmaps_flat.argmax(dim=-1, keepdim=True)
            r_pred = torch.gather(r_grid_flat, dim=-1, index=peak_idx)
            az_pred = torch.gather(az_grid_flat, dim=-1, index=peak_idx)
            el_pred = torch.gather(elevation_flat, dim=-1, index=peak_idx)
        else:
            # Soft-argmax
            heatmap_sum = heatmaps_flat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            r_pred = (r_grid_flat * heatmaps_flat).sum(dim=-1, keepdim=True) / heatmap_sum
            
            # Circular mean for azimuth
            sin_weighted = (torch.sin(az_grid_flat) * heatmaps_flat).sum(dim=-1)
            cos_weighted = (torch.cos(az_grid_flat) * heatmaps_flat).sum(dim=-1)
            az_pred = torch.atan2(sin_weighted, cos_weighted).unsqueeze(-1)
            
            el_pred = (elevation_flat * heatmaps_flat).sum(dim=-1, keepdim=True) / heatmap_sum
        
        # Denormalize elevation
        el_pred = el_pred * (self.el_max - self.el_min) / 2.0 + (self.el_max + self.el_min) / 2.0
        
        keypoints = torch.cat([r_pred, az_pred, el_pred], dim=-1)
        return keypoints
    
    def _stack_gt(self, data_samples: List, device: torch.device):
        """Stack ground truth keypoints from data samples."""
        if not data_samples:
            return None, None
        
        gt_list = []
        mask_list = []
        
        for sample in data_samples:
            gt = sample.get('gt_keypoints_3d', None)
            if gt is None:
                gt = torch.zeros((self.num_person, self.num_joints, 3), device=device)
                mask = torch.zeros((self.num_person, self.num_joints), device=device, dtype=torch.bool)
            else:
                if not isinstance(gt, torch.Tensor):
                    gt = torch.from_numpy(gt).to(device)
                else:
                    gt = gt.to(device)
                
                if gt.dim() == 2:
                    gt = gt.unsqueeze(0)

                # Some datasets include an extra confidence dimension; keep XYZ only.
                if gt.size(-1) > 3:
                    gt = gt[..., :3]

                mask = (gt[..., :3].abs().sum(dim=-1) > 0)
                
                if gt.size(0) < self.num_person:
                    pad_size = self.num_person - gt.size(0)
                    gt = torch.cat([
                        gt,
                        torch.zeros((pad_size, self.num_joints, 3), device=device)
                    ], dim=0)
                    mask = torch.cat([
                        mask,
                        torch.zeros((pad_size, self.num_joints), device=device, dtype=torch.bool)
                    ], dim=0)
                elif gt.size(0) > self.num_person:
                    gt = gt[:self.num_person]
                    mask = mask[:self.num_person]
            
            gt_list.append(gt)
            mask_list.append(mask)
        
        gt_keypoints = torch.stack(gt_list, dim=0)
        gt_mask = torch.stack(mask_list, dim=0)
        
        return gt_keypoints, gt_mask
    
    def loss(self, outputs: Dict, gt_keypoints: torch.Tensor, 
             mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all losses with unified matching.
        
        Key improvement: Perform matching ONCE here, then apply to all losses.
        This ensures heatmap, elevation, and coordinate losses use the same 
        pred-to-gt assignment, avoiding training instability.
        """
        pred_heatmaps = outputs['heatmaps']
        pred_elevation = outputs['elevation_maps']
        pred_keypoints = outputs['keypoints']
        
        B, P, J = pred_heatmaps.size(0), pred_heatmaps.size(1), pred_heatmaps.size(2)
        device = pred_heatmaps.device
        
        # Generate target heatmaps and elevation maps
        target_heatmaps = self._generate_target_heatmaps(gt_keypoints, mask, device)
        target_elevation = self._generate_target_elevation(gt_keypoints, mask, device)

        if not self.use_matching or self.num_person == 1:
            loss_heatmap = self.loss_heatmap_fn(
                pred_heatmaps,
                target_heatmaps,
                mask,
                perform_matching=False
            )

            loss_elevation = self.loss_elevation_fn(
                pred_elevation,
                target_elevation,
                target_heatmaps,
                mask
            )

            pred_cart = spherical_to_cartesian(pred_keypoints)
            gt_cart = spherical_to_cartesian(gt_keypoints)
            kpt_diff = F.smooth_l1_loss(
                pred_cart, gt_cart, beta=self.coord_loss_beta, reduction='none')
            kpt_mask = mask.unsqueeze(-1).float().expand_as(kpt_diff)
            num_valid_kpts = kpt_mask.sum().clamp(min=1.0)
            loss_coord = (kpt_diff * kpt_mask).sum() / num_valid_kpts

            losses = {
                'loss_heatmap': loss_heatmap,
                'loss_elevation': loss_elevation,
                'loss_coord': loss_coord * self.coord_loss_weight
            }

            if self.loss_skeleton_angle_fn is not None:
                losses['loss_skeleton_angle'] = self.loss_skeleton_angle_fn(
                    pred_keypoints, gt_keypoints, mask)
            if self.loss_skeleton_length_fn is not None:
                losses['loss_skeleton_length'] = self.loss_skeleton_length_fn(
                    pred_keypoints, gt_keypoints, mask)

            return losses
        
        # ========================================================================
        # UNIFIED MATCHING: Compute pred-to-gt assignments once for all losses
        # ========================================================================
        assignments = self._compute_assignments(pred_heatmaps, target_heatmaps, 
                                                gt_keypoints, mask)
        
        # Apply unified matching to compute all losses
        loss_heatmap = self._compute_matched_heatmap_loss(
            pred_heatmaps, target_heatmaps, mask, assignments)
        
        loss_elevation = self._compute_matched_elevation_loss(
            pred_elevation, target_elevation, target_heatmaps, mask, assignments)
        
        loss_coord = self._compute_matched_coord_loss(
            pred_keypoints, gt_keypoints, mask, assignments)
        
        losses = {
            'loss_heatmap': loss_heatmap,
            'loss_elevation': loss_elevation,
            'loss_coord': loss_coord
        }
        
        # Skeleton losses (also use matched assignments)
        if self.loss_skeleton_angle_fn is not None:
            losses['loss_skeleton_angle'] = self._compute_matched_skeleton_loss(
                pred_keypoints, gt_keypoints, mask, assignments, 'angle')
        
        if self.loss_skeleton_length_fn is not None:
            losses['loss_skeleton_length'] = self._compute_matched_skeleton_loss(
                pred_keypoints, gt_keypoints, mask, assignments, 'length')
        
        return losses
    
    def _compute_assignments(self, pred_heatmaps: torch.Tensor, 
                            target_heatmaps: torch.Tensor,
                            gt_keypoints: torch.Tensor,
                            mask: torch.Tensor) -> List[List[int]]:
        """
        Compute unified pred-to-gt assignments for all samples in batch.
        Uses Euclidean distance matching (same as evaluation metric) for consistency.
        
        Returns:
            List of assignments, one per sample. Each assignment[b][g] gives 
            the pred index for gt person g in sample b.
        """
        from mmwave3d.models.losses.heatmap_losses import find_best_assignment
        
        B, P, J = pred_heatmaps.size(0), pred_heatmaps.size(1), pred_heatmaps.size(2)
        assignments = []
        
        # Decode keypoints from heatmaps for matching
        pred_keypoints = self.decode_keypoints(pred_heatmaps, 
                                               self._generate_target_elevation(gt_keypoints, mask, pred_heatmaps.device),
                                               use_argmax=False)  # (B, P, J, 3) in spherical
        
        # Convert to Cartesian coordinates for Euclidean distance matching
        pred_cart = spherical_to_cartesian(pred_keypoints)  # (B, P, J, 3)
        gt_cart = spherical_to_cartesian(gt_keypoints)  # (B, P, J, 3)
        
        for b in range(B):
            # Identify valid GT persons
            person_valid = mask[b].any(dim=1)  # (P,)
            valid_gt_indices = torch.where(person_valid)[0]
            num_gt = len(valid_gt_indices)
            
            if num_gt == 0:
                # No GT: empty assignment
                assignments.append([])
                continue
            
            # Compute cost matrix using Euclidean distance in Cartesian space
            num_valid_gt = len(valid_gt_indices)
            cost_matrix = torch.zeros((P, num_valid_gt), device=pred_heatmaps.device)
            
            for pi in range(P):
                for gi, gt_idx in enumerate(valid_gt_indices):
                    # Compute MPJPE for this pair
                    joint_mask = mask[b, gt_idx]  # (J,)
                    
                    if joint_mask.any():
                        diff = pred_cart[b, pi, joint_mask] - gt_cart[b, gt_idx, joint_mask]
                        cost_matrix[pi, gi] = torch.norm(diff, dim=-1).mean()
                    else:
                        cost_matrix[pi, gi] = 1e6  # Large cost for no overlap
            
            # Hungarian matching
            assignment = find_best_assignment(cost_matrix, num_gt)
            
            # Convert local gt indices to global
            assignment_global = []
            for gt_local_idx, pred_idx in enumerate(assignment):
                gt_global_idx = valid_gt_indices[gt_local_idx].item()
                assignment_global.append((pred_idx, gt_global_idx))
            
            assignments.append(assignment_global)
        
        return assignments
    
    def _compute_matched_heatmap_loss(self, pred_heatmaps: torch.Tensor,
                                     target_heatmaps: torch.Tensor,
                                     mask: torch.Tensor,
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
                    pred_heatmaps[b], 
                    torch.zeros_like(target_heatmaps[b])
                )
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
                    target_heatmaps[b:b+1, gt_idx:gt_idx+1]
                )
                sample_loss += loss
            
            # Unmatched predictions (false positives)
            unmatched = set(range(P)) - matched_preds
            if unmatched:
                unmatched_idx = torch.tensor(list(unmatched), device=device)
                loss_fp = self.loss_heatmap_fn.focal_loss(
                    pred_heatmaps[b:b+1, unmatched_idx],
                    torch.zeros_like(pred_heatmaps[b:b+1, unmatched_idx])
                )
                sample_loss += loss_fp * 0.5
            
            total_loss += sample_loss
            valid_samples += 1
        
        loss = (total_loss / valid_samples) if valid_samples > 0 else total_loss
        return loss * self.loss_heatmap_fn.loss_weight
    
    def _compute_matched_elevation_loss(self, pred_elevation: torch.Tensor,
                                       target_elevation: torch.Tensor,
                                       heatmap_weight: torch.Tensor,
                                       mask: torch.Tensor,
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
            
            # Only compute loss for matched pairs
            for pred_idx, gt_idx in assignments[b]:
                weight_mask = (heatmap_weight[b, gt_idx] > threshold).float()
                joint_mask = mask[b, gt_idx].float().view(J, 1, 1).expand(J, H, W)
                weight_mask = weight_mask * joint_mask
                
                diff = torch.abs(pred_elevation[b, pred_idx] - target_elevation[b, gt_idx])
                weighted_loss = diff * weight_mask
                
                total_loss += weighted_loss.sum()
                total_weight += weight_mask.sum()
        
        loss = total_loss / total_weight.clamp(min=1.0)
        return loss * self.loss_elevation_fn.loss_weight
    
    def _compute_matched_coord_loss(self, pred_keypoints: torch.Tensor,
                                   gt_keypoints: torch.Tensor,
                                   mask: torch.Tensor,
                                   assignments: List[List]) -> torch.Tensor:
        """Compute coordinate loss using pre-computed assignments."""
        B = pred_keypoints.size(0)
        device = pred_keypoints.device
        
        total_loss = torch.tensor(0.0, device=device)
        valid_count = 0
        
        for b in range(B):
            if not assignments[b]:
                continue
            
            for pred_idx, gt_idx in assignments[b]:
                joint_mask = mask[b, gt_idx]
                if joint_mask.any():
                    diff = pred_keypoints[b, pred_idx, joint_mask] - gt_keypoints[b, gt_idx, joint_mask]
                    loss = F.smooth_l1_loss(diff, torch.zeros_like(diff),
                                          beta=self.coord_loss_beta, reduction='mean')
                    total_loss += loss
                    valid_count += 1
        
        if valid_count == 0:
            return pred_keypoints.new_tensor(0.0)
        
        return (total_loss / valid_count) * self.coord_loss_weight
    
    def _compute_matched_skeleton_loss(self, pred_keypoints: torch.Tensor,
                                      gt_keypoints: torch.Tensor,
                                      mask: torch.Tensor,
                                      assignments: List[List],
                                      loss_type: str) -> torch.Tensor:
        """Compute skeleton angle/length loss using pre-computed assignments."""
        B, P, J = pred_keypoints.size(0), pred_keypoints.size(1), pred_keypoints.size(2)
        device = pred_keypoints.device
        
        # Prepare matched predictions and targets
        matched_preds = []
        matched_gts = []
        matched_masks = []
        
        for b in range(B):
            for pred_idx, gt_idx in assignments[b]:
                matched_preds.append(pred_keypoints[b, pred_idx])
                matched_gts.append(gt_keypoints[b, gt_idx])
                matched_masks.append(mask[b, gt_idx])
        
        if not matched_preds:
            return pred_keypoints.new_tensor(0.0)
        
        # Stack and reshape to (B_matched, 1, J, 3) format expected by loss functions
        matched_preds = torch.stack(matched_preds).unsqueeze(1)  # (N, 1, J, 3)
        matched_gts = torch.stack(matched_gts).unsqueeze(1)
        matched_masks = torch.stack(matched_masks).unsqueeze(1)
        
        # Compute loss
        if loss_type == 'angle':
            return self.loss_skeleton_angle_fn(matched_preds, matched_gts, matched_masks)
        else:  # 'length'
            return self.loss_skeleton_length_fn(matched_preds, matched_gts, matched_masks)
    
    def _generate_target_heatmaps(self, gt_keypoints: torch.Tensor, 
                                  mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Generate Gaussian heatmaps from GT keypoints."""
        B, P, J, _ = gt_keypoints.shape
        H, W = self.range_bins, self.angle_bins
        
        target = torch.zeros((B, P, J, H, W), device=device)
        
        r_grid = self.r_coords.view(1, 1, 1, H, 1)
        az_grid = self.az_coords.view(1, 1, 1, 1, W)
        
        for b in range(B):
            for p in range(P):
                for j in range(J):
                    if not mask[b, p, j]:
                        continue
                    
                    r_gt, az_gt, el_gt = gt_keypoints[b, p, j]
                    
                    dr = (r_grid - r_gt) / self.sigma_r
                    daz = (az_grid - az_gt) / self.sigma_a
                    
                    gaussian = torch.exp(-(dr**2 + daz**2) / 2)
                    target[b, p, j] = gaussian.squeeze(0)
        
        return target
    
    def _generate_target_elevation(self, gt_keypoints: torch.Tensor,
                                   mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Generate target elevation maps with Gaussian weighting.
        
        Key improvement: Instead of a constant value across the entire map,
        elevation is weighted by the same Gaussian as the heatmap. This provides
        spatial structure and makes elevation learning more stable and aligned
        with heatmap learning.
        """
        B, P, J, _ = gt_keypoints.shape
        H, W = self.range_bins, self.angle_bins
        
        target = torch.zeros((B, P, J, H, W), device=device)
        
        r_grid = self.r_coords.view(1, 1, 1, H, 1)
        az_grid = self.az_coords.view(1, 1, 1, 1, W)
        
        for b in range(B):
            for p in range(P):
                for j in range(J):
                    if not mask[b, p, j]:
                        continue
                    
                    r_gt, az_gt, el_gt = gt_keypoints[b, p, j]
                    
                    # Normalize elevation
                    el_norm = (el_gt - (self.el_max + self.el_min) / 2.0) / \
                             ((self.el_max - self.el_min) / 2.0)
                    el_norm = torch.clamp(el_norm, -1.0, 1.0)
                    
                    # Generate Gaussian weight (same as heatmap)
                    dr = (r_grid - r_gt) / self.sigma_r
                    daz = (az_grid - az_gt) / self.sigma_a
                    gaussian = torch.exp(-(dr**2 + daz**2) / 2)
                    
                    # Elevation map = elevation value * Gaussian weight
                    # This provides spatial structure and makes the target more informative
                    target[b, p, j] = el_norm * gaussian.squeeze(0)
        
        return target
    
    def _compute_coord_loss(self, pred: torch.Tensor, gt: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        """Compute coordinate loss with matching."""
        pred_cart = spherical_to_cartesian(pred)
        gt_cart = spherical_to_cartesian(gt)
        
        # Legacy function kept for compatibility but unused
        # Matching is now done in main loss() function
        return pred.new_tensor(0.0)
