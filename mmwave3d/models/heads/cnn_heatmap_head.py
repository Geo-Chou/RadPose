import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmwave3d.models.losses.heatmap_losses import (
    find_best_assignment,
)
from mmwave3d.registry import MODELS
from mmwave3d.utils import spherical_to_cartesian


@MODELS.register_module()
class CNNHeatmapHead(nn.Module):
    def __init__(
        self,
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
        log_sigma_max: float = 7.0,
        use_elevation_heatmap: bool = False,
        elev_bins: int = 90,
        sigma_el: float = 0.0218,
        re_heatmap_weight: float = 1.0,
        range_consistency_weight: float = 0.0,
    ) -> None:
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

        self.r_min, self.az_min, self.el_min = point_cloud_range[:3]
        self.r_max, self.az_max, self.el_max = point_cloud_range[3:]

        r_coords = torch.linspace(self.r_min, self.r_max, range_bins)
        az_coords = torch.linspace(self.az_min, self.az_max, angle_bins)
        self.register_buffer("r_coords", r_coords, persistent=False)
        self.register_buffer("az_coords", az_coords, persistent=False)

        self.use_elevation_heatmap = bool(use_elevation_heatmap)
        self.elev_bins = int(elev_bins)
        self.sigma_el = float(sigma_el)
        self.re_heatmap_weight = float(re_heatmap_weight)
        self.range_consistency_weight = float(range_consistency_weight)
        el_coords = torch.linspace(self.el_min, self.el_max, self.elev_bins)
        self.register_buffer("el_coords", el_coords, persistent=False)

        initial_channels = decoder_channels[0]
        self.initial_channels = initial_channels
        spatial_dim = initial_spatial_size * initial_spatial_size * initial_channels
        self.initial_proj = nn.Sequential(
            nn.Linear(in_dim, spatial_dim), nn.ReLU(inplace=True)
        )

        self.decoder_layers = nn.ModuleList()
        in_ch = initial_channels

        self.spatial_proj = None
        if in_dim != initial_channels:
            self.spatial_proj = nn.Conv2d(in_dim, initial_channels, kernel_size=1)

        for out_ch in decoder_channels[1:]:
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_ch, out_ch, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch

        final_channels = decoder_channels[-1]

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(final_channels, final_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels, num_person * num_joints, kernel_size=1),
        )

        self.elevation_head = nn.Sequential(
            nn.Conv2d(final_channels, final_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels, num_person * num_joints, kernel_size=1),
        )

        self.re_proj = None
        self.re_heatmap_head = None
        if self.use_elevation_heatmap:
            self.re_proj = nn.Sequential(
                nn.Conv2d(final_channels, final_channels, kernel_size=1),
                nn.ReLU(inplace=True),
            )
            self.re_heatmap_head = nn.Sequential(
                nn.Conv2d(final_channels, final_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(final_channels, num_person * num_joints, kernel_size=1),
            )

        self.uncertainty_head = None
        if self.predict_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(final_channels, final_channels),
                nn.ReLU(inplace=True),
                nn.Linear(final_channels, num_person * num_joints),
            )

        self._init_weights()

        if loss_heatmap is None:
            loss_heatmap = dict(type="Heatmap2DLoss", loss_weight=1.0)
        self.loss_heatmap_fn = MODELS.build(loss_heatmap)

        if loss_elevation is None:
            loss_elevation = dict(type="ElevationLoss", loss_weight=1.0)
        self.loss_elevation_fn = MODELS.build(loss_elevation)

        self.loss_skeleton_angle_fn = (
            MODELS.build(loss_skeleton_angle) if loss_skeleton_angle else None
        )
        self.loss_skeleton_length_fn = (
            MODELS.build(loss_skeleton_length) if loss_skeleton_length else None
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, inputs: Dict[str, torch.Tensor], data_samples: Optional[List] = None
    ) -> Dict[str, torch.Tensor]:

        if "query_tokens" in inputs:
            query_tokens = inputs["query_tokens"]
            B, N_query, C = query_tokens.shape

            spatial_size = int(N_query**0.5)
            if spatial_size * spatial_size == N_query:
                feats = query_tokens.transpose(1, 2).view(
                    B, C, spatial_size, spatial_size
                )
            else:
                feats = query_tokens.mean(dim=1)
        else:
            feats = inputs["features"]
            B = feats.size(0)

        if data_samples is not None and len(data_samples) != B:
            raise RuntimeError(
                f"Batch size mismatch: features has {B} samples but "
                f"data_samples has {len(data_samples)} samples."
            )

        if len(feats.shape) == 2:
            x = self.initial_proj(feats)
            x = x.view(B, -1, self.initial_spatial_size, self.initial_spatial_size)
        else:
            if (
                feats.size(2) != self.initial_spatial_size
                or feats.size(3) != self.initial_spatial_size
            ):
                x = F.interpolate(
                    feats,
                    size=(self.initial_spatial_size, self.initial_spatial_size),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                x = feats
            if self.spatial_proj is not None and x.size(1) != self.initial_channels:
                x = self.spatial_proj(x)

        for layer in self.decoder_layers:
            x = layer(x)

        if x.size(2) != self.range_bins or x.size(3) != self.angle_bins:
            x = F.interpolate(
                x,
                size=(self.range_bins, self.angle_bins),
                mode="bilinear",
                align_corners=False,
            )

        heatmap_out = self.heatmap_head(x)
        heatmaps = heatmap_out.view(
            B, self.num_person, self.num_joints, self.range_bins, self.angle_bins
        )

        heatmaps = torch.sigmoid(heatmaps.float())

        joint_confidence = self._compute_joint_confidence(heatmaps)

        elev_out = self.elevation_head(x)
        elevation_maps = elev_out.view(
            B, self.num_person, self.num_joints, self.range_bins, self.angle_bins
        )
        elevation_maps = torch.tanh(elevation_maps)

        re_heatmaps = None
        if self.use_elevation_heatmap:
            x_re = self.re_proj(x)
            x_re = F.interpolate(
                x_re,
                size=(self.range_bins, self.elev_bins),
                mode="bilinear",
                align_corners=False,
            )
            re_out = self.re_heatmap_head(x_re)
            re_heatmaps = re_out.view(
                B, self.num_person, self.num_joints, self.range_bins, self.elev_bins
            )

            re_heatmaps = torch.sigmoid(re_heatmaps.float())

        use_argmax = not self.training and self.decode_inference_with_argmax
        if self.use_elevation_heatmap:
            keypoints = self.decode_keypoints_dual(
                heatmaps, re_heatmaps, use_argmax=use_argmax
            )
        else:
            keypoints = self.decode_keypoints(
                heatmaps, elevation_maps, use_argmax=use_argmax
            )

        joint_log_sigma = None
        if self.uncertainty_head is not None:
            pooled = x.mean(dim=(2, 3))
            joint_log_sigma = self.uncertainty_head(pooled)
            joint_log_sigma = joint_log_sigma.view(B, self.num_person, self.num_joints)
            joint_log_sigma = joint_log_sigma.clamp(
                self.log_sigma_min, self.log_sigma_max
            )

        outputs = {
            "heatmaps": heatmaps,
            "elevation_maps": elevation_maps,
            "keypoints": keypoints,
            "joint_confidence": joint_confidence,
        }
        if re_heatmaps is not None:
            outputs["re_heatmaps"] = re_heatmaps
        if joint_log_sigma is not None:
            outputs["joint_log_sigma"] = joint_log_sigma
            outputs["joint_sigma"] = torch.exp(joint_log_sigma)

        if data_samples is not None:
            gt_kpts, mask = self._stack_gt(data_samples, keypoints.device)
            if gt_kpts is not None:
                loss_dict = self.loss(outputs, gt_kpts, mask)
                outputs.update(loss_dict)

        return outputs

    @staticmethod
    def _compute_joint_confidence(
        heatmaps: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        B, P, J, H, W = heatmaps.shape
        heatmaps_flat = heatmaps.view(B, P, J, H * W)
        heatmap_sum = heatmaps_flat.sum(dim=-1, keepdim=True)
        prob = heatmaps_flat / (heatmap_sum + eps)
        entropy = -(prob * (prob + eps).log()).sum(dim=-1)
        entropy = entropy / math.log(H * W)
        confidence = 1.0 - entropy
        confidence = torch.where(
            heatmap_sum.squeeze(-1) > eps, confidence, confidence.new_zeros(())
        )
        return confidence.clamp(0.0, 1.0)

    def _soft_argmax_1d(
        self, dist: torch.Tensor, coords: torch.Tensor, use_argmax: bool
    ) -> torch.Tensor:
        if use_argmax:
            idx = dist.argmax(dim=-1, keepdim=True)
            return torch.gather(coords.expand_as(dist), dim=-1, index=idx)
        total = dist.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return (dist * coords).sum(dim=-1, keepdim=True) / total

    def _circular_argmax_1d(
        self, dist: torch.Tensor, coords: torch.Tensor, use_argmax: bool
    ) -> torch.Tensor:
        if use_argmax:
            idx = dist.argmax(dim=-1, keepdim=True)
            return torch.gather(coords.expand_as(dist), dim=-1, index=idx)
        sin_w = (torch.sin(coords) * dist).sum(dim=-1)
        cos_w = (torch.cos(coords) * dist).sum(dim=-1)
        return torch.atan2(sin_w, cos_w).unsqueeze(-1)

    def decode_keypoints_dual(
        self,
        heatmaps: torch.Tensor,
        re_heatmaps: torch.Tensor,
        use_argmax: bool = False,
    ) -> torch.Tensor:
        B, P, J, R, W = heatmaps.shape
        E = re_heatmaps.size(-1)

        ra_range = heatmaps.sum(dim=-1)
        ra_az = heatmaps.sum(dim=-2)
        re_range = re_heatmaps.sum(dim=-1)
        re_el = re_heatmaps.sum(dim=-2)

        r_coords = self.r_coords.view(1, 1, 1, R)
        az_coords = self.az_coords.view(1, 1, 1, W)
        el_coords = self.el_coords.view(1, 1, 1, E)

        r_from_ra = self._soft_argmax_1d(ra_range, r_coords, use_argmax)
        r_from_re = self._soft_argmax_1d(re_range, r_coords, use_argmax)
        r_pred = 0.5 * (r_from_ra + r_from_re)

        az_pred = self._circular_argmax_1d(ra_az, az_coords, use_argmax)
        el_pred = self._soft_argmax_1d(re_el, el_coords, use_argmax)

        return torch.cat([r_pred, az_pred, el_pred], dim=-1)

    def decode_keypoints(
        self,
        heatmaps: torch.Tensor,
        elevation_maps: torch.Tensor,
        use_argmax: bool = False,
    ) -> torch.Tensor:
        B, P, J, H, W = heatmaps.shape

        heatmaps_flat = heatmaps.view(B, P, J, H * W)
        elevation_flat = elevation_maps.view(B, P, J, H * W)

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
            heatmap_sum = heatmaps_flat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            r_pred = (r_grid_flat * heatmaps_flat).sum(
                dim=-1, keepdim=True
            ) / heatmap_sum

            sin_weighted = (torch.sin(az_grid_flat) * heatmaps_flat).sum(dim=-1)
            cos_weighted = (torch.cos(az_grid_flat) * heatmaps_flat).sum(dim=-1)
            az_pred = torch.atan2(sin_weighted, cos_weighted).unsqueeze(-1)

            el_pred = (elevation_flat * heatmaps_flat).sum(
                dim=-1, keepdim=True
            ) / heatmap_sum

        el_pred = (
            el_pred * (self.el_max - self.el_min) / 2.0
            + (self.el_max + self.el_min) / 2.0
        )

        keypoints = torch.cat([r_pred, az_pred, el_pred], dim=-1)
        return keypoints

    def _stack_gt(self, data_samples: List, device: torch.device):
        if not data_samples:
            return None, None

        gt_list = []
        mask_list = []

        for sample in data_samples:
            gt = sample.get("gt_keypoints_3d", None)
            if gt is None:
                gt = torch.zeros((self.num_person, self.num_joints, 3), device=device)
                mask = torch.zeros(
                    (self.num_person, self.num_joints), device=device, dtype=torch.bool
                )
            else:
                if not isinstance(gt, torch.Tensor):
                    gt = torch.from_numpy(gt).to(device)
                else:
                    gt = gt.to(device)

                if gt.dim() == 2:
                    gt = gt.unsqueeze(0)

                if gt.size(-1) > 3:
                    gt = gt[..., :3]

                mask = gt[..., :3].abs().sum(dim=-1) > 0

                if gt.size(0) < self.num_person:
                    pad_size = self.num_person - gt.size(0)
                    gt = torch.cat(
                        [
                            gt,
                            torch.zeros((pad_size, self.num_joints, 3), device=device),
                        ],
                        dim=0,
                    )
                    mask = torch.cat(
                        [
                            mask,
                            torch.zeros(
                                (pad_size, self.num_joints),
                                device=device,
                                dtype=torch.bool,
                            ),
                        ],
                        dim=0,
                    )
                elif gt.size(0) > self.num_person:
                    gt = gt[: self.num_person]
                    mask = mask[: self.num_person]

            gt_list.append(gt)
            mask_list.append(mask)

        gt_keypoints = torch.stack(gt_list, dim=0)
        gt_mask = torch.stack(mask_list, dim=0)

        return gt_keypoints, gt_mask

    def loss(
        self, outputs: Dict, gt_keypoints: torch.Tensor, mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        pred_heatmaps = outputs["heatmaps"]
        pred_elevation = outputs["elevation_maps"]
        pred_keypoints = outputs["keypoints"]

        _B, _P, _J = pred_heatmaps.size(0), pred_heatmaps.size(1), pred_heatmaps.size(2)
        device = pred_heatmaps.device

        target_heatmaps = self._generate_target_heatmaps(gt_keypoints, mask, device)
        target_elevation = self._generate_target_elevation(gt_keypoints, mask, device)

        if not self.use_matching or self.num_person == 1:
            loss_heatmap = self.loss_heatmap_fn(
                pred_heatmaps, target_heatmaps, mask, perform_matching=False
            )

            loss_elevation = self.loss_elevation_fn(
                pred_elevation, target_elevation, target_heatmaps, mask
            )

            pred_cart = spherical_to_cartesian(pred_keypoints)
            gt_cart = spherical_to_cartesian(gt_keypoints)
            kpt_diff = F.smooth_l1_loss(
                pred_cart, gt_cart, beta=self.coord_loss_beta, reduction="none"
            )
            kpt_mask = mask.unsqueeze(-1).float().expand_as(kpt_diff)
            num_valid_kpts = kpt_mask.sum().clamp(min=1.0)
            loss_coord = (kpt_diff * kpt_mask).sum() / num_valid_kpts

            losses = {
                "loss_heatmap": loss_heatmap,
                "loss_elevation": loss_elevation,
                "loss_coord": loss_coord * self.coord_loss_weight,
            }

            if self.use_elevation_heatmap and "re_heatmaps" in outputs:
                pred_re = outputs["re_heatmaps"]
                target_re = self._generate_target_re_heatmaps(
                    gt_keypoints, mask, device
                )
                losses["loss_re_heatmap"] = (
                    self.re_heatmap_weight
                    * self.loss_heatmap_fn(
                        pred_re, target_re, mask, perform_matching=False
                    )
                )

                if self.loss_elevation_fn is None or self.use_elevation_heatmap:
                    losses.pop("loss_elevation", None)
                if self.range_consistency_weight > 0.0:
                    r_coords = self.r_coords.view(1, 1, 1, -1)
                    r_ra = self._soft_argmax_1d(
                        pred_heatmaps.sum(dim=-1), r_coords, False
                    )
                    r_re = self._soft_argmax_1d(pred_re.sum(dim=-1), r_coords, False)
                    gap = (r_ra - r_re).abs().squeeze(-1)
                    denom = mask.to(gap.dtype).sum().clamp(min=1.0)
                    losses["loss_range_consistency"] = (
                        self.range_consistency_weight
                        * (gap * mask.to(gap.dtype)).sum()
                        / denom
                    )

            if self.loss_skeleton_angle_fn is not None:
                losses["loss_skeleton_angle"] = self.loss_skeleton_angle_fn(
                    pred_keypoints, gt_keypoints, mask
                )
            if self.loss_skeleton_length_fn is not None:
                losses["loss_skeleton_length"] = self.loss_skeleton_length_fn(
                    pred_keypoints, gt_keypoints, mask
                )

            return losses

        assignments = self._compute_assignments(
            pred_heatmaps, target_heatmaps, gt_keypoints, mask
        )

        loss_heatmap = self._compute_matched_heatmap_loss(
            pred_heatmaps, target_heatmaps, mask, assignments
        )

        loss_elevation = self._compute_matched_elevation_loss(
            pred_elevation, target_elevation, target_heatmaps, mask, assignments
        )

        loss_coord = self._compute_matched_coord_loss(
            pred_keypoints, gt_keypoints, mask, assignments
        )

        losses = {
            "loss_heatmap": loss_heatmap,
            "loss_elevation": loss_elevation,
            "loss_coord": loss_coord,
        }

        if self.loss_skeleton_angle_fn is not None:
            losses["loss_skeleton_angle"] = self._compute_matched_skeleton_loss(
                pred_keypoints, gt_keypoints, mask, assignments, "angle"
            )

        if self.loss_skeleton_length_fn is not None:
            losses["loss_skeleton_length"] = self._compute_matched_skeleton_loss(
                pred_keypoints, gt_keypoints, mask, assignments, "length"
            )

        return losses

    def _compute_assignments(
        self,
        pred_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
        gt_keypoints: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[List[int]]:

        B, P, _J = pred_heatmaps.size(0), pred_heatmaps.size(1), pred_heatmaps.size(2)
        assignments = []

        pred_keypoints = self.decode_keypoints(
            pred_heatmaps,
            self._generate_target_elevation(gt_keypoints, mask, pred_heatmaps.device),
            use_argmax=False,
        )

        pred_cart = spherical_to_cartesian(pred_keypoints)
        gt_cart = spherical_to_cartesian(gt_keypoints)

        for b in range(B):
            person_valid = mask[b].any(dim=1)
            valid_gt_indices = torch.where(person_valid)[0]
            num_gt = len(valid_gt_indices)

            if num_gt == 0:
                assignments.append([])
                continue

            num_valid_gt = len(valid_gt_indices)
            cost_matrix = torch.zeros((P, num_valid_gt), device=pred_heatmaps.device)

            for pi in range(P):
                for gi, gt_idx in enumerate(valid_gt_indices):
                    joint_mask = mask[b, gt_idx]

                    if joint_mask.any():
                        diff = (
                            pred_cart[b, pi, joint_mask]
                            - gt_cart[b, gt_idx, joint_mask]
                        )
                        cost_matrix[pi, gi] = torch.norm(diff, dim=-1).mean()
                    else:
                        cost_matrix[pi, gi] = 1e6

            assignment = find_best_assignment(cost_matrix, num_gt)

            assignment_global = []
            for gt_local_idx, pred_idx in enumerate(assignment):
                gt_global_idx = valid_gt_indices[gt_local_idx].item()
                assignment_global.append((pred_idx, gt_global_idx))

            assignments.append(assignment_global)

        return assignments

    def _compute_matched_heatmap_loss(
        self,
        pred_heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
        mask: torch.Tensor,
        assignments: List[List],
    ) -> torch.Tensor:
        B, P = pred_heatmaps.size(0), pred_heatmaps.size(1)
        device = pred_heatmaps.device

        total_loss = torch.tensor(0.0, device=device)
        valid_samples = 0

        for b in range(B):
            if not assignments[b]:
                loss = self.loss_heatmap_fn.focal_loss(
                    pred_heatmaps[b], torch.zeros_like(target_heatmaps[b])
                )
                total_loss += loss
                valid_samples += 1
                continue

            sample_loss = torch.tensor(0.0, device=device)
            matched_preds = set()

            for pred_idx, gt_idx in assignments[b]:
                matched_preds.add(pred_idx)
                loss = self.loss_heatmap_fn.focal_loss(
                    pred_heatmaps[b : b + 1, pred_idx : pred_idx + 1],
                    target_heatmaps[b : b + 1, gt_idx : gt_idx + 1],
                )
                sample_loss += loss

            unmatched = set(range(P)) - matched_preds
            if unmatched:
                unmatched_idx = torch.tensor(list(unmatched), device=device)
                loss_fp = self.loss_heatmap_fn.focal_loss(
                    pred_heatmaps[b : b + 1, unmatched_idx],
                    torch.zeros_like(pred_heatmaps[b : b + 1, unmatched_idx]),
                )
                sample_loss += loss_fp * 0.5

            total_loss += sample_loss
            valid_samples += 1

        loss = (total_loss / valid_samples) if valid_samples > 0 else total_loss
        return loss * self.loss_heatmap_fn.loss_weight

    def _compute_matched_elevation_loss(
        self,
        pred_elevation: torch.Tensor,
        target_elevation: torch.Tensor,
        heatmap_weight: torch.Tensor,
        mask: torch.Tensor,
        assignments: List[List],
    ) -> torch.Tensor:
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
                joint_mask = mask[b, gt_idx].float().view(J, 1, 1).expand(J, H, W)
                weight_mask = weight_mask * joint_mask

                diff = torch.abs(
                    pred_elevation[b, pred_idx] - target_elevation[b, gt_idx]
                )
                weighted_loss = diff * weight_mask

                total_loss += weighted_loss.sum()
                total_weight += weight_mask.sum()

        loss = total_loss / total_weight.clamp(min=1.0)
        return loss * self.loss_elevation_fn.loss_weight

    def _compute_matched_coord_loss(
        self,
        pred_keypoints: torch.Tensor,
        gt_keypoints: torch.Tensor,
        mask: torch.Tensor,
        assignments: List[List],
    ) -> torch.Tensor:
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
                    diff = (
                        pred_keypoints[b, pred_idx, joint_mask]
                        - gt_keypoints[b, gt_idx, joint_mask]
                    )
                    loss = F.smooth_l1_loss(
                        diff,
                        torch.zeros_like(diff),
                        beta=self.coord_loss_beta,
                        reduction="mean",
                    )
                    total_loss += loss
                    valid_count += 1

        if valid_count == 0:
            return pred_keypoints.new_tensor(0.0)

        return (total_loss / valid_count) * self.coord_loss_weight

    def _compute_matched_skeleton_loss(
        self,
        pred_keypoints: torch.Tensor,
        gt_keypoints: torch.Tensor,
        mask: torch.Tensor,
        assignments: List[List],
        loss_type: str,
    ) -> torch.Tensor:
        B, _P, _J = (
            pred_keypoints.size(0),
            pred_keypoints.size(1),
            pred_keypoints.size(2),
        )

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

        matched_preds = torch.stack(matched_preds).unsqueeze(1)
        matched_gts = torch.stack(matched_gts).unsqueeze(1)
        matched_masks = torch.stack(matched_masks).unsqueeze(1)

        if loss_type == "angle":
            return self.loss_skeleton_angle_fn(
                matched_preds, matched_gts, matched_masks
            )
        else:
            return self.loss_skeleton_length_fn(
                matched_preds, matched_gts, matched_masks
            )

    def _generate_target_heatmaps(
        self, gt_keypoints: torch.Tensor, mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
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

    def _generate_target_re_heatmaps(
        self, gt_keypoints: torch.Tensor, mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        B, P, J, _ = gt_keypoints.shape
        R, E = self.range_bins, self.elev_bins

        r_gt = gt_keypoints[..., 0].view(B, P, J, 1, 1)
        el_gt = gt_keypoints[..., 2].view(B, P, J, 1, 1)

        dr = (self.r_coords.view(1, 1, 1, R, 1) - r_gt) / self.sigma_r
        de = (self.el_coords.view(1, 1, 1, 1, E) - el_gt) / self.sigma_el
        target = torch.exp(-(dr**2 + de**2) / 2)
        return target * mask.view(B, P, J, 1, 1).to(target.dtype)

    def _generate_target_elevation(
        self, gt_keypoints: torch.Tensor, mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
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

                    el_norm = (el_gt - (self.el_max + self.el_min) / 2.0) / (
                        (self.el_max - self.el_min) / 2.0
                    )
                    el_norm = torch.clamp(el_norm, -1.0, 1.0)

                    dr = (r_grid - r_gt) / self.sigma_r
                    daz = (az_grid - az_gt) / self.sigma_a
                    gaussian = torch.exp(-(dr**2 + daz**2) / 2)

                    target[b, p, j] = el_norm * gaussian.squeeze(0)

        return target

    def _compute_coord_loss(
        self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        spherical_to_cartesian(pred)
        spherical_to_cartesian(gt)

        return pred.new_tensor(0.0)
