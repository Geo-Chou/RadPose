from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmwave3d.registry import MODELS
from mmwave3d.utils import FlashMultiheadAttention, spherical_to_cartesian


class GraphConv(nn.Module):
    """GCN layer with optional dynamic edge weighting: X' = A_norm @ X @ W."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
            x: (B, J, C) node features
            a_norm: (J, J) normalized adjacency matrix
            edge_weight: (B, J, J) optional dynamic edge weights based on confidence
        """
        if edge_weight is not None:
            a_norm = a_norm.unsqueeze(0) * edge_weight  # (B, J, J)
            x = torch.einsum('bij,bjf->bif', a_norm, x)
        else:
            x = torch.einsum('ij,bjf->bif', a_norm, x)
        return self.linear(x)


class ForwardKinematicsLayer(nn.Module):
    """Forward kinematics using fixed bone lengths and predicted directions."""

    def __init__(self,
                 num_joints: int,
                 parents: Sequence[int],
                 bone_lengths: Sequence[float],
                 root_index: int = 6) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.root_index = root_index
        self.parents = list(parents)
        self.register_buffer(
            'bone_lengths_base',
            torch.tensor(bone_lengths, dtype=torch.float32),
            persistent=False
        )
        self.order = self._build_topo_order()

    def _build_topo_order(self) -> List[int]:
        children = {i: [] for i in range(self.num_joints)}
        for j, p in enumerate(self.parents):
            if p >= 0:
                children[p].append(j)
        order = [self.root_index]
        idx = 0
        while idx < len(order):
            node = order[idx]
            order.extend(children[node])
            idx += 1
        if len(order) != self.num_joints:
            missing = [i for i in range(self.num_joints) if i not in order]
            order.extend(missing)
        return order

    def forward(self,
                root_pos: torch.Tensor,
                directions: torch.Tensor,
                bone_length_scales: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
            root_pos: (B, P, 3)
            directions: (B, P, J, 3) - unit vectors for each joint
            bone_length_scales: (B, P, J) optional scaling factors for bone lengths
        Returns:
            positions: (B, P, J, 3)
        """
        bsz, num_person, num_joints, _ = directions.shape
        positions = directions.new_zeros((bsz, num_person, num_joints, 3))
        positions[:, :, self.root_index] = root_pos
        
        if bone_length_scales is not None:
            lengths = self.bone_lengths_base.to(directions.device).view(1, 1, -1)  # (1, 1, J)
            lengths = lengths * bone_length_scales  # (B, P, J)
        else:
            lengths = self.bone_lengths_base.to(directions.device)
        
        for j in self.order:
            if j == self.root_index:
                continue
            parent = self.parents[j]
            if parent < 0:
                continue
            if bone_length_scales is not None:
                bone_len = lengths[:, :, j]  # (B, P)
                positions[:, :, j] = positions[:, :, parent] + directions[:, :, j] * bone_len.unsqueeze(-1)
            else:
                positions[:, :, j] = positions[:, :, parent] + directions[:, :, j] * lengths[j]
        return positions


def cartesian_to_spherical(xyz: torch.Tensor) -> torch.Tensor:
    if xyz is None or xyz.shape[-1] < 3:
        return xyz
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    r = torch.sqrt(x * x + y * y + z * z).clamp_min(1e-8)
    az = torch.atan2(z, x)
    el = torch.atan2(y, torch.sqrt(x * x + z * z).clamp_min(1e-8))
    out = torch.stack((r, az, el), dim=-1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@MODELS.register_module()
class IKHead(nn.Module):
    """Inverse kinematics head with GCN + forward kinematics."""

    def __init__(self,
                 in_dim: int,
                 num_person: int = 2,
                 num_joints: int = 16,
                 root_index: int = 6,
                 skeleton_edges: Sequence[Tuple[int, int]] = (),
                 bone_length_means: Sequence[Tuple[int, int, float]] = (),
                 gcn_hidden: int = 128,
                 gcn_layers: int = 3,
                 loss_ik_coord: Optional[Dict] = None,
                 loss_ik_bias_weight: float = 0.0,
                 loss_ik_bias_beta: float = 0.1,
                 loss_ik_angle_weight: float = 0.0,
                 loss_ik_angle_type: str = 'geodesic',
                 loss_ik_angle_beta: float = 0.1,
                 loss_fk_consistency_weight: float = 0.0,
                 loss_fk_consistency_beta: float = 0.1,
                 loss_graph_reg_weight: float = 0.0,
                 loss_bone_length_weight: float = 0.0,
                 bone_length_range: float = 0.15,
                 use_confidence: bool = True,
                 use_confidence_weighting: bool = True,
                 confidence_min: float = 1e-3,
                 confidence_max: float = 1.0,
                 point_cloud_range: Sequence[float] = (0.12, -0.524, -0.524, 6.0, 2.618, 0.524),
                 softargmax_temperature: float = 3.0,
                 joint_attn_num_heads: int = 8,
                 joint_attn_dropout: float = 0.0,
                 use_depth_weighting: bool = True,
                 depth_weight_scale: float = 0.5,
                 fk_residual_weight: float = 0.1,
                 fk_residual_beta: float = 0.1,
                 loss_fk_residual_reg_weight: float = 0.01,
                 loss_root_pos_weight: float = 0.5,
                 loss_root_residual_reg_weight: float = 0.01) -> None:
        super().__init__()
        if in_dim % joint_attn_num_heads != 0:
            raise ValueError(
                f'in_dim ({in_dim}) must be divisible by '
                f'joint_attn_num_heads ({joint_attn_num_heads}).'
            )

        self.in_dim = in_dim
        self.num_person = num_person
        self.num_joints = num_joints
        self.root_index = root_index
        self.r_min, self.az_min, self.el_min = point_cloud_range[:3]
        self.r_max, self.az_max, self.el_max = point_cloud_range[3:]
        self.softargmax_temperature = float(softargmax_temperature)

        parents, bone_lengths = self._build_skeleton(skeleton_edges, bone_length_means)
        self.parents = parents
        
        default_directions = [
            [-0.0203,  0.9988,  0.0443],
            [-0.0778,  0.9934, -0.0846],
            [-0.9512, -0.0645,  0.3018],
            [ 0.9106,  0.0037,  0.4133],
            [ 0.0921,  0.9922, -0.0838],
            [ 0.0093,  0.9987,  0.0500],
            [ 0.0000,  1.0000,  0.0000],
            [ 0.0112, -0.9946,  0.1031],
            [ 0.0151, -0.9619,  0.2731],
            [ 0.0114, -0.9987,  0.0495],
            [ 0.1471,  0.8432, -0.5170],
            [-0.1308,  0.9900, -0.0532],
            [-0.9085, -0.1237,  0.3991],
            [ 0.9087, -0.0919,  0.4072],
            [ 0.0757,  0.9950, -0.0656],
            [-0.2322,  0.8992, -0.3709],
        ]
        self.register_buffer('default_dir', torch.tensor(
            default_directions, dtype=torch.float32
        ), persistent=False)

        depth_map = torch.tensor([
            3,
            2,
            1,
            1,
            2,
            3,
            0,
            1,
            2,
            3,
            4,
            3,
            2,
            2,
            3,
            4,
        ], dtype=torch.float32)
        self.register_buffer('depth_map', depth_map, persistent=False)
        self.use_depth_weighting = use_depth_weighting
        self.depth_weight_scale = depth_weight_scale

        a_norm = self._build_adj_norm(skeleton_edges)
        self.register_buffer('a_norm', a_norm, persistent=False)

        # Per-joint query -> cross-attend transformer query tokens.
        self.joint_query = nn.Parameter(
            torch.randn(1, num_joints, in_dim, dtype=torch.float32)
        )
        self.kpt_to_query = nn.Linear(3, in_dim)
        self.joint_cross_attn = FlashMultiheadAttention(
            embed_dim=in_dim,
            num_heads=joint_attn_num_heads,
            dropout=joint_attn_dropout,
            batch_first=True
        )
        self.joint_attn_norm = nn.LayerNorm(in_dim)

        self.feat_proj = nn.Linear(in_dim, gcn_hidden)
        self.kpt_proj = nn.Linear(3, gcn_hidden)
        self.conf_proj = nn.Linear(1, gcn_hidden)
        self.joint_fuse = nn.Linear(gcn_hidden * 3, gcn_hidden)
        self.gcn_layers = nn.ModuleList([
            GraphConv(gcn_hidden, gcn_hidden) for _ in range(gcn_layers)
        ])
        self.out_proj = nn.Linear(gcn_hidden, 3)
        self.fk_residual_head = nn.Sequential(
            nn.Linear(gcn_hidden, gcn_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(gcn_hidden // 2, 3)
        )
        self.root_residual_head = nn.Sequential(
            nn.Linear(gcn_hidden, gcn_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(gcn_hidden // 2, 3)
        )
        self._init_residual_heads_zero()
        
        self.bone_length_head = nn.Sequential(
            nn.Linear(gcn_hidden, gcn_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(gcn_hidden // 2, 1)
        )

        self.fk = ForwardKinematicsLayer(
            num_joints=num_joints,
            parents=parents,
            bone_lengths=bone_lengths,
            root_index=root_index
        )

        if loss_ik_coord is None:
            loss_ik_coord = dict(type='MPJPELoss', loss_weight=1.0, reduction='mean')
        self.loss_ik_coord_fn = MODELS.build(loss_ik_coord)
        self.loss_ik_bias_weight = float(loss_ik_bias_weight)
        self.loss_ik_bias_beta = float(loss_ik_bias_beta)
        self.loss_ik_angle_weight = float(loss_ik_angle_weight)
        self.loss_ik_angle_type = str(loss_ik_angle_type)
        self.loss_ik_angle_beta = float(loss_ik_angle_beta)
        self.loss_fk_consistency_weight = float(loss_fk_consistency_weight)
        self.loss_fk_consistency_beta = float(loss_fk_consistency_beta)
        self.loss_graph_reg_weight = float(loss_graph_reg_weight)
        self.loss_bone_length_weight = float(loss_bone_length_weight)
        self.bone_length_range = float(bone_length_range)
        self.use_confidence = bool(use_confidence)
        self.use_confidence_weighting = bool(use_confidence_weighting)
        self.confidence_min = float(confidence_min)
        self.confidence_max = float(confidence_max)
        self.fk_residual_weight = float(fk_residual_weight)
        self.fk_residual_beta = float(fk_residual_beta)
        self.loss_fk_residual_reg_weight = float(loss_fk_residual_reg_weight)
        self.loss_root_pos_weight = float(loss_root_pos_weight)
        self.loss_root_residual_reg_weight = float(loss_root_residual_reg_weight)

    def _init_residual_heads_zero(self) -> None:
        """Initialize residual branches to output zero at training start."""
        for head in (self.fk_residual_head, self.root_residual_head):
            last = head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def _build_skeleton(self,
                        skeleton_edges: Sequence[Tuple[int, int]],
                        bone_length_means: Sequence[Tuple[int, int, float]]):
        parents = [-1] * self.num_joints
        for child, parent in skeleton_edges:
            parents[child] = parent
        bone_lengths = [0.0] * self.num_joints
        for child, parent, mean_len in bone_length_means:
            bone_lengths[child] = float(mean_len)
            if parents[child] == -1:
                parents[child] = parent
        if parents[self.root_index] != -1:
            parents[self.root_index] = -1
        return parents, bone_lengths

    def _build_adj_norm(self, skeleton_edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
        a = torch.zeros((self.num_joints, self.num_joints), dtype=torch.float32)
        for i in range(self.num_joints):
            a[i, i] = 1.0
        for child, parent in skeleton_edges:
            a[child, parent] = 1.0
            a[parent, child] = 1.0
        deg = a.sum(dim=1).clamp_min(1.0)
        d_inv_sqrt = torch.pow(deg, -0.5)
        a_norm = d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]
        return a_norm

    def _stack_gt(self, data_samples: List, device: torch.device):
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
        return torch.stack(gt_list, dim=0), torch.stack(mask_list, dim=0)

    def _build_joint_feat(self,
                          feats: Dict[str, torch.Tensor],
                          kpt_cart: torch.Tensor) -> torch.Tensor:
        bsz, num_person, num_joints, _ = kpt_cart.shape
        if 'query_tokens' not in feats:
            global_feat = feats['features']
            return global_feat.unsqueeze(1).unsqueeze(2).expand(
                bsz, num_person, num_joints, -1)

        query_tokens = feats['query_tokens']
        if query_tokens.dim() != 3:
            raise RuntimeError(
                f'Expected query_tokens as (B, N, C), got {tuple(query_tokens.shape)}.'
            )
        if query_tokens.size(0) != bsz:
            raise RuntimeError(
                f'Batch size mismatch: query_tokens B={query_tokens.size(0)} vs keypoints B={bsz}.'
            )
        if query_tokens.size(-1) != self.in_dim:
            raise RuntimeError(
                f'query_tokens channel ({query_tokens.size(-1)}) must equal IK in_dim ({self.in_dim}).'
            )

        bsz_person = bsz * num_person
        token_count = query_tokens.size(1)
        kv = query_tokens.unsqueeze(1).expand(
            bsz, num_person, token_count, self.in_dim
        ).reshape(bsz_person, token_count, self.in_dim)

        joint_query = self.joint_query.expand(bsz_person, -1, -1)
        pose_query = self.kpt_to_query(kpt_cart.reshape(bsz_person, num_joints, 3))
        query = joint_query + pose_query

        joint_feat, _ = self.joint_cross_attn(
            query=query,
            key=kv,
            value=kv,
            need_weights=False
        )
        joint_feat = self.joint_attn_norm(joint_feat + query)
        return joint_feat.reshape(bsz, num_person, num_joints, self.in_dim)

    def forward(self,
                feats: Dict[str, torch.Tensor],
                heatmap_outputs: Dict[str, torch.Tensor],
                data_samples: Optional[List] = None) -> Dict[str, torch.Tensor]:
        heatmaps = heatmap_outputs.get('heatmaps', None)
        elevation_maps = heatmap_outputs.get('elevation_maps', None)
        if heatmaps is None or elevation_maps is None:
            raise RuntimeError('IKHead requires heatmaps and elevation_maps for differentiable decoding.')
        confidence = heatmap_outputs.get('joint_confidence', None)
        log_sigma = heatmap_outputs.get('joint_log_sigma', None)
        keypoints = self._decode_keypoints_from_heatmap(heatmaps, elevation_maps)
        kpt_cart = spherical_to_cartesian(keypoints)

        bsz, num_person, num_joints, _ = kpt_cart.shape
        if num_joints != self.num_joints:
            raise RuntimeError(
                f'IKHead expected {self.num_joints} joints but got {num_joints}.'
            )

        joint_feat = self._build_joint_feat(feats, kpt_cart)
        feat_embed = self.feat_proj(joint_feat)
        kpt_embed = self.kpt_proj(kpt_cart)
        if confidence is not None:
            conf_feat = confidence.reshape(bsz, num_person, num_joints, 1)
        else:
            conf_feat = torch.ones(
                (bsz, num_person, num_joints, 1),
                device=kpt_cart.device,
                dtype=kpt_cart.dtype
            )
        conf_embed = self.conf_proj(conf_feat)
        x = self.joint_fuse(torch.cat([feat_embed, kpt_embed, conf_embed], dim=-1))
        x = x.reshape(bsz * num_person, num_joints, -1)

        edge_weight = None
        if self.use_confidence_weighting and confidence is not None:
            edge_weight = self._compute_edge_weights(
                confidence.reshape(bsz * num_person, num_joints)
            )  # (B*P, J, J)

        # GCN with dynamic edge weighting
        for gcn in self.gcn_layers:
            x = gcn(x, self.a_norm, edge_weight)
            x = F.relu(x, inplace=True)

        dir_resid = torch.tanh(self.out_proj(x)).view(bsz, num_person, num_joints, 3)
        
        bone_scale_logits = self.bone_length_head(x).view(bsz, num_person, num_joints)
        bone_length_scales = 1.0 + self.bone_length_range * torch.tanh(bone_scale_logits)
        dir_init = self._compute_initial_dirs(kpt_cart)
        directions = F.normalize(dir_init + dir_resid, dim=-1, eps=1e-6)
        directions[:, :, self.root_index] = 0.0

        root_base = kpt_cart[:, :, self.root_index]
        root_feat = x.view(bsz, num_person, num_joints, -1)[:, :, self.root_index, :]
        root_delta = self.root_residual_head(root_feat)
        root_pos = root_base + root_delta

        fk_cart = self.fk(root_pos, directions, bone_length_scales)
        pos_delta = self.fk_residual_weight * torch.tanh(
            self.fk_residual_head(x).view(bsz, num_person, num_joints, 3)
        )
        ik_cart = fk_cart + pos_delta
        ik_spherical = cartesian_to_spherical(ik_cart)

        outputs = {
            'ik_keypoints_fk_cartesian': fk_cart,
            'ik_keypoints_cartesian': ik_cart,
            'ik_keypoints': ik_spherical,
            'ik_directions': directions,
            'ik_bias': dir_resid,
            'bone_length_scales': bone_length_scales,
            'ik_pos_delta': pos_delta,
            'ik_root_pos_base': root_base,
            'ik_root_pos_delta': root_delta,
            'ik_root_pos': root_pos
        }
        if confidence is not None:
            outputs['ik_confidence'] = confidence
        if log_sigma is not None:
            outputs['ik_log_sigma'] = log_sigma

        if data_samples is not None:
            gt_kpts, mask = self._stack_gt(data_samples, ik_spherical.device)
            if gt_kpts is not None:
                gt_cart = spherical_to_cartesian(gt_kpts)
                coord_weights = None
                if self.use_confidence and confidence is not None:
                    coord_weights = confidence.clamp(
                        min=self.confidence_min,
                        max=self.confidence_max
                    )
                elif log_sigma is not None:
                    sigma = log_sigma.exp()
                    coord_weights = (1.0 / (sigma ** 2)).clamp(0.0, 1.0)

                if self.num_person > 1:
                    # Multi-person unified matching for IK losses.
                    assignments = self._compute_person_assignments(
                        ik_cart, gt_cart, mask, coord_weights)
                    matched_gt_kpts, matched_mask = self._build_matched_targets(
                        gt_kpts, mask, assignments)
                else:
                    # Single-person: keep direct index alignment.
                    matched_gt_kpts, matched_mask = gt_kpts, mask
                matched_gt_cart = spherical_to_cartesian(matched_gt_kpts)

                if log_sigma is not None:
                    try:
                        outputs['loss_ik_coord'] = self.loss_ik_coord_fn(
                            ik_spherical, matched_gt_kpts, mask=matched_mask, log_sigma=log_sigma)
                    except TypeError:
                        outputs['loss_ik_coord'] = self.loss_ik_coord_fn(
                            ik_spherical, matched_gt_kpts, mask=matched_mask, weights=coord_weights)
                else:
                    outputs['loss_ik_coord'] = self.loss_ik_coord_fn(
                        ik_spherical, matched_gt_kpts, mask=matched_mask, weights=coord_weights)

                if self.loss_ik_bias_weight > 0.0:
                    bias_loss = self._compute_bias_loss(
                        dir_resid, dir_init, matched_gt_cart, matched_mask)
                    outputs['loss_ik_bias'] = bias_loss
                if self.loss_ik_angle_weight > 0.0:
                    outputs['loss_ik_angle'] = self._compute_angle_loss(
                        directions, matched_gt_cart, matched_mask)
                if self.loss_fk_consistency_weight > 0.0:
                    outputs['loss_fk_consistency'] = self._compute_fk_consistency_loss(
                        ik_cart, matched_gt_cart, matched_mask)
                if self.loss_fk_residual_reg_weight > 0.0:
                    outputs['loss_fk_residual_reg'] = self._compute_delta_reg_loss(pos_delta, matched_mask)
                if self.loss_root_residual_reg_weight > 0.0:
                    outputs['loss_root_residual_reg'] = self._compute_root_delta_reg_loss(root_delta)
                if self.loss_root_pos_weight > 0.0:
                    outputs['loss_root_pos'] = self._compute_root_pos_loss(
                        root_pos, matched_gt_cart, matched_mask)
                if self.loss_graph_reg_weight > 0.0:
                    outputs['loss_graph_reg'] = self._compute_graph_reg_loss(
                        x.view(bsz, num_person, num_joints, -1), matched_mask)
                if self.loss_bone_length_weight > 0.0:
                    outputs['loss_bone_length'] = self._compute_bone_length_loss(
                        bone_length_scales, matched_gt_cart, matched_mask)

        return outputs

    def _compute_person_assignments(self,
                                    pred_cart: torch.Tensor,
                                    gt_cart: torch.Tensor,
                                    mask: torch.Tensor,
                                    weights: Optional[torch.Tensor] = None) -> List[List[Tuple[int, int]]]:
        """Compute person-level pred->gt matching for each sample."""
        from mmwave3d.models.losses.heatmap_losses import find_best_assignment

        bsz, num_pred, _, _ = pred_cart.shape
        assignments: List[List[Tuple[int, int]]] = []
        for b in range(bsz):
            valid_gt_indices = torch.where(mask[b].any(dim=1))[0]
            num_gt = int(valid_gt_indices.numel())
            if num_gt == 0:
                assignments.append([])
                continue

            cost_matrix = pred_cart.new_zeros((num_pred, num_gt))
            for pi in range(num_pred):
                for gi, gt_idx in enumerate(valid_gt_indices):
                    joint_mask = mask[b, gt_idx]
                    if joint_mask.any():
                        diff = pred_cart[b, pi, joint_mask] - gt_cart[b, gt_idx, joint_mask]
                        if weights is not None:
                            w = weights[b, pi, joint_mask].to(diff.dtype)
                            denom = w.sum().clamp_min(1e-6)
                            cost_matrix[pi, gi] = (diff.norm(dim=-1) * w).sum() / denom
                        else:
                            cost_matrix[pi, gi] = diff.norm(dim=-1).mean()
                    else:
                        cost_matrix[pi, gi] = pred_cart.new_tensor(1e6)

            gt_to_pred = find_best_assignment(cost_matrix, num_gt)
            sample_pairs: List[Tuple[int, int]] = []
            for gi, pred_idx in enumerate(gt_to_pred):
                if pred_idx < 0 or pred_idx >= num_pred:
                    continue
                sample_pairs.append((int(pred_idx), int(valid_gt_indices[gi].item())))
            assignments.append(sample_pairs)
        return assignments

    def _build_matched_targets(self,
                               gt_keypoints: torch.Tensor,
                               gt_mask: torch.Tensor,
                               assignments: List[List[Tuple[int, int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reorder GT tensors into prediction slots using pred->gt assignments."""
        matched_gt = torch.zeros_like(gt_keypoints)
        matched_mask = torch.zeros_like(gt_mask)
        bsz = gt_keypoints.size(0)
        for b in range(bsz):
            for pred_idx, gt_idx in assignments[b]:
                matched_gt[b, pred_idx] = gt_keypoints[b, gt_idx]
                matched_mask[b, pred_idx] = gt_mask[b, gt_idx]
        return matched_gt, matched_mask

    def _decode_keypoints_from_heatmap(self,
                                       heatmaps: torch.Tensor,
                                       elevation_maps: torch.Tensor) -> torch.Tensor:
        """Decode spherical keypoints from heatmaps using sharpened soft-argmax."""
        bsz, num_person, num_joints, h, w = heatmaps.shape
        device = heatmaps.device
        dtype = heatmaps.dtype

        if self.softargmax_temperature != 1.0:
            heatmaps = torch.pow(heatmaps.clamp_min(1e-8), self.softargmax_temperature)

        heatmaps_flat = heatmaps.view(bsz, num_person, num_joints, h * w)
        elevation_flat = elevation_maps.view(bsz, num_person, num_joints, h * w)

        r_coords = torch.linspace(self.r_min, self.r_max, h, device=device, dtype=dtype)
        az_coords = torch.linspace(self.az_min, self.az_max, w, device=device, dtype=dtype)

        r_grid = r_coords.view(1, 1, 1, h, 1).expand(bsz, num_person, num_joints, h, w)
        r_grid_flat = r_grid.reshape(bsz, num_person, num_joints, h * w)
        az_grid = az_coords.view(1, 1, 1, 1, w).expand(bsz, num_person, num_joints, h, w)
        az_grid_flat = az_grid.reshape(bsz, num_person, num_joints, h * w)

        heatmap_sum = heatmaps_flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        r_pred = (r_grid_flat * heatmaps_flat).sum(dim=-1, keepdim=True) / heatmap_sum

        sin_weighted = (torch.sin(az_grid_flat) * heatmaps_flat).sum(dim=-1)
        cos_weighted = (torch.cos(az_grid_flat) * heatmaps_flat).sum(dim=-1)
        az_pred = torch.atan2(sin_weighted, cos_weighted).unsqueeze(-1)

        el_pred = (elevation_flat * heatmaps_flat).sum(dim=-1, keepdim=True) / heatmap_sum
        el_pred = el_pred * (self.el_max - self.el_min) / 2.0 + (self.el_max + self.el_min) / 2.0
        return torch.cat([r_pred, az_pred, el_pred], dim=-1)

    def _compute_initial_dirs(self, kpt_cart: torch.Tensor) -> torch.Tensor:
        bsz, num_person, num_joints, _ = kpt_cart.shape
        dirs = kpt_cart.new_zeros((bsz, num_person, num_joints, 3))
        for j, parent in enumerate(self.parents):
            if parent < 0:
                continue
            delta = kpt_cart[:, :, j] - kpt_cart[:, :, parent]
            norm = delta.norm(dim=-1, keepdim=True)
            fallback = self.default_dir[j].view(1, 1, 3).to(kpt_cart.device)
            use_delta = norm > 1e-6
            dirs[:, :, j] = torch.where(use_delta, delta / norm.clamp_min(1e-6), fallback)
        return dirs

    def _compute_bias_loss(self,
                           pred_bias: torch.Tensor,
                           dir_init: torch.Tensor,
                           gt_cart: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        dir_gt = self._compute_initial_dirs(gt_cart)
        bias_gt = dir_gt - dir_init

        joint_mask = mask.clone()
        for j, parent in enumerate(self.parents):
            if parent < 0:
                joint_mask[..., j] = False
            else:
                joint_mask[..., j] = mask[..., j] & mask[..., parent]

        valid = joint_mask.unsqueeze(-1).to(dtype=pred_bias.dtype)
        loss = F.smooth_l1_loss(
            pred_bias, bias_gt, reduction='none', beta=self.loss_ik_bias_beta)
        loss = loss * valid
        denom = valid.sum().clamp_min(1.0)
        loss = loss.sum() / denom
        return loss * self.loss_ik_bias_weight

    def _compute_angle_loss(self,
                            pred_dirs: torch.Tensor,
                            gt_cart: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        gt_dirs = self._compute_initial_dirs(gt_cart)
        joint_mask = mask.clone()
        for j, parent in enumerate(self.parents):
            if parent < 0:
                joint_mask[..., j] = False
            else:
                joint_mask[..., j] = mask[..., j] & mask[..., parent]

        valid = joint_mask.unsqueeze(-1).to(dtype=pred_dirs.dtype)
        if self.loss_ik_angle_type.lower() == 'l1':
            loss = F.smooth_l1_loss(
                pred_dirs, gt_dirs, reduction='none', beta=self.loss_ik_angle_beta)
            loss = loss * valid
            denom = valid.sum().clamp_min(1.0)
            loss = loss.sum() / denom
        else:
            dot = (pred_dirs * gt_dirs).sum(dim=-1).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
            geodesic = torch.acos(dot)
            scalar_valid = joint_mask.to(dtype=geodesic.dtype)
            denom = scalar_valid.sum().clamp_min(1.0)
            loss = (geodesic * scalar_valid).sum() / denom
        return loss * self.loss_ik_angle_weight

    def _compute_fk_consistency_loss(self,
                                     pred_cart: torch.Tensor,
                                     gt_cart: torch.Tensor,
                                     mask: torch.Tensor) -> torch.Tensor:
        loss = F.smooth_l1_loss(
            pred_cart, gt_cart, reduction='none', beta=self.loss_fk_consistency_beta)
        
        if self.use_depth_weighting:
            # weight = 1.0 + depth_weight_scale * depth
            depth_weights = 1.0 + self.depth_weight_scale * self.depth_map.to(pred_cart.device)
            depth_weights = depth_weights.view(1, 1, -1, 1)  # (1, 1, J, 1)
            loss = loss * depth_weights
        
        valid = mask.unsqueeze(-1).to(dtype=pred_cart.dtype)
        loss = loss * valid
        denom = valid.sum().clamp_min(1.0)
        loss = loss.sum() / denom
        return loss * self.loss_fk_consistency_weight

    def _compute_graph_reg_loss(self,
                                node_feat: torch.Tensor,
                                mask: torch.Tensor) -> torch.Tensor:
        if self.loss_graph_reg_weight <= 0.0:
            return node_feat.sum() * 0.0

        # Normalize node features first to remove scale effects.
        node_feat = F.normalize(node_feat, dim=-1, eps=1e-6)
        edge_terms = []
        edge_valids = []
        for child, parent in enumerate(self.parents):
            if parent < 0:
                continue
            cos_sim = (node_feat[:, :, child] * node_feat[:, :, parent]).sum(dim=-1)
            edge_terms.append(1.0 - cos_sim.clamp(min=-1.0, max=1.0))
            edge_valids.append((mask[:, :, child] & mask[:, :, parent]).to(dtype=node_feat.dtype))

        if not edge_terms:
            return node_feat.sum() * 0.0

        edge_loss = torch.stack(edge_terms, dim=-1)
        edge_valid = torch.stack(edge_valids, dim=-1)
        denom = edge_valid.sum().clamp_min(1.0)
        loss = (edge_loss * edge_valid).sum() / denom
        return loss * self.loss_graph_reg_weight

    def _compute_edge_weights(self, confidence: torch.Tensor) -> torch.Tensor:
        """Compute confidence-based dynamic edge weights.

        Args:
            confidence: (B, J) joint confidence scores

        Returns:
            edge_weight: (B, J, J) dynamic edge weights
        """
        bsz, num_joints = confidence.shape
        
        conf_norm = confidence.clamp(self.confidence_min, self.confidence_max)
        
        conf_matrix = conf_norm.unsqueeze(-1) * conf_norm.unsqueeze(1)  # (B, J, J)
        edge_weight = torch.sqrt(conf_matrix.clamp_min(1e-6))
        
        skeleton_mask = (self.a_norm > 0).float()  # (J, J)
        edge_weight = edge_weight * skeleton_mask.unsqueeze(0) + (1.0 - skeleton_mask.unsqueeze(0))
        
        return edge_weight

    def _compute_bone_length_loss(self,
                                  pred_scales: torch.Tensor,
                                  gt_cart: torch.Tensor,
                                  mask: torch.Tensor) -> torch.Tensor:
        """Compute the bone-length prediction loss.

        Args:
            pred_scales: (B, P, J) predicted bone length scales
            gt_cart: (B, P, J, 3) ground truth positions in Cartesian
            mask: (B, P, J) validity mask

        Returns:
            loss: scalar loss
        """
        if self.loss_bone_length_weight <= 0.0:
            return pred_scales.sum() * 0.0
        
        bsz, num_person, num_joints = pred_scales.shape
        base_lengths = self.fk.bone_lengths_base.to(pred_scales.device)  # (J,)
        
        gt_bone_lengths = gt_cart.new_zeros((bsz, num_person, num_joints))
        for j, parent in enumerate(self.parents):
            if parent < 0:
                continue
            delta = gt_cart[:, :, j] - gt_cart[:, :, parent]  # (B, P, 3)
            gt_bone_lengths[:, :, j] = delta.norm(dim=-1)  # (B, P)
        
        gt_scales = gt_bone_lengths / base_lengths.view(1, 1, -1).clamp_min(1e-6)
        
        joint_mask = mask.clone()
        for j, parent in enumerate(self.parents):
            if parent < 0:
                joint_mask[:, :, j] = False
            else:
                joint_mask[:, :, j] = mask[:, :, j] & mask[:, :, parent]
        
        # Smooth L1 loss
        diff = pred_scales - gt_scales
        loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction='none', beta=0.1)
        
        valid = joint_mask.to(dtype=loss.dtype)
        loss = loss * valid
        denom = valid.sum().clamp_min(1.0)
        loss = loss.sum() / denom
        
        return loss * self.loss_bone_length_weight

    def _compute_delta_reg_loss(self,
                                pos_delta: torch.Tensor,
                                mask: torch.Tensor) -> torch.Tensor:
        if self.loss_fk_residual_reg_weight <= 0.0:
            return pos_delta.sum() * 0.0
        valid = mask.unsqueeze(-1).to(dtype=pos_delta.dtype)
        zeros = torch.zeros_like(pos_delta)
        loss = F.smooth_l1_loss(
            pos_delta, zeros, reduction='none', beta=self.fk_residual_beta
        )
        loss = loss * valid
        denom = valid.sum().clamp_min(1.0)
        return (loss.sum() / denom) * self.loss_fk_residual_reg_weight

    def _compute_root_delta_reg_loss(self, root_delta: torch.Tensor) -> torch.Tensor:
        if self.loss_root_residual_reg_weight <= 0.0:
            return root_delta.sum() * 0.0
        loss = F.smooth_l1_loss(
            root_delta, torch.zeros_like(root_delta), reduction='mean', beta=self.fk_residual_beta
        )
        return loss * self.loss_root_residual_reg_weight

    def _compute_root_pos_loss(self,
                               root_pos: torch.Tensor,
                               gt_cart: torch.Tensor,
                               mask: torch.Tensor) -> torch.Tensor:
        if self.loss_root_pos_weight <= 0.0:
            return root_pos.sum() * 0.0
        gt_root = gt_cart[:, :, self.root_index]
        root_valid = mask[:, :, self.root_index].to(dtype=root_pos.dtype).unsqueeze(-1)
        loss = F.smooth_l1_loss(
            root_pos, gt_root, reduction='none', beta=self.fk_residual_beta
        )
        loss = loss * root_valid
        denom = root_valid.sum().clamp_min(1.0)
        return (loss.sum() / denom) * self.loss_root_pos_weight
