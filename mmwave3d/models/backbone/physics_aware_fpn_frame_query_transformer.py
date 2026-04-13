from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from mmwave3d.registry import MODELS

from .fpn_frame_query_transformer import (
    FPNFrameQueryTransformer,
    HAS_SPCONV,
    spconv,
)


@MODELS.register_module()
class PhysicsAwareFPNFrameQueryTransformer(FPNFrameQueryTransformer):
    """FPNFrameQueryTransformer with a physics-aware anisotropic sparse stem.

    The downstream temporal/query pipeline is kept unchanged. Only the local
    sparse feature extractor is replaced by three lightweight branches:
    radial `(k,1,1)`, angular `(1,k,k)`, and isotropic `(1,1,1)`, fused with
    gates predicted from voxel features and normalized range.
    """

    def __init__(self,
                 input_token_dim: int = 3,
                 pos_dim: int = 4,
                 embed_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 ffn_dim: int = 1024,
                 dropout: float = 0.1,
                 fpn_channels: Sequence[int] = (64, 128, 256),
                 spconv_channels: Sequence[int] = (64, 64),
                 spconv_kernel_size: int = 3,
                 temporal_layers: int = 2,
                 num_query_tokens: int = 256,
                 downsample_tokens: int = 2048,
                 downsample_method: str = 'attention',
                 non_center_downsample_tokens: Optional[int] = None,
                 norm_first: bool = True,
                 pos_freq_bands: Optional[Sequence[float]] = (1.0, 2.0, 4.0, 8.0),
                 use_isotropic_branch: bool = True) -> None:
        super().__init__(
            input_token_dim=input_token_dim,
            pos_dim=pos_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            fpn_channels=fpn_channels,
            spconv_channels=spconv_channels,
            spconv_kernel_size=spconv_kernel_size,
            temporal_layers=temporal_layers,
            num_query_tokens=num_query_tokens,
            downsample_tokens=downsample_tokens,
            downsample_method=downsample_method,
            non_center_downsample_tokens=non_center_downsample_tokens,
            norm_first=norm_first,
            pos_freq_bands=pos_freq_bands,
        )

        self.use_isotropic_branch = bool(use_isotropic_branch)
        self.num_physics_branches = 3 if self.use_isotropic_branch else 2

        self.branch_fusion = nn.Linear(
            self.num_physics_branches * self.spconv_out_channels,
            self.spconv_out_channels)
        self.stem_residual = nn.Linear(input_token_dim, self.spconv_out_channels)

        if self.use_spconv and HAS_SPCONV:
            self.radial_spconv = self._build_spconv_branch(
                input_token_dim, spconv_channels, (spconv_kernel_size, 1, 1),
                branch_name='psa_radial')
            self.angular_spconv = self._build_spconv_branch(
                input_token_dim, spconv_channels, (1, spconv_kernel_size, spconv_kernel_size),
                branch_name='psa_angular')
            if self.use_isotropic_branch:
                self.isotropic_spconv = self._build_spconv_branch(
                    input_token_dim, spconv_channels, (1, 1, 1),
                    branch_name='psa_isotropic')
        else:
            self.radial_linear = nn.Sequential(
                nn.Linear(input_token_dim, self.spconv_out_channels),
                nn.ReLU(inplace=True),
            )
            self.angular_linear = nn.Sequential(
                nn.Linear(input_token_dim, self.spconv_out_channels),
                nn.ReLU(inplace=True),
            )
            if self.use_isotropic_branch:
                self.isotropic_linear = nn.Sequential(
                    nn.Linear(input_token_dim, self.spconv_out_channels),
                    nn.ReLU(inplace=True),
                )

    def _build_spconv_branch(self,
                             input_dim: int,
                             channels: Sequence[int],
                             kernel_size: Tuple[int, int, int],
                             branch_name: str):
        layers = []
        in_ch = input_dim
        for layer_idx, out_ch in enumerate(channels):
            layers.extend([
                spconv.SubMConv3d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    padding=tuple(k // 2 for k in kernel_size),
                    bias=False,
                    indice_key=f'{branch_name}_{layer_idx}',
                ),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        return spconv.SparseSequential(*layers)

    def _match_sparse_features(self,
                               sparse_out,
                               all_in_lin_cat: torch.Tensor,
                               sort_idx: torch.Tensor,
                               sorted_out_lin: torch.Tensor) -> torch.Tensor:
        matched = sparse_out.features.new_zeros(
            (all_in_lin_cat.numel(), sparse_out.features.size(1)))
        if sorted_out_lin.numel() == 0:
            return matched
        pos = torch.searchsorted(sorted_out_lin, all_in_lin_cat)
        valid = (pos < sorted_out_lin.numel()) & (sorted_out_lin[pos] == all_in_lin_cat)
        if valid.any():
            matched[valid] = sparse_out.features[sort_idx[pos[valid]]]
        return matched

    def _apply_spconv_sparse(self, sparse_tokens: torch.Tensor, sparse_indices: torch.Tensor,
                             sparse_shape: torch.Tensor, sparse_mask: torch.Tensor) -> torch.Tensor:
        bsz, frames, max_tokens, _ = sparse_tokens.shape
        device = sparse_tokens.device

        if isinstance(sparse_shape, torch.Tensor):
            spatial_shape = sparse_shape[0].cpu().tolist() if sparse_shape.dim() > 1 else sparse_shape.cpu().tolist()
        else:
            spatial_shape = list(sparse_shape)

        spatial_shape_t = torch.tensor(spatial_shape, device=device, dtype=torch.long)
        stride_r = spatial_shape_t[1] * spatial_shape_t[2]
        stride_a = spatial_shape_t[2]
        frame_stride = stride_r * spatial_shape_t[0]

        all_feats = []
        all_coords = []
        all_in_lin = []
        all_out_pos = []
        all_residual_inputs = []

        for b in range(bsz):
            for f in range(frames):
                valid_mask = sparse_mask[b, f]
                if not valid_mask.any():
                    continue

                valid_tokens = sparse_tokens[b, f, valid_mask]
                valid_indices = sparse_indices[b, f, valid_mask].long()
                batch_id = b * frames + f
                batch_idx = torch.full((valid_tokens.size(0), 1), batch_id,
                                       dtype=torch.int32, device=device)
                coords = torch.cat([batch_idx, valid_indices.int()], dim=1)

                lin = (batch_id * frame_stride
                       + valid_indices[:, 0] * stride_r
                       + valid_indices[:, 1] * stride_a
                       + valid_indices[:, 2])
                out_pos = batch_id * max_tokens + valid_mask.nonzero(as_tuple=True)[0].long()

                all_feats.append(valid_tokens)
                all_coords.append(coords)
                all_in_lin.append(lin)
                all_out_pos.append(out_pos)
                all_residual_inputs.append(valid_tokens)

        output = torch.zeros((bsz, frames, max_tokens, self.spconv_out_channels),
                             device=device, dtype=sparse_tokens.dtype)
        if not all_feats:
            return output

        all_feats_cat = torch.cat(all_feats, dim=0)
        all_coords_cat = torch.cat(all_coords, dim=0)
        all_in_lin_cat = torch.cat(all_in_lin, dim=0)
        all_out_pos_cat = torch.cat(all_out_pos, dim=0)
        residual_inputs = torch.cat(all_residual_inputs, dim=0)

        sparse_tensor = spconv.SparseConvTensor(
            features=all_feats_cat,
            indices=all_coords_cat,
            spatial_shape=spatial_shape,
            batch_size=bsz * frames
        )

        radial_out = self.radial_spconv(sparse_tensor)
        angular_out = self.angular_spconv(sparse_tensor)
        branch_outputs = [radial_out, angular_out]
        if self.use_isotropic_branch:
            branch_outputs.append(self.isotropic_spconv(sparse_tensor))

        out_indices = radial_out.indices.long()
        out_lin = (out_indices[:, 0] * frame_stride
                   + out_indices[:, 1] * stride_r
                   + out_indices[:, 2] * stride_a
                   + out_indices[:, 3])
        sorted_out_lin, sort_idx = torch.sort(out_lin)

        matched_branches = [
            self._match_sparse_features(branch_out, all_in_lin_cat, sort_idx, sorted_out_lin)
            for branch_out in branch_outputs
        ]
        branch_concat = torch.cat(matched_branches, dim=1)
        fused = self.branch_fusion(branch_concat)
        fused = fused + self.stem_residual(residual_inputs)

        output_flat = output.view(bsz * frames * max_tokens, self.spconv_out_channels)
        if fused.dtype != output_flat.dtype:
            fused = fused.to(output_flat.dtype)
        output_flat.index_copy_(0, all_out_pos_cat, fused)
        return output

    def _apply_physics_fallback(self,
                                tokens: torch.Tensor,
                                sparse_indices: torch.Tensor,
                                sparse_shape: torch.Tensor) -> torch.Tensor:
        if isinstance(sparse_shape, torch.Tensor):
            if sparse_shape.dim() > 1:
                spatial_shape_t = sparse_shape[0].to(device=tokens.device, dtype=torch.long)
            else:
                spatial_shape_t = sparse_shape.to(device=tokens.device, dtype=torch.long)
        else:
            spatial_shape_t = torch.tensor(sparse_shape, device=tokens.device, dtype=torch.long)

        flat_tokens = tokens.reshape(-1, tokens.size(-1))
        branches = [
            self.radial_linear(flat_tokens),
            self.angular_linear(flat_tokens),
        ]
        if self.use_isotropic_branch:
            branches.append(self.isotropic_linear(flat_tokens))
        branch_concat = torch.cat(branches, dim=1)
        fused = self.branch_fusion(branch_concat)
        fused = fused + self.stem_residual(flat_tokens)
        return fused.view(*tokens.shape[:3], self.spconv_out_channels)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if (not self.use_spconv or not HAS_SPCONV) and 'mmwave_sparse_tokens' in inputs:
            inputs = dict(inputs)
            inputs['mmwave_sparse_tokens'] = self._apply_physics_fallback(
                inputs['mmwave_sparse_tokens'],
                inputs['mmwave_sparse_indices'],
                inputs['mmwave_sparse_shape'])
            return super().forward(inputs)
        return super().forward(inputs)
