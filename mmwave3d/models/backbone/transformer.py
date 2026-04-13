from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmwave3d.registry import MODELS

@MODELS.register_module()
class MmWaveTransformerBackbone(nn.Module):
    """Transformer encoder with per-frame spatial attention + temporal fusion.
    
    Args:
        learnable_downsample_stages: Number of learnable 3D conv+pool stages to apply
            to voxel grids before tokenization. Each stage uses 3D conv with learnable
            weights followed by pooling. Set to 0 to disable.
        learnable_downsample_channels: Output channels for each downsample stage.
            If int, same channels for all stages. If list, one per stage.
        learnable_downsample_kernel: Kernel size for 3D conv in each stage.
        learnable_downsample_stride: Stride for pooling in each stage.
        learnable_downsample_pool: 'max' or 'avg' pooling type.
    """

    def __init__(self,
                 token_dim: int = 1,
                 pos_dim: int = 4,
                 embed_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ffn_dim: int = 1024,
                 dropout: float = 0.1,
                 norm_first: bool = True,
                 temporal_layers: int = 2,
                 temporal_heads: Optional[int] = None,
                 temporal_ffn_dim: Optional[int] = None,
                 temporal_dropout: Optional[float] = None,
                 pos_freq_bands: Optional[Sequence[float]] = (1.0, 2.0, 4.0, 8.0),
                 learnable_downsample_stages: int = 0,
                 learnable_downsample_channels: Union[int, Sequence[int]] = 32,
                 learnable_downsample_kernel: Union[int, Tuple[int, int, int]] = 3,
                 learnable_downsample_stride: Union[int, Tuple[int, int, int]] = 2,
                 learnable_downsample_pool: str = 'max') -> None:
        super().__init__()
        
        # Learnable downsampling (3D conv + pool stages)
        self.learnable_downsample_stages = learnable_downsample_stages
        self.learnable_downsample_pool = learnable_downsample_pool
        if learnable_downsample_stages > 0:
            if isinstance(learnable_downsample_kernel, int):
                learnable_downsample_kernel = (learnable_downsample_kernel,) * 3
            if isinstance(learnable_downsample_stride, int):
                learnable_downsample_stride = (learnable_downsample_stride,) * 3
            if isinstance(learnable_downsample_channels, int):
                channels = [learnable_downsample_channels] * learnable_downsample_stages
            else:
                channels = list(learnable_downsample_channels)
                assert len(channels) == learnable_downsample_stages
            
            self.downsample_convs = nn.ModuleList()
            self.downsample_bns = nn.ModuleList()
            in_ch = token_dim
            for ch in channels:
                self.downsample_convs.append(
                    nn.Conv3d(in_ch, ch, kernel_size=learnable_downsample_kernel,
                             padding=tuple(k // 2 for k in learnable_downsample_kernel)))
                self.downsample_bns.append(nn.BatchNorm3d(ch))
                in_ch = ch
            
            # Update token_dim after downsampling
            self.downsample_out_channels = in_ch
            self.downsample_stride = learnable_downsample_stride
            self.downsample_kernel = learnable_downsample_kernel
        else:
            self.downsample_out_channels = token_dim
        
        self.token_embed = nn.Linear(self.downsample_out_channels, embed_dim)
        if pos_freq_bands is None:
            self.register_buffer('pos_freq_bands', None, persistent=False)
            pos_in_dim = pos_dim
        else:
            if isinstance(pos_freq_bands, torch.Tensor):
                freq_tensor = pos_freq_bands.clone().detach().float()
            else:
                freq_tensor = torch.tensor(pos_freq_bands, dtype=torch.float32)
            self.register_buffer('pos_freq_bands', freq_tensor, persistent=False)
            freq_count = int(freq_tensor.numel())
            pos_in_dim = pos_dim * (freq_count + 1)
        self.pos_embed = nn.Linear(pos_in_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=False,
            norm_first=norm_first)
        self.spatial_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=temporal_heads or num_heads,
            dim_feedforward=temporal_ffn_dim or ffn_dim,
            dropout=temporal_dropout if temporal_dropout is not None else dropout,
            batch_first=False,
            norm_first=norm_first)
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_encoder_layer, num_layers=temporal_layers)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode mmWave voxels with spatial + temporal attention."""
        if 'mmwave_sparse_tokens' in inputs:
            tokens = inputs['mmwave_sparse_tokens']  # (B, F, N, C)
            pos_grid = inputs['mmwave_sparse_pos']  # (B, F, N, 3)
            valid_mask = inputs.get('mmwave_sparse_mask', None)
            bsz, frames, seq_len, _ = tokens.shape
            if valid_mask is None:
                valid_mask = torch.ones(
                    (bsz, frames, seq_len),
                    device=tokens.device,
                    dtype=torch.bool)
        else:
            voxels = inputs['mmwave_voxels']  # (B, F, R, A, E)
            pos_grid = inputs['mmwave_pos_grid']  # (B, F, R, A, E, 3)
            bsz, frames, r, a, e = voxels.shape
            
            # Apply learnable downsampling if configured
            if self.learnable_downsample_stages > 0:
                voxels, pos_grid = self._apply_learnable_downsample(
                    voxels, pos_grid, (r, a, e))
                # Update dimensions after downsampling
                bsz, frames, r, a, e = voxels.shape
            
            seq_len = r * a * e
            tokens = voxels.view(bsz, frames, seq_len, -1)
            pos_grid = pos_grid.view(bsz, frames, seq_len, -1)
            valid_mask = torch.ones(
                (bsz, frames, seq_len),
                device=voxels.device,
                dtype=torch.bool)

        frame_coord = torch.arange(
            frames, device=tokens.device, dtype=tokens.dtype)
        frame_coord = frame_coord.view(1, frames, 1, 1).expand(bsz, frames, seq_len, 1)
        pos = torch.cat([pos_grid, frame_coord], dim=-1)

        token_feat = self.token_embed(tokens)
        pos_enc = self._encode_pos(pos)
        pos_feat = self.pos_embed(pos_enc)
        x = token_feat + pos_feat  # (B, F, L, C)

        # Spatial self-attention within each frame
        x_spatial = x.view(bsz * frames, seq_len, -1).permute(1, 0, 2)
        spatial_mask = ~valid_mask.view(bsz * frames, seq_len)
        x_spatial = self.spatial_encoder(
            x_spatial, src_key_padding_mask=spatial_mask)
        x_spatial = x_spatial.permute(1, 0, 2).view(bsz, frames, seq_len, -1)

        # Temporal attention across frames for each voxel location
        x_temp = x_spatial.permute(0, 2, 1, 3)  # (B, L, F, C)
        x_temp = x_temp.reshape(bsz * seq_len, frames, -1).permute(1, 0, 2)
        temporal_mask = ~valid_mask.permute(0, 2, 1).reshape(bsz * seq_len, frames)
        x_temp = self.temporal_encoder(
            x_temp, src_key_padding_mask=temporal_mask)
        x_temp = x_temp.permute(1, 0, 2).reshape(bsz, seq_len, frames, -1)

        x_out = x_temp.permute(0, 2, 1, 3).reshape(bsz, frames * seq_len, -1)
        pos_flat = pos.reshape(bsz, frames * seq_len, -1)
        outputs = dict(features=x_out, positions=pos_flat)
        return outputs

    def _encode_pos(self, pos: torch.Tensor) -> torch.Tensor:
        """Apply cosine positional encoding before linear projection."""
        if self.pos_freq_bands is None:
            return pos
        freq = self.pos_freq_bands.to(device=pos.device, dtype=pos.dtype)
        # Broadcast freq over the last dim, keeping all leading dims intact.
        freq_shape = (1,) * (pos.dim() - 1) + (freq.numel(), 1)
        pos_expanded = pos.unsqueeze(-2) * freq.view(*freq_shape)
        cos_feats = torch.cos(pos_expanded)
        cos_feats = cos_feats.reshape(*pos.shape[:-1], -1)
        return torch.cat([pos, cos_feats], dim=-1)

    def _apply_learnable_downsample(
            self, voxels: torch.Tensor, pos_grid: torch.Tensor,
            original_shape: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply learnable 3D conv+pool downsampling to voxel grids.
        
        Args:
            voxels: (B, F, R, A, E) voxel features
            pos_grid: (B, F, R, A, E, 3) position grid
            original_shape: (R, A, E) original grid dimensions
            
        Returns:
            downsampled_voxels: (B, F, R', A', E') downsampled features
            downsampled_pos: (B, F, R', A', E', 3) downsampled positions
        """
        bsz, frames, r, a, e = voxels.shape
        # Reshape to (B*F, C, R, A, E) for Conv3d
        x = voxels.view(bsz * frames, -1, r, a, e)
        
        # Apply conv+bn+relu+pool stages
        for i in range(self.learnable_downsample_stages):
            x = self.downsample_convs[i](x)
            x = self.downsample_bns[i](x)
            x = F.relu(x, inplace=True)
            if self.learnable_downsample_pool == 'max':
                x = F.max_pool3d(x, kernel_size=self.downsample_stride,
                                stride=self.downsample_stride)
            else:
                x = F.avg_pool3d(x, kernel_size=self.downsample_stride,
                                stride=self.downsample_stride)
        
        # Reshape back to (B, F, R', A', E', C)
        _, c_out, r_new, a_new, e_new = x.shape
        x = x.view(bsz, frames, c_out, r_new, a_new, e_new)
        x = x.permute(0, 1, 3, 4, 5, 2)  # (B, F, R', A', E', C)
        
        # Downsample position grid accordingly
        # Use stride to subsample positions
        stride_r, stride_a, stride_e = self.downsample_stride
        total_stride_r = stride_r ** self.learnable_downsample_stages
        total_stride_a = stride_a ** self.learnable_downsample_stages
        total_stride_e = stride_e ** self.learnable_downsample_stages
        
        pos_downsampled = pos_grid[
            :, :,
            ::total_stride_r,
            ::total_stride_a,
            ::total_stride_e,
            :
        ]
        
        # Ensure shape consistency
        pos_downsampled = pos_downsampled[:, :, :r_new, :a_new, :e_new, :]
        
        return x, pos_downsampled
