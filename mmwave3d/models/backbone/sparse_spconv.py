from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    import spconv.pytorch as spconv
except Exception as exc:  # pragma: no cover - import guard for optional dep
    spconv = None
    _spconv_import_error = exc

from mmwave3d.registry import MODELS


@MODELS.register_module()
class MmWaveSpConvTransformerBackbone(nn.Module):
    """Sparse conv + transformer backbone for mmWave voxel tokens."""

    def __init__(self,
                 input_token_dim: int = 3,
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
                 spconv_channels: Sequence[int] = (64, 64),
                 spconv_kernel_size: int = 3,
                 # Learnable downsampling parameters
                 enable_learnable_downsample: bool = False,
                 downsample_tokens: Optional[int] = None,
                 downsample_ratio: float = 0.5,
                 downsample_method: str = 'attention') -> None:
        super().__init__()
        if spconv is None:
            raise ImportError(
                'spconv is required for MmWaveSpConvTransformerBackbone. '
                'Install spconv and try again.') from _spconv_import_error

        if not spconv_channels:
            raise ValueError('spconv_channels must contain at least one channel.')
        self.input_token_dim = input_token_dim
        self.spconv_out_dim = int(spconv_channels[-1])
        self.enable_learnable_downsample = enable_learnable_downsample
        self.downsample_tokens = downsample_tokens
        self.downsample_ratio = downsample_ratio
        self.downsample_method = downsample_method

        padding = spconv_kernel_size // 2
        layers = []
        in_channels = input_token_dim
        for out_channels in spconv_channels:
            layers.append(
                spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=spconv_kernel_size,
                    padding=padding,
                    bias=False))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.sparse_encoder = spconv.SparseSequential(*layers)

        self.token_embed = nn.Linear(self.spconv_out_dim, embed_dim)
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
        
        # Learnable downsampling module (optional)
        if self.enable_learnable_downsample:
            if downsample_method == 'attention':
                if self.downsample_tokens is None:
                    raise ValueError(
                        'downsample_tokens must be set for attention-based downsampling.')
                # Attention-based downsampling runs before token embedding.
                self.downsample_query = nn.Parameter(
                    torch.randn(1, self.downsample_tokens, self.spconv_out_dim))
                self.downsample_attn = nn.MultiheadAttention(
                    self.spconv_out_dim, num_heads=4, dropout=dropout, batch_first=True)
            elif downsample_method == 'mlp':
                # MLP-based token scoring
                self.downsample_score = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dim // 2, 1),
                    nn.Sigmoid()
                )
            else:
                raise ValueError(f"Unsupported downsample_method: {downsample_method}")

    def _get_downsample_target(self, seq_len: int) -> int:
        if self.downsample_tokens is not None:
            return max(1, min(self.downsample_tokens, seq_len))
        return max(int(seq_len * self.downsample_ratio), 1)

    def _apply_learnable_downsample(self, tokens, pos_grid, mask):
        """Apply learnable downsampling to reduce token count.
        
        Args:
            tokens: (B, F, N, C) - input tokens
            pos_grid: (B, F, N, 3) - input positions
            mask: (B, F, N) - valid token mask
            
        Returns:
            downsampled_tokens: (B, F, N', C)
            downsampled_pos: (B, F, N', 3)
            downsampled_mask: (B, F, N')
        """
        B, F, N, C = tokens.shape
        device = tokens.device
        
        # Target number of tokens after downsampling
        target_n = self._get_downsample_target(N)
        
        if self.downsample_method == 'attention':
            # Attention-based selection: use learnable query to attend to input tokens
            out_tokens = []
            out_pos = []
            out_masks = []
            
            for b in range(B):
                for f in range(F):
                    frame_tokens = tokens[b, f]  # (N, C)
                    frame_pos = pos_grid[b, f]  # (N, 3)
                    frame_mask = mask[b, f]  # (N,)
                    
                    if not frame_mask.any():
                        # No valid tokens
                        sampled = torch.zeros(target_n, C, device=device)
                        sampled_pos = torch.zeros(target_n, 3, device=device)
                        sampled_mask = torch.zeros(target_n, dtype=torch.bool, device=device)
                    else:
                        # Apply attention: query attends to all tokens
                        query = self.downsample_query[:, :target_n, :]  # (1, target_n, C)
                        key_value = frame_tokens.unsqueeze(0)  # (1, N, C)
                        
                        # Create attention mask (mask out invalid tokens)
                        key_padding_mask = ~frame_mask.unsqueeze(0)  # (1, N)
                        
                        # Multihead attention
                        sampled, attn_weights = self.downsample_attn(
                            query, key_value, key_value,
                            key_padding_mask=key_padding_mask,
                            need_weights=True
                        )  # (1, target_n, C)
                        sampled = sampled.squeeze(0)  # (target_n, C)
                        sampled_pos = torch.bmm(attn_weights, frame_pos.unsqueeze(0)).squeeze(0)
                        sampled_mask = torch.ones(target_n, dtype=torch.bool, device=device)
                    
                    out_tokens.append(sampled)
                    out_pos.append(sampled_pos)
                    out_masks.append(sampled_mask)
            
            # Reshape back
            out_tokens = torch.stack(out_tokens).view(B, F, target_n, C)
            out_pos = torch.stack(out_pos).view(B, F, target_n, 3)
            out_masks = torch.stack(out_masks).view(B, F, target_n)
            
        elif self.downsample_method == 'mlp':
            # MLP-based scoring: score each token and keep top-k
            out_tokens = []
            out_pos = []
            out_masks = []
            
            for b in range(B):
                for f in range(F):
                    frame_tokens = tokens[b, f]  # (N, C)
                    frame_pos = pos_grid[b, f]  # (N, 3)
                    frame_mask = mask[b, f]  # (N,)
                    
                    if not frame_mask.any():
                        sampled = torch.zeros(
                            target_n, C, device=device, dtype=frame_tokens.dtype)
                        sampled_pos = torch.zeros(
                            target_n, 3, device=device, dtype=frame_pos.dtype)
                        sampled_mask = torch.zeros(target_n, dtype=torch.bool, device=device)
                    else:
                        # Score tokens
                        scores = self.downsample_score(frame_tokens).squeeze(-1)  # (N,)
                        scores = scores.masked_fill(~frame_mask, -1e9)
                        
                        # Select top-k
                        _, top_indices = torch.topk(
                            scores, min(target_n, int(frame_mask.sum().item())))
                        sampled = frame_tokens[top_indices]  # (target_n, C)
                        sampled_pos = frame_pos[top_indices]
                        
                        # Pad if necessary
                        if sampled.size(0) < target_n:
                            pad = torch.zeros(
                                target_n - sampled.size(0), C, device=device, dtype=sampled.dtype)
                            pad_pos = torch.zeros(
                                target_n - sampled_pos.size(0), 3, device=device, dtype=sampled_pos.dtype)
                            sampled = torch.cat([sampled, pad], dim=0)
                            sampled_pos = torch.cat([sampled_pos, pad_pos], dim=0)
                            sampled_mask = torch.zeros(target_n, dtype=torch.bool, device=device)
                            sampled_mask[:top_indices.size(0)] = True
                        else:
                            sampled_mask = torch.ones(target_n, dtype=torch.bool, device=device)
                    
                    out_tokens.append(sampled)
                    out_pos.append(sampled_pos)
                    out_masks.append(sampled_mask)
            
            out_tokens = torch.stack(out_tokens).view(B, F, target_n, C)
            out_pos = torch.stack(out_pos).view(B, F, target_n, 3)
            out_masks = torch.stack(out_masks).view(B, F, target_n)
        
        return out_tokens, out_pos, out_masks

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tokens = inputs['mmwave_sparse_tokens']  # (B, F, N, C)
        pos_grid = inputs['mmwave_sparse_pos']  # (B, F, N, 3)
        valid_mask = inputs['mmwave_sparse_mask']  # (B, F, N)
        indices = inputs['mmwave_sparse_indices']  # (B, F, N, 3)
        frame_lin = inputs['mmwave_sparse_lin']  # (B, F, N)
        spatial_shape = inputs['mmwave_sparse_shape']  # (3,)

        bsz, frames, seq_len, _ = tokens.shape
        device = tokens.device

        coords_list = []
        feats_list = []
        for b in range(bsz):
            for f in range(frames):
                mask = valid_mask[b, f]
                if not mask.any():
                    continue
                coords = indices[b, f][mask].to(dtype=torch.int32)
                batch_idx = torch.full(
                    (coords.shape[0], 1),
                    b * frames + f,
                    device=device,
                    dtype=torch.int32)
                coords4 = torch.cat([batch_idx, coords], dim=1)
                feats = tokens[b, f][mask]
                coords_list.append(coords4)
                feats_list.append(feats)

        if not coords_list:
            out_tokens = tokens.new_zeros((bsz, frames, seq_len, self.spconv_out_dim))
            out_mask = valid_mask.clone()
            return self._run_transformer(out_tokens, pos_grid, out_mask)

        coords = torch.cat(coords_list, dim=0)
        feats = torch.cat(feats_list, dim=0)
        out_mask = valid_mask.clone()
        sparse_tensor = spconv.SparseConvTensor(
            feats, coords, spatial_shape=spatial_shape.tolist(), batch_size=bsz * frames)
        sparse_out = self.sparse_encoder(sparse_tensor)

        out_coords = sparse_out.indices
        out_feats = sparse_out.features
        out_tokens = out_feats.new_zeros((bsz, frames, seq_len, self.spconv_out_dim))
        if out_coords.numel() > 0:
            a_stride = int(spatial_shape[1].item() * spatial_shape[2].item())
            e_stride = int(spatial_shape[2].item())
            r_idx = out_coords[:, 1]
            a_idx = out_coords[:, 2]
            e_idx = out_coords[:, 3]
            lin_out = r_idx * a_stride + a_idx * e_stride + e_idx
            bf = out_coords[:, 0]
            b_idx = bf // frames
            f_idx = bf % frames
            for b in range(bsz):
                sel = b_idx == b
                if not sel.any():
                    continue
                lin_b = lin_out[sel]
                feats_b = out_feats[sel]
                f_b = f_idx[sel]
                for f in range(frames):
                    sel_f = f_b == f
                    if not sel_f.any():
                        continue
                    lin_f = lin_b[sel_f]
                    feats_f = feats_b[sel_f]
                    lin_frame = frame_lin[b, f]
                    pos = torch.searchsorted(lin_frame, lin_f)
                    out_tokens[b, f, pos] = feats_f
                    out_mask[b, f, pos] = True

        # Apply learnable downsampling if enabled
        if self.enable_learnable_downsample:
            out_tokens, pos_grid, out_mask = self._apply_learnable_downsample(
                out_tokens, pos_grid, out_mask)

        return self._run_transformer(out_tokens, pos_grid, out_mask)

    def _encode_pos(self, pos: torch.Tensor) -> torch.Tensor:
        """Apply cosine positional encoding before linear projection."""
        if self.pos_freq_bands is None:
            return pos
        freq = self.pos_freq_bands.to(device=pos.device, dtype=pos.dtype)
        freq_shape = (1,) * (pos.dim() - 1) + (freq.numel(), 1)
        pos_expanded = pos.unsqueeze(-2) * freq.view(*freq_shape)
        cos_feats = torch.cos(pos_expanded)
        cos_feats = cos_feats.reshape(*pos.shape[:-1], -1)
        return torch.cat([pos, cos_feats], dim=-1)

    def _run_transformer(self,
                         tokens: torch.Tensor,
                         pos_grid: torch.Tensor,
                         valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, frames, seq_len, _ = tokens.shape
        device = tokens.device

        frame_coord = torch.arange(
            frames, device=device, dtype=tokens.dtype)
        frame_coord = frame_coord.view(1, frames, 1, 1).expand(bsz, frames, seq_len, 1)
        pos = torch.cat([pos_grid.to(dtype=tokens.dtype), frame_coord], dim=-1)

        token_feat = self.token_embed(tokens)
        pos_enc = self._encode_pos(pos)
        pos_feat = self.pos_embed(pos_enc)
        x = token_feat + pos_feat  # (B, F, L, C)

        x_spatial = x.view(bsz * frames, seq_len, -1).permute(1, 0, 2)
        spatial_mask = ~valid_mask.view(bsz * frames, seq_len)
        x_spatial = self.spatial_encoder(
            x_spatial, src_key_padding_mask=spatial_mask)
        x_spatial = x_spatial.permute(1, 0, 2).view(bsz, frames, seq_len, -1)

        x_temp = x_spatial.permute(0, 2, 1, 3)
        x_temp = x_temp.reshape(bsz * seq_len, frames, -1).permute(1, 0, 2)
        temporal_mask = ~valid_mask.permute(0, 2, 1).reshape(bsz * seq_len, frames)
        x_temp = self.temporal_encoder(
            x_temp, src_key_padding_mask=temporal_mask)
        x_temp = x_temp.permute(1, 0, 2).reshape(bsz, seq_len, frames, -1)

        x_out = x_temp.permute(0, 2, 1, 3).reshape(bsz, frames * seq_len, -1)
        pos_flat = pos.reshape(bsz, frames * seq_len, -1)
        outputs = dict(features=x_out, positions=pos_flat)
        return outputs
