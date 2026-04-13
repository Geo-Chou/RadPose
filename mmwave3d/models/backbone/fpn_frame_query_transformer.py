"""
Multi-scale Transformer with FPN and Frame Query mechanism.

Features:
1. Multi-scale feature extraction with FPN
2. Frame-level temporal attention (not position-wise)
3. Center frame query for target frame prediction
4. Cross-attention for temporal aggregation
"""

from typing import Dict, Optional, Sequence, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import MMLogger
from mmwave3d.utils import FlashMultiheadAttention

try:
    import spconv.pytorch as spconv
    HAS_SPCONV = True
    # print("✅ SpConv imported successfully")
except ImportError:
    spconv = None
    HAS_SPCONV = False

from mmwave3d.registry import MODELS


_SPCONV_STATUS_MESSAGES_EMITTED = set()


def _emit_spconv_status_once(status: str, message: str, level: str = 'warning') -> None:
    if status in _SPCONV_STATUS_MESSAGES_EMITTED:
        return
    _SPCONV_STATUS_MESSAGES_EMITTED.add(status)

    logger = MMLogger.get_current_instance()
    if logger is not None:
        log_fn = getattr(logger, level, logger.info)
        log_fn(message)
        return

    if level == 'warning':
        warnings.warn(message, stacklevel=2)
    else:
        print(message)


@MODELS.register_module()
class FPNFrameQueryTransformer(nn.Module):
    """
    Transformer backbone with FPN and Frame Query mechanism.
    
    Architecture Overview:
    1. SpConv feature extraction (multi-scale)
    2. FPN for multi-scale feature fusion
    3. Spatial self-attention per frame
    4. Frame-level temporal fusion (all tokens across frames)
    5. Center frame query for target prediction
    
    Args:
        input_token_dim: Input feature dimension per voxel
        pos_dim: Positional encoding dimension
        embed_dim: Transformer embedding dimension
        num_layers: Number of spatial transformer layers
        num_heads: Number of attention heads
        ffn_dim: Feed-forward network dimension
        dropout: Dropout rate
        fpn_channels: Channel sizes for FPN levels (default: [64, 128, 256])
        temporal_layers: Number of frame-level temporal attention layers
        num_query_tokens: Number of learnable query tokens for center frame
        downsample_tokens: Target number of tokens after downsampling
        norm_first: Use pre-norm or post-norm
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
                 pos_freq_bands: Optional[Sequence[float]] = (1.0, 2.0, 4.0, 8.0)) -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_query_tokens = num_query_tokens
        self.downsample_tokens = downsample_tokens
        self.downsample_method = downsample_method
        self.non_center_downsample_tokens = non_center_downsample_tokens
        
        # Check if SpConv is available and CUDA is available
        self.use_spconv = HAS_SPCONV and torch.cuda.is_available()
        
        # SpConv feature extraction
        # Note: For dense voxelization, input is occupancy map (1 channel)
        # For sparse tokens, input_token_dim is used (count + features)
        self.spconv_in_channels = input_token_dim if self.use_spconv else input_token_dim
        self.spconv_out_channels = spconv_channels[-1] if spconv_channels else input_token_dim
        self.spconv_kernel_size = spconv_kernel_size
        
        if self.use_spconv:
            # Real SpConv for local spatial feature extraction (CUDA only)
            # Input: sparse tokens with features (count + optional point features)
            spconv_layers = []
            in_ch = input_token_dim  # For sparse tokens
            for out_ch in spconv_channels:
                spconv_layers.extend([
                    spconv.SubMConv3d(
                        in_ch, out_ch,
                        kernel_size=spconv_kernel_size,
                        padding=spconv_kernel_size // 2,
                        bias=False,
                        indice_key=f'subm{len(spconv_layers)}'
                    ),
                    nn.BatchNorm1d(out_ch),  # Changed from SyncBatchNorm: avoid cross-GPU sync overhead
                    nn.ReLU(inplace=True)
                ])
                in_ch = out_ch
            self.spconv_layers = spconv.SparseSequential(*spconv_layers)
            _emit_spconv_status_once(
                'spconv_enabled',
                'Using SpConv with BatchNorm1d (no cross-GPU sync) for faster training',
                level='info')
        
        # Always create linear projection as fallback for sparse tokens
        # This handles the case where use_spconv=True but input is sparse_tokens format
        if not self.use_spconv or input_token_dim != self.spconv_out_channels:
            self.feature_proj = nn.Linear(input_token_dim, self.spconv_out_channels)
            if not self.use_spconv:
                if not HAS_SPCONV:
                    _emit_spconv_status_once(
                        'spconv_missing',
                        'SpConv not available, using Linear projection')
                else:
                    _emit_spconv_status_once(
                        'spconv_cuda_unavailable',
                        'CUDA not available, using Linear projection for SpConv')
        
        # Token embedding: spconv_out_channels -> embed_dim
        self.token_embed = nn.Linear(self.spconv_out_channels, embed_dim)
        
        # Positional encoding
        if pos_freq_bands is None:
            self.register_buffer('pos_freq_bands', None, persistent=False)
            pos_in_dim = pos_dim
        else:
            freq_tensor = torch.tensor(pos_freq_bands, dtype=torch.float32)
            self.register_buffer('pos_freq_bands', freq_tensor, persistent=False)
            pos_in_dim = pos_dim * (len(pos_freq_bands) + 1)
        
        self.pos_embed = nn.Linear(pos_in_dim, embed_dim)
        
        # FPN feature extraction at multiple scales
        self.fpn_layers = nn.ModuleList()
        for ch in fpn_channels:
            self.fpn_layers.append(nn.Sequential(
                nn.Linear(embed_dim, ch),
                nn.LayerNorm(ch),
                nn.ReLU(inplace=True)
            ))
        
        # FPN fusion
        self.fpn_fusion = nn.ModuleList()
        for ch in fpn_channels:
            self.fpn_fusion.append(nn.Linear(ch, embed_dim))
        
        # Learnable downsampling
        if downsample_method == 'attention':
            self.downsample_queries = nn.Parameter(
                torch.randn(1, downsample_tokens, embed_dim).contiguous()
            )
            self.downsample_attn = FlashMultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
        
        # Spatial transformer encoder (per-frame)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=False,
            norm_first=norm_first
        )
        self.spatial_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Frame-level temporal attention
        # Key insight: Let tokens attend to ALL other frames' tokens, not just same position
        self.temporal_cross_attn = nn.ModuleList()
        for _ in range(temporal_layers):
            self.temporal_cross_attn.append(FlashMultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            ))
            
        self.temporal_ffn = nn.ModuleList()
        for _ in range(temporal_layers):
            self.temporal_ffn.append(nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
                nn.Dropout(dropout)
            ))
        
        self.temporal_norm1 = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(temporal_layers)
        ])
        self.temporal_norm2 = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(temporal_layers)
        ])
        
        # Center frame query tokens (learnable)
        self.center_frame_query = nn.Parameter(
            torch.randn(1, num_query_tokens, embed_dim).contiguous()
        )
        
        # Frame position embedding for query (to specify target frame)
        # Frame pos is 1D, so after encoding: 1 * (len(freq_bands) + 1)
        frame_pos_in_dim = 1 * (len(pos_freq_bands) + 1) if pos_freq_bands else 1
        self.frame_pos_embed = nn.Linear(frame_pos_in_dim, embed_dim)
        
        # Query cross-attention to aggregate multi-frame information
        self.query_cross_attn = FlashMultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with FPN and frame query mechanism.
        
        Args:
            inputs: Dict with 'mmwave_voxels', 'mmwave_pos_grid', or 'mmwave_sparse_tokens'
            
        Returns:
            Dict with 'features' (B, C) - global pooled features for head
        """
        # import time
        # torch.cuda.synchronize()
        # t_start = time.time()
        
        # Extract tokens and positions
        if 'mmwave_sparse_tokens' in inputs:
            tokens = inputs['mmwave_sparse_tokens']  # (B, F, N, C)
            pos_grid = inputs['mmwave_sparse_pos']  # (B, F, N, 3)
            bsz, frames, seq_len, _ = tokens.shape
            valid_mask = inputs.get('mmwave_sparse_mask', None)
            if valid_mask is None:
                valid_mask = torch.ones(
                    (bsz, frames, seq_len),
                    device=tokens.device,
                    dtype=torch.bool)
            else:
                valid_mask = valid_mask.to(device=tokens.device, dtype=torch.bool)
        else:
            voxels = inputs['mmwave_voxels']  # (B, F, R, A, E)
            pos_grid = inputs['mmwave_pos_grid']  # (B, F, R, A, E, 3)
            bsz, frames, r, a, e = voxels.shape
            seq_len = r * a * e
            tokens = voxels.view(bsz, frames, seq_len, -1)
            pos_grid = pos_grid.view(bsz, frames, seq_len, -1)
            valid_mask = torch.ones(
                (bsz, frames, seq_len),
                device=tokens.device,
                dtype=torch.bool)
        
        # torch.cuda.synchronize()
        # t_extract = time.time() - t_start
        
        # Add frame coordinate to position
        frame_coord = torch.arange(frames, device=tokens.device, dtype=tokens.dtype)
        frame_coord = frame_coord.view(1, frames, 1, 1).expand(bsz, frames, seq_len, 1)
        pos = torch.cat([pos_grid, frame_coord], dim=-1)
        
        # # Feature projection with SpConv or Linear
        # torch.cuda.synchronize()
        # t_proj_start = time.time()
        
        if self.use_spconv and HAS_SPCONV and 'mmwave_sparse_indices' in inputs:
            # Use SpConv for sparse token feature extraction
            tokens_proj = self._apply_spconv_sparse(
                inputs['mmwave_sparse_tokens'],
                inputs['mmwave_sparse_indices'],
                inputs['mmwave_sparse_shape'],
                inputs['mmwave_sparse_mask']
            )
        elif self.use_spconv and HAS_SPCONV and 'mmwave_voxels' in inputs:
            # Use SpConv for dense voxel feature extraction
            tokens_proj = self._apply_spconv(inputs['mmwave_voxels'], 
                                            inputs.get('mmwave_coords', None))
        elif hasattr(self, 'feature_proj'):
            # Fallback: Linear projection
            tokens_proj = self.feature_proj(tokens)  # (B, F, L, spconv_out)
        else:
            # Direct pass-through if spconv_out == input_token_dim
            tokens_proj = tokens
        
        # torch.cuda.synchronize()
        # t_spconv = time.time() - t_proj_start
        
        # # Token embedding
        # t_embed_start = time.time()
        token_feat = self.token_embed(tokens_proj)  # (B, F, L, embed_dim)
        pos_enc = self._encode_pos(pos)
        pos_feat = self.pos_embed(pos_enc)
        x = token_feat + pos_feat
        # torch.cuda.synchronize()
        # t_embed = time.time() - t_embed_start
        
        # # Multi-scale feature extraction with FPN
        # torch.cuda.synchronize()
        # t_fpn_start = time.time()
        fpn_features = []
        for fpn_layer in self.fpn_layers:
            fpn_feat = fpn_layer(x)  # (B, F, L, fpn_ch)
            fpn_features.append(fpn_feat)
        
        # Fuse FPN features
        fused_features = []
        for fpn_feat, fusion_layer in zip(fpn_features, self.fpn_fusion):
            fused = fusion_layer(fpn_feat)  # (B, F, L, C)
            fused_features.append(fused)
        
        # Combine multi-scale features
        x = sum(fused_features) / len(fused_features)  # Average fusion
        # torch.cuda.synchronize()
        # t_fpn = time.time() - t_fpn_start
        
        # # Learnable downsampling per frame
        # torch.cuda.synchronize()
        # t_down_start = time.time()
        if self.downsample_method == 'attention' and seq_len > self.downsample_tokens:
            x_down = []
            mask_down = []
            for f in range(frames):
                frame_tokens = x[:, f, :, :]  # (B, L, C)
                frame_mask = valid_mask[:, f, :]  # (B, L)
                queries = self.downsample_queries.expand(bsz, -1, -1)  # (B, N_down, C)
                down_tokens, _ = self.downsample_attn(
                    query=queries,
                    key=frame_tokens,
                    value=frame_tokens,
                    key_padding_mask=self._to_key_padding_mask(frame_mask)
                )
                x_down.append(down_tokens)
                down_mask = frame_mask.any(dim=1, keepdim=True).expand(
                    -1, self.downsample_tokens)
                mask_down.append(down_mask)
            x = torch.stack(x_down, dim=1)  # (B, F, N_down, C)
            valid_mask = torch.stack(mask_down, dim=1)  # (B, F, N_down)
            seq_len = self.downsample_tokens
        # torch.cuda.synchronize()
        # t_downsample = time.time() - t_down_start
        
        # # Spatial self-attention within each frame
        # torch.cuda.synchronize()
        # t_spatial_start = time.time()
        x_spatial = x.view(bsz * frames, seq_len, -1).permute(1, 0, 2)  # (L, B*F, C)
        spatial_valid_mask = valid_mask.view(bsz * frames, seq_len)
        x_spatial = self.spatial_encoder(
            x_spatial,
            src_key_padding_mask=self._to_key_padding_mask(spatial_valid_mask))
        x_spatial = x_spatial.permute(1, 0, 2).view(bsz, frames, seq_len, -1)  # (B, F, L, C)
        x_spatial = x_spatial * valid_mask.unsqueeze(-1).to(x_spatial.dtype)
        # torch.cuda.synchronize()
        # t_spatial = time.time() - t_spatial_start
        
        # # Frame-level temporal fusion
        # torch.cuda.synchronize()
        # t_temporal_start = time.time()
        # Reshape: keep full center frame tokens, compress non-center frames to K tokens
        if self.non_center_downsample_tokens is not None and frames > 1:
            center_frame_id = frames // 2
            center_tokens = x_spatial[:, center_frame_id]  # (B, L, C)
            center_mask = valid_mask[:, center_frame_id]  # (B, L)
            k = self.non_center_downsample_tokens
            if k > self.downsample_tokens:
                k = self.downsample_tokens
            other_tokens = []
            other_masks = []
            for f in range(frames):
                if f == center_frame_id:
                    continue
                frame_tokens = x_spatial[:, f]  # (B, L, C)
                frame_mask = valid_mask[:, f]  # (B, L)
                if k < seq_len and self.downsample_method == 'attention':
                    queries = self.downsample_queries[:, :k, :].expand(bsz, -1, -1)  # (B, K, C)
                    pooled, _ = self.downsample_attn(
                        query=queries,
                        key=frame_tokens,
                        value=frame_tokens,
                        key_padding_mask=self._to_key_padding_mask(frame_mask)
                    )
                    pooled_mask = frame_mask.any(dim=1, keepdim=True).expand(-1, k)
                elif k < seq_len:
                    # Fallback: uniform sampling when attention pooling is unavailable
                    idx = torch.linspace(0, seq_len - 1, steps=k, device=frame_tokens.device).long()
                    pooled = frame_tokens.index_select(1, idx)
                    pooled_mask = frame_mask.index_select(1, idx)
                else:
                    pooled = frame_tokens
                    pooled_mask = frame_mask
                other_tokens.append(pooled)
                other_masks.append(pooled_mask)
            if other_tokens:
                x_all_frames = torch.cat([center_tokens] + other_tokens, dim=1)  # (B, L + (F-1)*K, C)
                all_frames_mask = torch.cat([center_mask] + other_masks, dim=1)
            else:
                x_all_frames = center_tokens  # (B, L, C)
                all_frames_mask = center_mask
        else:
            # Reshape: separate each frame's tokens for cross-frame attention
            x_all_frames = x_spatial.view(bsz, frames * seq_len, -1)  # (B, F*L, C)
            all_frames_mask = valid_mask.view(bsz, frames * seq_len)

        temporal_key_padding_mask = self._to_key_padding_mask(all_frames_mask)
        
        # Apply temporal cross-attention layers
        for i in range(len(self.temporal_cross_attn)):
            # Each token can attend to all tokens from all frames
            residual = x_all_frames
            x_all_frames = self.temporal_norm1[i](x_all_frames)
            attn_out, _ = self.temporal_cross_attn[i](
                query=x_all_frames,
                key=x_all_frames,
                value=x_all_frames,
                key_padding_mask=temporal_key_padding_mask
            )
            x_all_frames = residual + attn_out
            
            # FFN
            residual = x_all_frames
            x_all_frames = self.temporal_norm2[i](x_all_frames)
            x_all_frames = residual + self.temporal_ffn[i](x_all_frames)
            x_all_frames = x_all_frames * all_frames_mask.unsqueeze(-1).to(x_all_frames.dtype)
        # torch.cuda.synchronize()
        # t_temporal = time.time() - t_temporal_start
        
        # # Center frame query: aggregate information for center frame prediction
        # torch.cuda.synchronize()
        # t_query_start = time.time()
        center_query = self.center_frame_query.expand(bsz, -1, -1)  # (B, N_query, C)
        
        # Explicitly encode center frame position
        center_frame_id = frames // 2  # Middle frame index (e.g., frame 2 in [0,1,2,3,4])
        frame_pos = torch.full(
            (bsz, self.num_query_tokens, 1),
            center_frame_id,
            device=center_query.device,
            dtype=center_query.dtype
        )  # (B, N_query, 1)
        
        # Encode frame position and add to query
        frame_pos_enc = self._encode_pos(frame_pos)  # (B, N_query, pos_enc_dim)
        frame_pos_feat = self.frame_pos_embed(frame_pos_enc)  # (B, N_query, C)
        center_query = center_query + frame_pos_feat  # Inject center frame info
        
        # Cross-attention: query attends to all frame tokens
        query_out, attn_weights = self.query_cross_attn(
            query=center_query,
            key=x_all_frames,
            value=x_all_frames,
            key_padding_mask=temporal_key_padding_mask
        )
        query_out = self.query_norm(query_out)  # (B, N_query, C)
        # torch.cuda.synchronize()
        # t_query = time.time() - t_query_start
        
        # # Performance profiling
        # t_total = time.time() - t_start
        # if hasattr(self, '_print_counter'):
        #     self._print_counter += 1
        # else:
        #     self._print_counter = 0
            
        # if self._print_counter % 10 == 0:  # Print every 10 iterations
        #     print(f"\n⏱️  Transformer Performance Breakdown (seq_len={seq_len}):")
        #     print(f"   Extract:     {t_extract*1000:6.2f}ms")
        #     print(f"   SpConv:      {t_spconv*1000:6.2f}ms  ⚠️")
        #     print(f"   Embedding:   {t_embed*1000:6.2f}ms")
        #     print(f"   FPN:         {t_fpn*1000:6.2f}ms")
        #     print(f"   Downsample:  {t_downsample*1000:6.2f}ms  ⚠️")
        #     print(f"   Spatial Attn:{t_spatial*1000:6.2f}ms  ⚠️")
        #     print(f"   Temporal:    {t_temporal*1000:6.2f}ms  ⚠️")
        #     print(f"   Query:       {t_query*1000:6.2f}ms")
        #     print(f"   TOTAL:       {t_total*1000:6.2f}ms\n")
        
        # Global pooling for backward compatibility
        global_feat = query_out.mean(dim=1)  # (B, C)
        
        outputs = {
            'features': global_feat,  # For simple heads that expect (B, C)
            'query_tokens': query_out,  # (B, 512, 256) - Rich spatial features
            'all_frame_features': x_all_frames
        }
        
        return outputs

    def _to_key_padding_mask(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """Convert valid-token mask to key_padding_mask expected by attention."""
        key_padding_mask = ~valid_mask.bool()
        if key_padding_mask.dim() != 2:
            raise ValueError(
                f'key_padding_mask expects 2D mask, got shape {tuple(key_padding_mask.shape)}')
        all_padded = key_padding_mask.all(dim=1)
        if all_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padded, 0] = False
        return key_padding_mask
    
    def _encode_pos(self, pos: torch.Tensor) -> torch.Tensor:
        """Apply frequency-based positional encoding."""
        if self.pos_freq_bands is None:
            return pos
        
        freq = self.pos_freq_bands.to(device=pos.device, dtype=pos.dtype)
        freq_shape = (1,) * (pos.dim() - 1) + (freq.numel(), 1)
        pos_expanded = pos.unsqueeze(-2) * freq.view(*freq_shape)
        cos_feats = torch.cos(pos_expanded)
        cos_feats = cos_feats.reshape(*pos.shape[:-1], -1)
        
        # Concatenate with original position
        return torch.cat([pos, cos_feats], dim=-1)
    
    def _apply_spconv(self, voxels: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply SpConv to voxel features.
        
        Args:
            voxels: (B, F, R, A, E) or (B, F, R, A, E, C) - voxelized features
            coords: Optional voxel coordinates
            
        Returns:
            features: (B, F, R*A*E, spconv_out_channels) - processed features
        """
        device = voxels.device
        
        # Handle both 5D (B, F, R, A, E) and 6D (B, F, R, A, E, C) inputs
        if voxels.dim() == 5:
            # No channel dimension, treat occupancy as single channel
            bsz, frames, r, a, e = voxels.shape
            voxels = voxels.unsqueeze(-1)  # (B, F, R, A, E, 1)
            c = 1
        else:
            bsz, frames, r, a, e, c = voxels.shape
        
        # Process each frame with SpConv
        all_frame_feats = []
        
        for f in range(frames):
            frame_voxels = voxels[:, f]  # (B, R, A, E, C)
            
            # Create sparse tensor for each sample in batch
            batch_feats = []
            for b in range(bsz):
                # Find non-zero voxels
                sample_voxels = frame_voxels[b]  # (R, A, E, C)
                non_zero_mask = sample_voxels.abs().sum(dim=-1) > 1e-6  # (R, A, E)
                
                if non_zero_mask.any():
                    # Get coordinates of non-zero voxels
                    nz_coords = torch.nonzero(non_zero_mask, as_tuple=False)  # (N, 3)
                    nz_feats = sample_voxels[non_zero_mask]  # (N, C)
                    
                    # Add batch index
                    batch_indices = torch.full(
                        (nz_coords.size(0), 1), b, 
                        dtype=torch.int32, device=device
                    )
                    nz_coords_with_batch = torch.cat([batch_indices, nz_coords.int()], dim=1)
                    
                    # Create sparse tensor
                    spatial_shape = [r, a, e]
                    sparse_tensor = spconv.SparseConvTensor(
                        features=nz_feats,
                        indices=nz_coords_with_batch,
                        spatial_shape=spatial_shape,
                        batch_size=bsz
                    )
                    
                    # Apply SpConv
                    sparse_out = self.spconv_layers(sparse_tensor)
                    
                    # Convert back to dense for this sample
                    dense_feats = torch.zeros(
                        (r, a, e, self.spconv_out_channels),
                        device=device, dtype=nz_feats.dtype
                    )
                    out_coords = sparse_out.indices[sparse_out.indices[:, 0] == b][:, 1:]
                    dense_feats[out_coords[:, 0], out_coords[:, 1], out_coords[:, 2]] = \
                        sparse_out.features[sparse_out.indices[:, 0] == b]
                    
                    batch_feats.append(dense_feats.view(-1, self.spconv_out_channels))
                else:
                    # All zeros, create zero output
                    zero_feats = torch.zeros(
                        (r * a * e, self.spconv_out_channels),
                        device=device, dtype=voxels.dtype
                    )
                    batch_feats.append(zero_feats)
            
            frame_feats = torch.stack(batch_feats, dim=0)  # (B, R*A*E, C)
            all_frame_feats.append(frame_feats)
        
        # Stack all frames
        output = torch.stack(all_frame_feats, dim=1)  # (B, F, R*A*E, C)
        return output
    
    def _apply_spconv_sparse(self, sparse_tokens: torch.Tensor, sparse_indices: torch.Tensor,
                            sparse_shape: torch.Tensor, sparse_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply SpConv - OPTIMIZED: Single SpConv call for all frames×batch.
        
        Args:
            sparse_tokens: (B, F, N, C) - voxel features
            sparse_indices: (B, F, N, 3) - voxel indices (r, a, e)
            sparse_shape: (B, 3) or (3,) - grid shape [R, A, E]
            sparse_mask: (B, F, N) - valid voxel mask
            
        Returns:
            features: (B, F, N, spconv_out_channels)
        """
        bsz, frames, max_tokens, feat_dim = sparse_tokens.shape
        device = sparse_tokens.device
        
        # Get spatial shape
        if isinstance(sparse_shape, torch.Tensor):
            spatial_shape = sparse_shape[0].cpu().tolist() if sparse_shape.dim() > 1 else sparse_shape.cpu().tolist()
        else:
            spatial_shape = list(sparse_shape)
        
        # Collect ALL valid voxels from all frames and batches in one go
        all_feats = []
        all_coords = []
        all_in_lin = []
        all_out_pos = []
        
        spatial_shape_t = torch.tensor(spatial_shape, device=device, dtype=torch.long)
        stride_r = spatial_shape_t[1] * spatial_shape_t[2]
        stride_a = spatial_shape_t[2]

        for b in range(bsz):
            for f in range(frames):
                valid_mask = sparse_mask[b, f]
                if valid_mask.any():
                    valid_tokens = sparse_tokens[b, f, valid_mask]
                    valid_indices = sparse_indices[b, f, valid_mask].long()

                    # Unique batch ID: b * frames + f
                    batch_id = b * frames + f
                    batch_idx = torch.full((valid_tokens.size(0), 1), batch_id,
                                           dtype=torch.int32, device=device)
                    coords = torch.cat([batch_idx, valid_indices.int()], dim=1)

                    lin = (batch_id * (stride_r * spatial_shape_t[0])
                           + valid_indices[:, 0] * stride_r
                           + valid_indices[:, 1] * stride_a
                           + valid_indices[:, 2])
                    out_pos = batch_id * max_tokens + valid_mask.nonzero(as_tuple=True)[0].long()

                    all_feats.append(valid_tokens)
                    all_coords.append(coords)
                    all_in_lin.append(lin)
                    all_out_pos.append(out_pos)
        
        if not all_feats:
            return torch.zeros((bsz, frames, max_tokens, self.spconv_out_channels),
                             device=device, dtype=sparse_tokens.dtype)
        
        # Single SpConv call for ALL data
        all_feats_cat = torch.cat(all_feats, dim=0)
        all_coords_cat = torch.cat(all_coords, dim=0)
        all_in_lin_cat = torch.cat(all_in_lin, dim=0)
        all_out_pos_cat = torch.cat(all_out_pos, dim=0)
        
        sparse_tensor = spconv.SparseConvTensor(
            features=all_feats_cat,
            indices=all_coords_cat,
            spatial_shape=spatial_shape,
            batch_size=bsz * frames
        )
        sparse_out = self.spconv_layers(sparse_tensor)
        
        # Reconstruct output - vectorized index matching
        output = torch.zeros((bsz, frames, max_tokens, self.spconv_out_channels),
                             device=device, dtype=sparse_tokens.dtype)
        if sparse_out.indices.numel() == 0:
            return output

        out_indices = sparse_out.indices.long()
        out_lin = (out_indices[:, 0] * (stride_r * spatial_shape_t[0])
                   + out_indices[:, 1] * stride_r
                   + out_indices[:, 2] * stride_a
                   + out_indices[:, 3])

        sorted_out_lin, sort_idx = torch.sort(out_lin)
        pos = torch.searchsorted(sorted_out_lin, all_in_lin_cat)
        valid = (pos < sorted_out_lin.numel()) & (sorted_out_lin[pos] == all_in_lin_cat)
        if valid.any():
            matched_feats = sparse_out.features[sort_idx[pos[valid]]]
            output_flat = output.view(bsz * frames * max_tokens, self.spconv_out_channels)
            if matched_feats.dtype != output_flat.dtype:
                matched_feats = matched_feats.to(output_flat.dtype)
            output_flat.index_copy_(0, all_out_pos_cat[valid], matched_feats)
        
        return output
