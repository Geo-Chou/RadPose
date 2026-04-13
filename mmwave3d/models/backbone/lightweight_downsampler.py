"""
Lightweight Token Downsampler for SpConv preprocessing.

This module reduces the number of tokens before SpConv processing using
lightweight attention or pooling mechanisms.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightTokenDownsampler(nn.Module):
    """
    Lightweight token downsampler that reduces voxel count before SpConv.
    
    This module uses a simple attention mechanism or learned pooling to
    select the most important tokens, reducing computational cost for
    downstream SpConv operations.
    
    Args:
        input_dim: Input feature dimension per token
        max_tokens: Maximum number of tokens to keep
        downsample_method: Method to use ('attention', 'importance', 'fps', 'random')
        hidden_dim: Hidden dimension for attention/importance network
        num_heads: Number of attention heads for attention-based downsampling
        dropout: Dropout rate
    """
    
    def __init__(self,
                 input_dim: int = 3,
                 max_tokens: int = 2048,
                 downsample_method: str = 'attention',
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.max_tokens = max_tokens
        self.downsample_method = downsample_method
        self.hidden_dim = hidden_dim
        
        if downsample_method == 'attention':
            # Learnable query tokens for downsampling
            self.downsample_queries = nn.Parameter(
                torch.randn(1, max_tokens, hidden_dim)
            )
            # Lightweight projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            # Cross-attention
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            # Output projection back to input space
            self.output_proj = nn.Linear(hidden_dim, input_dim)
            
        elif downsample_method == 'importance':
            # Learned importance scoring
            self.importance_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
            
        elif downsample_method in ('fps', 'random'):
            # No learnable parameters for FPS or random sampling
            pass
        else:
            raise ValueError(f"Unknown downsample_method: {downsample_method}")
    
    def forward(self,
                tokens: torch.Tensor,
                positions: torch.Tensor,
                indices: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Downsample tokens to max_tokens.
        
        Args:
            tokens: (B, F, N, C) - input token features
            positions: (B, F, N, 3) - token positions (r, azimuth, elevation)
            indices: (B, F, N, 3) - voxel grid indices
            mask: (B, F, N) - valid token mask
            
        Returns:
            downsampled_tokens: (B, F, N_down, C)
            downsampled_positions: (B, F, N_down, 3)
            downsampled_indices: (B, F, N_down, 3)
            downsampled_mask: (B, F, N_down)
        """
        bsz, frames, num_tokens, feat_dim = tokens.shape
        
        # Check if downsampling is needed
        valid_tokens_per_frame = mask.sum(dim=-1)  # (B, F)
        max_valid = valid_tokens_per_frame.max().item()
        
        if max_valid <= self.max_tokens:
            # No downsampling needed
            return tokens, positions, indices, mask
        
        # Process each frame separately
        down_tokens_list = []
        down_pos_list = []
        down_idx_list = []
        down_mask_list = []
        
        for f in range(frames):
            frame_tokens = tokens[:, f]  # (B, N, C)
            frame_pos = positions[:, f]  # (B, N, 3)
            frame_idx = indices[:, f]  # (B, N, 3)
            frame_mask = mask[:, f]  # (B, N)
            
            if self.downsample_method == 'attention':
                down_t, down_p, down_i, down_m = self._downsample_attention(
                    frame_tokens, frame_pos, frame_idx, frame_mask
                )
            elif self.downsample_method == 'importance':
                down_t, down_p, down_i, down_m = self._downsample_importance(
                    frame_tokens, frame_pos, frame_idx, frame_mask
                )
            elif self.downsample_method == 'fps':
                down_t, down_p, down_i, down_m = self._downsample_fps(
                    frame_tokens, frame_pos, frame_idx, frame_mask
                )
            elif self.downsample_method == 'random':
                down_t, down_p, down_i, down_m = self._downsample_random(
                    frame_tokens, frame_pos, frame_idx, frame_mask
                )
            else:
                raise ValueError(f"Unknown method: {self.downsample_method}")
            
            down_tokens_list.append(down_t)
            down_pos_list.append(down_p)
            down_idx_list.append(down_i)
            down_mask_list.append(down_m)
        
        # Stack frames back
        downsampled_tokens = torch.stack(down_tokens_list, dim=1)  # (B, F, N_down, C)
        downsampled_positions = torch.stack(down_pos_list, dim=1)  # (B, F, N_down, 3)
        downsampled_indices = torch.stack(down_idx_list, dim=1)  # (B, F, N_down, 3)
        downsampled_mask = torch.stack(down_mask_list, dim=1)  # (B, F, N_down)
        
        return downsampled_tokens, downsampled_positions, downsampled_indices, downsampled_mask
    
    def _downsample_attention(self,
                             tokens: torch.Tensor,
                             positions: torch.Tensor,
                             indices: torch.Tensor,
                             mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attention-based downsampling using learnable queries."""
        bsz = tokens.shape[0]
        device = tokens.device
        
        # Project tokens to hidden space
        tokens_proj = self.input_proj(tokens)  # (B, N, H)
        
        # Expand learnable queries
        queries = self.downsample_queries.expand(bsz, -1, -1)  # (B, N_down, H)
        
        # Cross-attention: queries attend to input tokens
        # Note: mask True positions are ignored
        # Create attention mask to ignore invalid tokens
        # attn_mask shape: (B, N_down, N) or (B*num_heads, N_down, N)
        attn_mask = ~mask.unsqueeze(1).expand(-1, self.max_tokens, -1)  # (B, N_down, N)
        
        down_feats, attn_weights = self.cross_attn(
            query=queries,
            key=tokens_proj,
            value=tokens_proj,
            key_padding_mask=~mask  # True positions are ignored
        )  # (B, N_down, H), (B, N_down, N)
        
        # Project back to original space
        down_tokens = self.output_proj(down_feats)  # (B, N_down, C)
        
        # For positions and indices, use attention weights to compute weighted average
        # or select top-K attended positions
        attn_weights = attn_weights.detach()  # (B, N_down, N)
        
        # Masked softmax for position/index selection
        attn_weights_masked = attn_weights.masked_fill(~mask.unsqueeze(1), -1e9)
        attn_weights_norm = F.softmax(attn_weights_masked, dim=-1)  # (B, N_down, N)
        
        # Weighted average of positions
        down_positions = torch.bmm(attn_weights_norm, positions)  # (B, N_down, 3)
        
        # For indices, select the most attended index per query
        max_attn_idx = attn_weights_masked.argmax(dim=-1)  # (B, N_down)

        down_indices = torch.gather(
            indices,
            1,
            max_attn_idx.unsqueeze(-1).expand(-1, -1, 3)
        )  # (B, N_down, 3)
        
        # All downsampled tokens are valid
        down_mask = torch.ones((bsz, self.max_tokens), device=device, dtype=torch.bool)
        
        return down_tokens, down_positions, down_indices, down_mask
    
    def _downsample_importance(self,
                               tokens: torch.Tensor,
                               positions: torch.Tensor,
                               indices: torch.Tensor,
                               mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Importance-based downsampling using learned scoring."""
        bsz, num_tokens, _ = tokens.shape
        device = tokens.device
        
        # Compute importance scores
        importance = self.importance_net(tokens).squeeze(-1)  # (B, N)
        
        # Mask invalid tokens
        importance = importance.masked_fill(~mask, -1e9)
        
        # Select top-K tokens
        if num_tokens > self.max_tokens:
            _, top_indices = torch.topk(importance, k=self.max_tokens, dim=-1)  # (B, N_down)
            top_indices, _ = top_indices.sort(dim=-1)  # Keep sorted for consistency
        else:
            top_indices = torch.arange(num_tokens, device=device).unsqueeze(0).expand(bsz, -1)
        
        # Gather selected tokens
        down_tokens = torch.gather(
            tokens, 1, top_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        )
        down_positions = torch.gather(
            positions, 1, top_indices.unsqueeze(-1).expand(-1, -1, 3)
        )
        down_indices = torch.gather(
            indices, 1, top_indices.unsqueeze(-1).expand(-1, -1, 3)
        )
        down_mask = torch.gather(mask, 1, top_indices)
        
        return down_tokens, down_positions, down_indices, down_mask
    
    def _downsample_fps(self,
                       tokens: torch.Tensor,
                       positions: torch.Tensor,
                       indices: torch.Tensor,
                       mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Farthest Point Sampling (FPS) downsampling."""
        bsz, num_tokens, _ = tokens.shape
        device = tokens.device
        
        # Simple FPS implementation (batch-wise)
        selected_indices = []
        
        for b in range(bsz):
            valid_mask_b = mask[b]
            valid_pos_b = positions[b, valid_mask_b]  # (N_valid, 3)
            
            if valid_pos_b.shape[0] == 0:
                # No valid tokens
                selected_indices.append(torch.zeros(self.max_tokens, dtype=torch.long, device=device))
                continue
            
            n_valid = valid_pos_b.shape[0]
            n_sample = min(self.max_tokens, n_valid)
            
            # FPS algorithm
            sampled = torch.zeros(n_sample, dtype=torch.long, device=device)
            distances = torch.full((n_valid,), 1e10, device=device)
            
            # Start with random point
            farthest = torch.randint(0, n_valid, (1,), device=device).item()
            
            for i in range(n_sample):
                sampled[i] = farthest
                centroid = valid_pos_b[farthest].unsqueeze(0)  # (1, 3)
                dist = torch.sum((valid_pos_b - centroid) ** 2, dim=-1)
                distances = torch.min(distances, dist)
                farthest = torch.argmax(distances).item()
            
            # Map back to original indices
            valid_original_idx = torch.where(valid_mask_b)[0]
            selected_b = valid_original_idx[sampled]
            
            # Pad if necessary
            if selected_b.shape[0] < self.max_tokens:
                padding = torch.zeros(
                    self.max_tokens - selected_b.shape[0],
                    dtype=torch.long,
                    device=device
                )
                selected_b = torch.cat([selected_b, padding], dim=0)
            
            selected_indices.append(selected_b)
        
        selected_indices = torch.stack(selected_indices, dim=0)  # (B, N_down)
        
        # Gather tokens
        down_tokens = torch.gather(
            tokens, 1, selected_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        )
        down_positions = torch.gather(
            positions, 1, selected_indices.unsqueeze(-1).expand(-1, -1, 3)
        )
        down_indices = torch.gather(
            indices, 1, selected_indices.unsqueeze(-1).expand(-1, -1, 3)
        )
        down_mask = torch.gather(mask, 1, selected_indices)
        
        return down_tokens, down_positions, down_indices, down_mask
    
    def _downsample_random(self,
                          tokens: torch.Tensor,
                          positions: torch.Tensor,
                          indices: torch.Tensor,
                          mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Random downsampling."""
        bsz, num_tokens, _ = tokens.shape
        device = tokens.device
        
        selected_indices = []
        
        for b in range(bsz):
            valid_mask_b = mask[b]
            valid_count = valid_mask_b.sum().item()
            
            if valid_count == 0:
                selected_indices.append(torch.zeros(self.max_tokens, dtype=torch.long, device=device))
                continue
            
            valid_idx = torch.where(valid_mask_b)[0]
            n_sample = min(self.max_tokens, valid_count)
            
            # Random sampling
            perm = torch.randperm(valid_count, device=device)[:n_sample]
            selected_b = valid_idx[perm]
            
            # Pad if necessary
            if selected_b.shape[0] < self.max_tokens:
                padding = torch.zeros(
                    self.max_tokens - selected_b.shape[0],
                    dtype=torch.long,
                    device=device
                )
                selected_b = torch.cat([selected_b, padding], dim=0)
            
            selected_indices.append(selected_b)
        
        selected_indices = torch.stack(selected_indices, dim=0)  # (B, N_down)
        
        # Gather tokens
        down_tokens = torch.gather(
            tokens, 1, selected_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        )
        down_positions = torch.gather(
            positions, 1, selected_indices.unsqueeze(-1).expand(-1, -1, 3)
        )
        down_indices = torch.gather(
            indices, 1, selected_indices.unsqueeze(-1).expand(-1, -1, 3)
        )
        down_mask = torch.gather(mask, 1, selected_indices)
        
        return down_tokens, down_positions, down_indices, down_mask
