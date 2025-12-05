from typing import Dict, Optional, Sequence
import time

import torch
import torch.nn as nn

from mmwave3d.registry import MODELS
from mmengine.logging import MMLogger

@MODELS.register_module()
class MmWaveTransformerBackbone(nn.Module):
    """Transformer encoder that consumes voxel tokens and position encodings."""

    def __init__(self,
                 token_dim: int = 1,
                 pos_dim: int = 4,
                 embed_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ffn_dim: int = 1024,
                 dropout: float = 0.1,
                 norm_first: bool = True,
                 pos_freq_bands: Optional[Sequence[float]] = (1.0, 2.0, 4.0, 8.0)) -> None:
        super().__init__()
        self.token_embed = nn.Linear(token_dim, embed_dim)
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode mmWave voxel tokens and return features for downstream heads."""
        tokens = inputs['mmwave_tokens']  # (B, N, token_dim)
        pos = inputs['mmwave_pos']  # (B, N, pos_dim)
        bsz, seq_len, _ = tokens.shape
        # torch.cuda.synchronize()
        # t_start = time.time()
        token_feat = self.token_embed(tokens)
        # torch.cuda.synchronize()
        # token_t = time.time()
        # token_time = token_t-t_start
        pos_enc = self._encode_pos(pos)
        pos_feat = self.pos_embed(pos_enc)
        # torch.cuda.synchronize()
        # pos_t = time.time()
        # pos_time = pos_t - token_t
        x = token_feat + pos_feat
        x = x.permute(1, 0, 2)  # (N, B, C) for TransformerEncoder
        x = self.encoder(x)
        # torch.cuda.synchronize()
        # trans_t = time.time()
        # trans_time = trans_t - pos_t
        x = x.permute(1, 0, 2)  # back to (B, N, C)

        outputs = dict(features=x)
        # logger = MMLogger.get_current_instance()
        # if logger is not None:
        #             logger.info(
        #                 'Tansformer timings - token embedding: %.4fs  pos embedding: %.4fs  transformers: %.4fs ',
        #                 token_time, pos_time, trans_time)
        return outputs

    def _encode_pos(self, pos: torch.Tensor) -> torch.Tensor:
        """Apply cosine positional encoding before linear projection."""
        if self.pos_freq_bands is None:
            return pos
        freq = self.pos_freq_bands.to(device=pos.device, dtype=pos.dtype)
        # pos: (B, N, D). Expand freq to broadcast over the last dim.
        pos_expanded = pos.unsqueeze(-2) * freq.view(1, 1, -1, 1)
        cos_feats = torch.cos(pos_expanded)
        cos_feats = cos_feats.reshape(pos.shape[0], pos.shape[1], -1)
        return torch.cat([pos, cos_feats], dim=-1)
