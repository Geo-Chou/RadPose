from typing import Dict, Optional

import torch
import torch.nn as nn

from mmwave3d.registry import MODELS


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
                 norm_first: bool = True) -> None:
        super().__init__()
        self.token_embed = nn.Linear(token_dim, embed_dim)
        self.pos_embed = nn.Linear(pos_dim, embed_dim)
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

        token_feat = self.token_embed(tokens)
        pos_feat = self.pos_embed(pos)
        x = token_feat + pos_feat
        x = x.permute(1, 0, 2)  # (N, B, C) for TransformerEncoder
        x = self.encoder(x)
        x = x.permute(1, 0, 2)  # back to (B, N, C)

        outputs = dict(features=x)
        return outputs
