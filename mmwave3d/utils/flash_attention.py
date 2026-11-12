from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if not batch_first:
            raise ValueError(
                "FlashMultiheadAttention currently supports batch_first=True only."
            )

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.head_dim = self.embed_dim // self.num_heads
        self.batch_first = True

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if need_weights:
            raise NotImplementedError(
                "FlashMultiheadAttention only supports need_weights=False. "
                "Use nn.MultiheadAttention for paths that require attention weights."
            )
        if average_attn_weights is False:
            raise NotImplementedError("average_attn_weights=False is unsupported.")
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError("Expected query/key/value to be shaped (B, L, C).")

        bsz, tgt_len, _ = query.shape
        src_len = key.size(1)

        q = self._reshape_heads(self.q_proj(query), bsz, tgt_len)
        k = self._reshape_heads(self.k_proj(key), bsz, src_len)
        v = self._reshape_heads(self.v_proj(value), bsz, src_len)

        merged_mask = self._merge_masks(
            batch_size=bsz,
            tgt_len=tgt_len,
            src_len=src_len,
            device=query.device,
            dtype=q.dtype,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=merged_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        out = self.out_proj(out)
        out = torch.nan_to_num(out)
        return out, None

    def _reshape_heads(
        self, x: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

    def _merge_masks(
        self,
        batch_size: int,
        tgt_len: int,
        src_len: int,
        device: torch.device,
        dtype: torch.dtype,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        merged_mask = None

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                merged_mask = attn_mask.view(1, 1, tgt_len, src_len)
            elif attn_mask.dim() == 3:
                if attn_mask.size(0) == batch_size:
                    merged_mask = attn_mask.view(batch_size, 1, tgt_len, src_len)
                elif attn_mask.size(0) == batch_size * self.num_heads:
                    merged_mask = attn_mask.view(
                        batch_size, self.num_heads, tgt_len, src_len
                    )
                else:
                    raise ValueError("Unsupported attn_mask shape.")
            else:
                raise ValueError("attn_mask must be 2D or 3D.")
            merged_mask = merged_mask.to(device=device)
            if merged_mask.dtype == torch.bool:
                merged_mask = ~merged_mask
            else:
                merged_mask = merged_mask.to(dtype=dtype)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device=device, dtype=torch.bool)
            if key_padding_mask.dim() != 2:
                raise ValueError("key_padding_mask must be 2D with shape (B, S).")
            all_padded = key_padding_mask.all(dim=1)
            if all_padded.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_padded, 0] = False
            allowed_kpm = (~key_padding_mask).view(batch_size, 1, 1, src_len)
            if merged_mask is None:
                merged_mask = allowed_kpm
            elif merged_mask.dtype == torch.bool:
                merged_mask = merged_mask & allowed_kpm
            else:
                padding_bias = torch.zeros(
                    (batch_size, 1, 1, src_len), device=device, dtype=dtype
                )
                padding_bias = padding_bias.masked_fill(
                    ~allowed_kpm, torch.finfo(dtype).min
                )
                merged_mask = merged_mask + padding_bias

        return merged_mask
