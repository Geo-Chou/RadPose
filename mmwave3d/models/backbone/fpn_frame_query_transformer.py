import warnings
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from mmengine.logging import MMLogger

from mmwave3d.utils import FlashMultiheadAttention

try:
    import spconv.pytorch as spconv

    HAS_SPCONV = True

except ImportError:
    spconv = None
    HAS_SPCONV = False

from mmwave3d.registry import MODELS

_SPCONV_STATUS_MESSAGES_EMITTED = set()


def _emit_spconv_status_once(status: str, message: str, level: str = "warning") -> None:
    if status in _SPCONV_STATUS_MESSAGES_EMITTED:
        return
    _SPCONV_STATUS_MESSAGES_EMITTED.add(status)

    logger = MMLogger.get_current_instance()
    if logger is not None:
        log_fn = getattr(logger, level, logger.info)
        log_fn(message)
        return

    warnings.warn(message, stacklevel=2)


@MODELS.register_module()
class FPNFrameQueryTransformer(nn.Module):
    def __init__(
        self,
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
        downsample_method: str = "attention",
        non_center_downsample_tokens: Optional[int] = None,
        norm_first: bool = True,
        pos_freq_bands: Optional[Sequence[float]] = (1.0, 2.0, 4.0, 8.0),
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_query_tokens = num_query_tokens
        self.downsample_tokens = downsample_tokens
        self.downsample_method = downsample_method
        self.non_center_downsample_tokens = non_center_downsample_tokens

        self.use_spconv = HAS_SPCONV and torch.cuda.is_available()

        self.spconv_in_channels = (
            input_token_dim if self.use_spconv else input_token_dim
        )
        self.spconv_out_channels = (
            spconv_channels[-1] if spconv_channels else input_token_dim
        )
        self.spconv_kernel_size = spconv_kernel_size

        if self.use_spconv:
            spconv_layers = []
            in_ch = input_token_dim
            for out_ch in spconv_channels:
                spconv_layers.extend(
                    [
                        spconv.SubMConv3d(
                            in_ch,
                            out_ch,
                            kernel_size=spconv_kernel_size,
                            padding=spconv_kernel_size // 2,
                            bias=False,
                            indice_key=f"subm{len(spconv_layers)}",
                        ),
                        nn.BatchNorm1d(out_ch),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_ch = out_ch
            self.spconv_layers = spconv.SparseSequential(*spconv_layers)
            _emit_spconv_status_once(
                "spconv_enabled",
                "Using SpConv with BatchNorm1d (no cross-GPU sync) for faster training",
                level="info",
            )

        if not self.use_spconv or input_token_dim != self.spconv_out_channels:
            self.feature_proj = nn.Linear(input_token_dim, self.spconv_out_channels)
            if not self.use_spconv:
                if not HAS_SPCONV:
                    _emit_spconv_status_once(
                        "spconv_missing",
                        "SpConv not available, using Linear projection",
                    )
                else:
                    _emit_spconv_status_once(
                        "spconv_cuda_unavailable",
                        "CUDA not available, using Linear projection for SpConv",
                    )

        self.token_embed = nn.Linear(self.spconv_out_channels, embed_dim)

        if pos_freq_bands is None:
            self.register_buffer("pos_freq_bands", None, persistent=False)
            pos_in_dim = pos_dim
        else:
            freq_tensor = torch.tensor(pos_freq_bands, dtype=torch.float32)
            self.register_buffer("pos_freq_bands", freq_tensor, persistent=False)
            pos_in_dim = pos_dim * (len(pos_freq_bands) + 1)

        self.pos_embed = nn.Linear(pos_in_dim, embed_dim)

        self.fpn_layers = nn.ModuleList()
        for ch in fpn_channels:
            self.fpn_layers.append(
                nn.Sequential(
                    nn.Linear(embed_dim, ch), nn.LayerNorm(ch), nn.ReLU(inplace=True)
                )
            )

        self.fpn_fusion = nn.ModuleList()
        for ch in fpn_channels:
            self.fpn_fusion.append(nn.Linear(ch, embed_dim))

        if downsample_method == "attention":
            self.downsample_queries = nn.Parameter(
                torch.randn(1, downsample_tokens, embed_dim).contiguous()
            )
            self.downsample_attn = FlashMultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=False,
            norm_first=norm_first,
        )
        self.spatial_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.temporal_cross_attn = nn.ModuleList()
        for _ in range(temporal_layers):
            self.temporal_cross_attn.append(
                FlashMultiheadAttention(
                    embed_dim, num_heads, dropout=dropout, batch_first=True
                )
            )

        self.temporal_ffn = nn.ModuleList()
        for _ in range(temporal_layers):
            self.temporal_ffn.append(
                nn.Sequential(
                    nn.Linear(embed_dim, ffn_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, embed_dim),
                    nn.Dropout(dropout),
                )
            )

        self.temporal_norm1 = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(temporal_layers)]
        )
        self.temporal_norm2 = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(temporal_layers)]
        )

        self.center_frame_query = nn.Parameter(
            torch.randn(1, num_query_tokens, embed_dim).contiguous()
        )

        frame_pos_in_dim = 1 * (len(pos_freq_bands) + 1) if pos_freq_bands else 1
        self.frame_pos_embed = nn.Linear(frame_pos_in_dim, embed_dim)

        self.query_cross_attn = FlashMultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        if "mmwave_sparse_tokens" in inputs:
            tokens = inputs["mmwave_sparse_tokens"]
            pos_grid = inputs["mmwave_sparse_pos"]
            bsz, frames, seq_len, _ = tokens.shape
            valid_mask = inputs.get("mmwave_sparse_mask", None)
            if valid_mask is None:
                valid_mask = torch.ones(
                    (bsz, frames, seq_len), device=tokens.device, dtype=torch.bool
                )
            else:
                valid_mask = valid_mask.to(device=tokens.device, dtype=torch.bool)
        else:
            voxels = inputs["mmwave_voxels"]
            pos_grid = inputs["mmwave_pos_grid"]
            bsz, frames, r, a, e = voxels.shape
            seq_len = r * a * e
            tokens = voxels.view(bsz, frames, seq_len, -1)
            pos_grid = pos_grid.view(bsz, frames, seq_len, -1)
            valid_mask = torch.ones(
                (bsz, frames, seq_len), device=tokens.device, dtype=torch.bool
            )

        frame_coord = torch.arange(frames, device=tokens.device, dtype=tokens.dtype)
        frame_coord = frame_coord.view(1, frames, 1, 1).expand(bsz, frames, seq_len, 1)
        pos = torch.cat([pos_grid, frame_coord], dim=-1)

        if self.use_spconv and HAS_SPCONV and "mmwave_sparse_indices" in inputs:
            tokens_proj = self._apply_spconv_sparse(
                inputs["mmwave_sparse_tokens"],
                inputs["mmwave_sparse_indices"],
                inputs["mmwave_sparse_shape"],
                inputs["mmwave_sparse_mask"],
            )
        elif self.use_spconv and HAS_SPCONV and "mmwave_voxels" in inputs:
            tokens_proj = self._apply_spconv(
                inputs["mmwave_voxels"], inputs.get("mmwave_coords", None)
            )
        elif tokens.size(-1) == self.spconv_out_channels:
            tokens_proj = tokens
        elif hasattr(self, "feature_proj"):
            tokens_proj = self.feature_proj(tokens)
        else:
            tokens_proj = tokens

        token_feat = self.token_embed(tokens_proj)
        pos_enc = self._encode_pos(pos)
        pos_feat = self.pos_embed(pos_enc)
        x = token_feat + pos_feat

        fpn_features = []
        for fpn_layer in self.fpn_layers:
            fpn_feat = fpn_layer(x)
            fpn_features.append(fpn_feat)

        fused_features = []
        for fpn_feat, fusion_layer in zip(fpn_features, self.fpn_fusion):
            fused = fusion_layer(fpn_feat)
            fused_features.append(fused)

        x = sum(fused_features) / len(fused_features)

        if self.downsample_method == "attention" and seq_len > self.downsample_tokens:
            x_down = []
            mask_down = []
            for f in range(frames):
                frame_tokens = x[:, f, :, :]
                frame_mask = valid_mask[:, f, :]
                queries = self.downsample_queries.expand(bsz, -1, -1)
                down_tokens, _ = self.downsample_attn(
                    query=queries,
                    key=frame_tokens,
                    value=frame_tokens,
                    key_padding_mask=self._to_key_padding_mask(frame_mask),
                )
                x_down.append(down_tokens)
                down_mask = frame_mask.any(dim=1, keepdim=True).expand(
                    -1, self.downsample_tokens
                )
                mask_down.append(down_mask)
            x = torch.stack(x_down, dim=1)
            valid_mask = torch.stack(mask_down, dim=1)
            seq_len = self.downsample_tokens

        x_spatial = x.view(bsz * frames, seq_len, -1).permute(1, 0, 2)
        spatial_valid_mask = valid_mask.view(bsz * frames, seq_len)
        x_spatial = self.spatial_encoder(
            x_spatial,
            src_key_padding_mask=self._to_key_padding_mask(spatial_valid_mask),
        )
        x_spatial = x_spatial.permute(1, 0, 2).view(bsz, frames, seq_len, -1)
        x_spatial = x_spatial * valid_mask.unsqueeze(-1).to(x_spatial.dtype)

        if self.non_center_downsample_tokens is not None and frames > 1:
            center_frame_id = frames // 2
            center_tokens = x_spatial[:, center_frame_id]
            center_mask = valid_mask[:, center_frame_id]
            k = self.non_center_downsample_tokens
            if k > self.downsample_tokens:
                k = self.downsample_tokens
            other_tokens = []
            other_masks = []
            for f in range(frames):
                if f == center_frame_id:
                    continue
                frame_tokens = x_spatial[:, f]
                frame_mask = valid_mask[:, f]
                if k < seq_len and self.downsample_method == "attention":
                    queries = self.downsample_queries[:, :k, :].expand(bsz, -1, -1)
                    pooled, _ = self.downsample_attn(
                        query=queries,
                        key=frame_tokens,
                        value=frame_tokens,
                        key_padding_mask=self._to_key_padding_mask(frame_mask),
                    )
                    pooled_mask = frame_mask.any(dim=1, keepdim=True).expand(-1, k)
                elif k < seq_len:
                    idx = torch.linspace(
                        0, seq_len - 1, steps=k, device=frame_tokens.device
                    ).long()
                    pooled = frame_tokens.index_select(1, idx)
                    pooled_mask = frame_mask.index_select(1, idx)
                else:
                    pooled = frame_tokens
                    pooled_mask = frame_mask
                other_tokens.append(pooled)
                other_masks.append(pooled_mask)
            if other_tokens:
                x_all_frames = torch.cat([center_tokens] + other_tokens, dim=1)
                all_frames_mask = torch.cat([center_mask] + other_masks, dim=1)
            else:
                x_all_frames = center_tokens
                all_frames_mask = center_mask
        else:
            x_all_frames = x_spatial.view(bsz, frames * seq_len, -1)
            all_frames_mask = valid_mask.view(bsz, frames * seq_len)

        temporal_key_padding_mask = self._to_key_padding_mask(all_frames_mask)

        for i in range(len(self.temporal_cross_attn)):
            residual = x_all_frames
            x_all_frames = self.temporal_norm1[i](x_all_frames)
            attn_out, _ = self.temporal_cross_attn[i](
                query=x_all_frames,
                key=x_all_frames,
                value=x_all_frames,
                key_padding_mask=temporal_key_padding_mask,
            )
            x_all_frames = residual + attn_out

            residual = x_all_frames
            x_all_frames = self.temporal_norm2[i](x_all_frames)
            x_all_frames = residual + self.temporal_ffn[i](x_all_frames)
            x_all_frames = x_all_frames * all_frames_mask.unsqueeze(-1).to(
                x_all_frames.dtype
            )

        center_query = self.center_frame_query.expand(bsz, -1, -1)

        center_frame_id = frames // 2
        frame_pos = torch.full(
            (bsz, self.num_query_tokens, 1),
            center_frame_id,
            device=center_query.device,
            dtype=center_query.dtype,
        )

        frame_pos_enc = self._encode_pos(frame_pos)
        frame_pos_feat = self.frame_pos_embed(frame_pos_enc)
        center_query = center_query + frame_pos_feat

        query_out, attn_weights = self.query_cross_attn(
            query=center_query,
            key=x_all_frames,
            value=x_all_frames,
            key_padding_mask=temporal_key_padding_mask,
        )
        query_out = self.query_norm(query_out)

        global_feat = query_out.mean(dim=1)

        outputs = {
            "features": global_feat,
            "query_tokens": query_out,
            "all_frame_features": x_all_frames,
        }

        return outputs

    def _to_key_padding_mask(self, valid_mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~valid_mask.bool()
        if key_padding_mask.dim() != 2:
            raise ValueError(
                f"key_padding_mask expects 2D mask, got shape {tuple(key_padding_mask.shape)}"
            )
        all_padded = key_padding_mask.all(dim=1)
        if all_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padded, 0] = False
        return key_padding_mask

    def _encode_pos(self, pos: torch.Tensor) -> torch.Tensor:
        if self.pos_freq_bands is None:
            return pos

        freq = self.pos_freq_bands.to(device=pos.device, dtype=pos.dtype)
        freq_shape = (1,) * (pos.dim() - 1) + (freq.numel(), 1)
        pos_expanded = pos.unsqueeze(-2) * freq.view(*freq_shape)
        cos_feats = torch.cos(pos_expanded)
        cos_feats = cos_feats.reshape(*pos.shape[:-1], -1)

        return torch.cat([pos, cos_feats], dim=-1)

    def _apply_spconv(
        self, voxels: torch.Tensor, coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        device = voxels.device

        if voxels.dim() == 5:
            bsz, frames, r, a, e = voxels.shape
            voxels = voxels.unsqueeze(-1)
            c = 1
        else:
            bsz, frames, r, a, e, c = voxels.shape

        all_frame_feats = []

        for f in range(frames):
            frame_voxels = voxels[:, f]

            batch_feats = []
            for b in range(bsz):
                sample_voxels = frame_voxels[b]
                non_zero_mask = sample_voxels.abs().sum(dim=-1) > 1e-6

                if non_zero_mask.any():
                    nz_coords = torch.nonzero(non_zero_mask, as_tuple=False)
                    nz_feats = sample_voxels[non_zero_mask]

                    batch_indices = torch.full(
                        (nz_coords.size(0), 1), b, dtype=torch.int32, device=device
                    )
                    nz_coords_with_batch = torch.cat(
                        [batch_indices, nz_coords.int()], dim=1
                    )

                    spatial_shape = [r, a, e]
                    sparse_tensor = spconv.SparseConvTensor(
                        features=nz_feats,
                        indices=nz_coords_with_batch,
                        spatial_shape=spatial_shape,
                        batch_size=bsz,
                    )

                    sparse_out = self.spconv_layers(sparse_tensor)

                    dense_feats = torch.zeros(
                        (r, a, e, self.spconv_out_channels),
                        device=device,
                        dtype=nz_feats.dtype,
                    )
                    out_coords = sparse_out.indices[sparse_out.indices[:, 0] == b][
                        :, 1:
                    ]
                    dense_feats[
                        out_coords[:, 0], out_coords[:, 1], out_coords[:, 2]
                    ] = sparse_out.features[sparse_out.indices[:, 0] == b]

                    batch_feats.append(dense_feats.view(-1, self.spconv_out_channels))
                else:
                    zero_feats = torch.zeros(
                        (r * a * e, self.spconv_out_channels),
                        device=device,
                        dtype=voxels.dtype,
                    )
                    batch_feats.append(zero_feats)

            frame_feats = torch.stack(batch_feats, dim=0)
            all_frame_feats.append(frame_feats)

        output = torch.stack(all_frame_feats, dim=1)
        return output

    def _apply_spconv_sparse(
        self,
        sparse_tokens: torch.Tensor,
        sparse_indices: torch.Tensor,
        sparse_shape: torch.Tensor,
        sparse_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, frames, max_tokens, feat_dim = sparse_tokens.shape
        device = sparse_tokens.device

        if isinstance(sparse_shape, torch.Tensor):
            spatial_shape = (
                sparse_shape[0].cpu().tolist()
                if sparse_shape.dim() > 1
                else sparse_shape.cpu().tolist()
            )
        else:
            spatial_shape = list(sparse_shape)

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

                    batch_id = b * frames + f
                    batch_idx = torch.full(
                        (valid_tokens.size(0), 1),
                        batch_id,
                        dtype=torch.int32,
                        device=device,
                    )
                    coords = torch.cat([batch_idx, valid_indices.int()], dim=1)

                    lin = (
                        batch_id * (stride_r * spatial_shape_t[0])
                        + valid_indices[:, 0] * stride_r
                        + valid_indices[:, 1] * stride_a
                        + valid_indices[:, 2]
                    )
                    out_pos = (
                        batch_id * max_tokens
                        + valid_mask.nonzero(as_tuple=True)[0].long()
                    )

                    all_feats.append(valid_tokens)
                    all_coords.append(coords)
                    all_in_lin.append(lin)
                    all_out_pos.append(out_pos)

        if not all_feats:
            return torch.zeros(
                (bsz, frames, max_tokens, self.spconv_out_channels),
                device=device,
                dtype=sparse_tokens.dtype,
            )

        all_feats_cat = torch.cat(all_feats, dim=0)
        all_coords_cat = torch.cat(all_coords, dim=0)
        all_in_lin_cat = torch.cat(all_in_lin, dim=0)
        all_out_pos_cat = torch.cat(all_out_pos, dim=0)

        sparse_tensor = spconv.SparseConvTensor(
            features=all_feats_cat,
            indices=all_coords_cat,
            spatial_shape=spatial_shape,
            batch_size=bsz * frames,
        )
        sparse_out = self.spconv_layers(sparse_tensor)

        output = torch.zeros(
            (bsz, frames, max_tokens, self.spconv_out_channels),
            device=device,
            dtype=sparse_tokens.dtype,
        )
        if sparse_out.indices.numel() == 0:
            return output

        out_indices = sparse_out.indices.long()
        out_lin = (
            out_indices[:, 0] * (stride_r * spatial_shape_t[0])
            + out_indices[:, 1] * stride_r
            + out_indices[:, 2] * stride_a
            + out_indices[:, 3]
        )

        sorted_out_lin, sort_idx = torch.sort(out_lin)
        pos = torch.searchsorted(sorted_out_lin, all_in_lin_cat)
        valid = (pos < sorted_out_lin.numel()) & (sorted_out_lin[pos] == all_in_lin_cat)
        if valid.any():
            matched_feats = sparse_out.features[sort_idx[pos[valid]]]
            output_flat = output.view(
                bsz * frames * max_tokens, self.spconv_out_channels
            )
            if matched_feats.dtype != output_flat.dtype:
                matched_feats = matched_feats.to(output_flat.dtype)
            output_flat.index_copy_(0, all_out_pos_cat[valid], matched_feats)

        return output
