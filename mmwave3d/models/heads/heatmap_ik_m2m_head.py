from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from mmwave3d.registry import MODELS


@MODELS.register_module()
class HeatmapIKM2MHead(nn.Module):
    """Run the existing heatmap+IK heads independently for every input frame."""

    def __init__(self,
                 heatmap_head: Dict,
                 ik_head: Dict,
                 num_frames: int = 5,
                 center_frame_as_default: bool = True) -> None:
        super().__init__()
        self.heatmap_head = MODELS.build(heatmap_head)
        self.ik_head = MODELS.build(ik_head)
        self.num_frames = int(num_frames)
        self.center_frame_as_default = center_frame_as_default

    def forward(self,
                feats: Dict[str, torch.Tensor],
                data_samples: Optional[List] = None) -> Dict[str, torch.Tensor]:
        frame_feats = self._get_frame_features(feats)
        bsz, frames, _, channels = frame_feats.shape

        flat_feats = {
            'features': frame_feats.mean(dim=2).reshape(bsz * frames, channels)
        }
        flat_samples = self._flatten_frame_samples(data_samples, frames)

        heatmap_outputs = self.heatmap_head(flat_feats, flat_samples)
        ik_outputs = self.ik_head(flat_feats, heatmap_outputs, flat_samples)

        outputs = self._reshape_frame_outputs(
            heatmap_outputs, bsz, frames, prefix='')
        outputs.update(self._reshape_frame_outputs(
            ik_outputs, bsz, frames, prefix=''))

        outputs['heatmap_keypoints_frames'] = outputs.get('keypoints_frames')
        if 'ik_keypoints_frames' in outputs:
            outputs['keypoints_frames'] = outputs['ik_keypoints_frames']

        center_id = frames // 2
        if self.center_frame_as_default:
            self._expose_center_frame(outputs, center_id)

        return outputs

    def _get_frame_features(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        if 'frame_features' in feats:
            frame_feats = feats['frame_features']
            if frame_feats.dim() != 4:
                raise RuntimeError(
                    'frame_features must have shape (B, F, N, C), got '
                    f'{tuple(frame_feats.shape)}.')
            return frame_feats

        features = feats['features']
        if features.dim() != 3:
            raise RuntimeError(
                'HeatmapIKM2MHead requires frame_features or flattened '
                f'features with shape (B, F*N, C), got {tuple(features.shape)}.')
        bsz, total_tokens, channels = features.shape
        if total_tokens % self.num_frames != 0:
            raise RuntimeError(
                f'Cannot split {total_tokens} tokens into '
                f'{self.num_frames} frames.')
        tokens_per_frame = total_tokens // self.num_frames
        return features.reshape(
            bsz, self.num_frames, tokens_per_frame, channels)

    def _flatten_frame_samples(self,
                               data_samples: Optional[List],
                               frames: int) -> Optional[List[Dict]]:
        if data_samples is None:
            return None

        flat_samples: List[Dict] = []
        for sample in data_samples:
            frame_gt = self._get_sample_field(sample, 'gt_keypoints_3d_frames')
            if frame_gt is None:
                center_gt = self._get_sample_field(sample, 'gt_keypoints_3d')
                if center_gt is None:
                    frame_gt = [None] * frames
                else:
                    frame_gt = center_gt.unsqueeze(0).expand(
                        frames, *center_gt.shape)

            for frame_id in range(frames):
                if isinstance(frame_gt, torch.Tensor):
                    gt = frame_gt[frame_id]
                else:
                    gt = frame_gt[frame_id]
                flat_samples.append({'gt_keypoints_3d': gt})
        return flat_samples

    @staticmethod
    def _get_sample_field(sample, name: str):
        if isinstance(sample, dict):
            return sample.get(name, None)
        if hasattr(sample, 'get'):
            value = sample.get(name, None)
            if value is not None:
                return value
        return getattr(sample, name, None)

    @staticmethod
    def _reshape_frame_outputs(outputs: Dict[str, torch.Tensor],
                               bsz: int,
                               frames: int,
                               prefix: str) -> Dict[str, torch.Tensor]:
        reshaped: Dict[str, torch.Tensor] = {}
        for key, value in outputs.items():
            if not torch.is_tensor(value):
                reshaped[key] = value
                continue
            if key.startswith('loss'):
                reshaped[key] = value
                continue
            if value.size(0) == bsz * frames:
                reshaped[f'{prefix}{key}_frames'] = value.reshape(
                    bsz, frames, *value.shape[1:])
            else:
                reshaped[key] = value
        return reshaped

    @staticmethod
    def _expose_center_frame(outputs: Dict[str, torch.Tensor],
                             center_id: int) -> None:
        for key, value in list(outputs.items()):
            if not key.endswith('_frames') or not torch.is_tensor(value):
                continue
            base_key = key[:-7]
            if value.dim() >= 2:
                outputs[base_key] = value[:, center_id]

