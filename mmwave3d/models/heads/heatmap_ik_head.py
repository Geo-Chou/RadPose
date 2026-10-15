from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from mmwave3d.registry import MODELS


@MODELS.register_module()
class HeatmapIKHead(nn.Module):
    def __init__(self, heatmap_head: Dict, ik_head: Dict) -> None:
        super().__init__()
        self.heatmap_head = MODELS.build(heatmap_head)
        self.ik_head = MODELS.build(ik_head)

    def forward(
        self, feats: Dict[str, torch.Tensor], data_samples: Optional[List] = None
    ) -> Dict[str, torch.Tensor]:
        heatmap_outputs = self.heatmap_head(feats, data_samples)
        ik_outputs = self.ik_head(feats, heatmap_outputs, data_samples)

        outputs = dict(heatmap_outputs)
        outputs["heatmap_keypoints"] = heatmap_outputs.get("keypoints", None)
        if "ik_keypoints" in ik_outputs:
            outputs["keypoints"] = ik_outputs["ik_keypoints"]

        outputs.update(ik_outputs)
        return outputs
