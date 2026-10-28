from typing import Dict, List, Optional

import torch
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement

from mmwave3d.datasets.transforms.structures import MmWaveDataSample
from mmwave3d.registry import MODELS


@MODELS.register_module()
class MmKeypoints(BaseModel):
    def __init__(
        self,
        data_preprocessor: Dict,
        transformer: Dict,
        head: Dict,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.voxelizer = self.data_preprocessor
        self.transformer = MODELS.build(transformer)
        self.head = MODELS.build(head)

    def extract_feat(
        self, batch_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        feats = self.transformer(batch_inputs)
        return feats

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        data_samples: Optional[List] = None,
        mode: str = "tensor",
    ):
        if mode == "loss":
            return self.loss(inputs, data_samples)
        elif mode == "predict":
            return self.predict(inputs, data_samples)
        elif mode == "tensor":
            return self._forward(inputs, data_samples)
        else:
            raise ValueError(f"Invalid mode {mode}")

    def _forward(
        self, inputs: Dict[str, torch.Tensor], data_samples: Optional[List] = None
    ):
        feats = self.extract_feat(inputs)
        outputs = self.head(feats, data_samples)
        return outputs

    def loss(self, inputs: Dict[str, torch.Tensor], data_samples: List):
        points_counts = self._get_points_count_from_samples(data_samples)
        if points_counts is None:
            points_counts = self._compute_points_count(inputs)
        if points_counts is not None:
            self.head._last_points_count = points_counts

        outputs = self._forward(inputs, data_samples)

        return {k: v for k, v in outputs.items() if k.startswith("loss")}

    def _get_points_count_from_samples(self, data_samples: List):
        if not data_samples:
            return None
        counts = []
        for sample in data_samples:
            metainfo = getattr(sample, "metainfo", None)
            if metainfo is None:
                counts.append(None)
                continue
            counts.append(metainfo.get("points_count_after_temporal_filter", None))
        if all(c is None for c in counts):
            return None
        return [int(c or 0) for c in counts]

    def _compute_points_count(self, inputs: Dict[str, torch.Tensor]):
        points = inputs.get("points", None)
        if points is None:
            return None

        def count_points(sample_points) -> int:
            if sample_points is None:
                return 0
            if isinstance(sample_points, torch.Tensor):
                return int(sample_points.shape[0])
            if isinstance(sample_points, (list, tuple)):
                return sum(count_points(p) for p in sample_points)
            return 0

        if isinstance(points, torch.Tensor):
            if points.dim() == 3:
                return [int(points.shape[1])] * int(points.shape[0])
            return [int(points.shape[0])]
        if isinstance(points, (list, tuple)):
            return [count_points(sample_points) for sample_points in points]
        return None

    def predict(
        self, inputs: Dict[str, torch.Tensor], data_samples: Optional[List] = None
    ) -> List[BaseDataElement]:
        with torch.no_grad():
            outputs = self._forward(inputs, data_samples)
        keypoints = outputs.get("keypoints", None)
        heatmap_keypoints = outputs.get("heatmap_keypoints", None)
        scores = outputs.get("scores", None)
        person_scores = outputs.get("person_scores", None)
        heatmaps = outputs.get("heatmaps", None)
        elevation_maps = outputs.get("elevation_maps", None)
        re_heatmaps = outputs.get("re_heatmaps", None)
        ik_keypoints = outputs.get("ik_keypoints", None)
        batch = 0
        if heatmap_keypoints is not None:
            batch = heatmap_keypoints.size(0)
        elif keypoints is not None:
            batch = keypoints.size(0)
        elif data_samples is not None:
            batch = len(data_samples)

        if data_samples is None:
            pred_samples: List[MmWaveDataSample] = [
                MmWaveDataSample() for _ in range(batch)
            ]
        else:
            pred_samples = []
            for idx in range(batch):
                if idx < len(data_samples):
                    pred_samples.append(data_samples[idx].clone())
                else:
                    pred_samples.append(MmWaveDataSample())

        for idx in range(batch):
            if heatmap_keypoints is not None:
                pred_samples[idx].pred_keypoints_3d = heatmap_keypoints[idx]
                pred_samples[idx].pred_keypoints = heatmap_keypoints[idx]
            elif keypoints is not None:
                pred_samples[idx].pred_keypoints_3d = keypoints[idx]
                pred_samples[idx].pred_keypoints = keypoints[idx]
            if keypoints is not None:
                pred_samples[idx].keypoints = keypoints[idx]
            if ik_keypoints is not None:
                pred_samples[idx].pred_ik_keypoints_3d = ik_keypoints[idx]
                pred_samples[idx].ik_keypoints = ik_keypoints[idx]
            if scores is not None:
                pred_samples[idx].scores = scores[idx]
                pred_samples[idx].pred_scores = scores[idx]
            if person_scores is not None:
                pred_samples[idx].person_scores = torch.sigmoid(person_scores[idx])

            if heatmaps is not None:
                pred_samples[idx].pred_heatmaps = heatmaps[idx]
                pred_samples[idx].heatmaps = heatmaps[idx]
            if elevation_maps is not None:
                pred_samples[idx].pred_elevation_maps = elevation_maps[idx]
                pred_samples[idx].elevation_maps = elevation_maps[idx]

            if re_heatmaps is not None:
                pred_samples[idx].pred_re_heatmaps = re_heatmaps[idx]
                pred_samples[idx].re_heatmaps = re_heatmaps[idx]

        loss_items = {k: v for k, v in outputs.items() if k.startswith("loss")}
        if loss_items:
            loss_sample = BaseDataElement()
            loss_sample.loss = loss_items
            pred_samples.append(loss_sample)
        return pred_samples
