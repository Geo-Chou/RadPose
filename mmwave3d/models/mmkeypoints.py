from typing import Dict, List, Optional

import torch
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement

from mmwave3d.datasets.transforms.structures import MmWaveDataSample
from mmwave3d.registry import MODELS


@MODELS.register_module()
class MmKeypoints(BaseModel):
    """Full mmWave keypoint pipeline: voxelization -> transformer -> head."""

    def __init__(self,
                 data_preprocessor: Dict,
                 transformer: Dict,
                 head: Dict,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.voxelizer = self.data_preprocessor
        self.transformer = MODELS.build(transformer)
        self.head = MODELS.build(head)

    def extract_feat(self, batch_inputs: Dict[str, torch.Tensor]
                     ) -> Dict[str, torch.Tensor]:
        """Alias to keep compatibility with BaseModel hooks."""
        feats = self.transformer(batch_inputs)
        return feats

    def forward(self,
                inputs: Dict[str, torch.Tensor],
                data_samples: Optional[List] = None,
                mode: str = 'tensor'):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise ValueError(f'Invalid mode {mode}')

    def _forward(self,
                 inputs: Dict[str, torch.Tensor],
                 data_samples: Optional[List] = None):
        feats = self.extract_feat(inputs)
        outputs = self.head(feats, data_samples)
        return outputs

    def loss(self, inputs: Dict[str, torch.Tensor], data_samples: List):
        outputs = self._forward(inputs, data_samples)
        # print('succeed in this loop')
        return {k: v for k, v in outputs.items() if k.startswith('loss')}

    def predict(self,
                inputs: Dict[str, torch.Tensor],
                data_samples: Optional[List] = None) -> List[BaseDataElement]:
        with torch.no_grad():
            outputs = self._forward(inputs, data_samples)
        keypoints = outputs.get('keypoints', None)
        scores = outputs.get('scores', None)
        batch = 0
        if keypoints is not None:
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
            if keypoints is not None:
                pred_samples[idx].keypoints = keypoints[idx]
            if scores is not None:
                pred_samples[idx].scores = scores[idx]

        loss_items = {
            k: v
            for k, v in outputs.items()
            if k.startswith('loss')
        }
        if loss_items:
            loss_sample = BaseDataElement()
            loss_sample.loss = loss_items
            pred_samples.append(loss_sample)
        return pred_samples
