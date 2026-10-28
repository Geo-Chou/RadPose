from typing import List, Sequence, Union

import mmengine
import numpy as np
import torch
from mmcv import BaseTransform
from mmengine.structures import InstanceData
from numpy import dtype

from mmwave3d.registry import TRANSFORMS

from .structures import MmWaveDataSample, RadarPoints


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int, float],
) -> torch.Tensor:

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        if data.dtype is dtype("float64"):
            data = data.astype(np.float32)
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class Pack3DInputs(BaseTransform):
    INPUTS_KEYS = ["points"]
    INSTANCEDATA_3D_KEYS = ["gt_keypoints_3d"]

    def __init__(
        self,
        keys: tuple,
        meta_keys: tuple = (
            "num_frames",
            "radar_points_paths",
            "sample_name",
            "points_count_after_temporal_filter",
        ),
    ) -> None:
        self.keys = keys
        self.meta_keys = meta_keys

    def _remove_prefix(self, key: str) -> str:
        if key.startswith("gt_"):
            key = key[3:]
        return key

    def _stack_frame_keypoints(self, frame_keypoints: Sequence) -> torch.Tensor:
        tensors = [to_tensor(item).float() for item in frame_keypoints]
        if not tensors:
            return torch.zeros((0, 0, 0, 3), dtype=torch.float32)

        num_joints = 0
        coord_dim = 3
        max_person = 0
        for tensor in tensors:
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.numel() == 0:
                continue
            max_person = max(max_person, int(tensor.size(0)))
            num_joints = max(num_joints, int(tensor.size(1)))
            coord_dim = int(tensor.size(-1))
        if num_joints == 0:
            return torch.zeros((len(tensors), 0, 0, coord_dim), dtype=torch.float32)

        stacked = []
        for tensor in tensors:
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.size(-1) > coord_dim:
                tensor = tensor[..., :coord_dim]
            padded = tensor.new_zeros((max_person, num_joints, coord_dim))
            if tensor.numel() > 0:
                p = min(max_person, tensor.size(0))
                j = min(num_joints, tensor.size(1))
                d = min(coord_dim, tensor.size(-1))
                padded[:p, :j, :d] = tensor[:p, :j, :d]
            stacked.append(padded)
        return torch.stack(stacked, dim=0)

    def transform(self, results: Union[dict, List[dict]]) -> Union[dict, List[dict]]:

        if isinstance(results, list):
            if len(results) == 1:
                return self.pack_single_results(results[0])
            pack_results = []
            for single_result in results:
                pack_results.append(self.pack_single_results(single_result))
            return pack_results

        elif isinstance(results, dict):
            return self.pack_single_results(results)
        else:
            raise NotImplementedError

    def pack_single_results(self, results: dict) -> dict:

        if "points" in results:
            if isinstance(results["points"], RadarPoints):
                results["points"] = results["points"].tensor
            elif isinstance(results["points"], List):
                results["points"] = [p.tensor for p in results["points"]]

        for key in ["gt_keypoints_3d"]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])

        data_sample = MmWaveDataSample()
        gt_instances_3d = InstanceData()

        data_metas = {}
        for key in self.meta_keys:
            if key in results:
                data_metas[key] = results[key]

        data_sample.set_metainfo(data_metas)
        if "sample_name" in data_metas:
            data_sample.dump_name = data_metas["sample_name"]

        inputs = {}
        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                else:
                    raise NotImplementedError(
                        f"Please modified "
                        f"`Pack3DDetInputs` "
                        f"to put {key} to "
                        f"corresponding field"
                    )

        data_sample.gt_instances_3d = gt_instances_3d
        if "gt_keypoints_3d" in results:
            data_sample.gt_keypoints_3d = to_tensor(results["gt_keypoints_3d"])
        if "gt_keypoints_3d_frames" in results:
            data_sample.gt_keypoints_3d_frames = self._stack_frame_keypoints(
                results["gt_keypoints_3d_frames"]
            )
        if "eval_ann_info" in results:
            data_sample.eval_ann_info = results["eval_ann_info"]
        else:
            data_sample.eval_ann_info = None

        packed_results = dict()
        packed_results["data_samples"] = data_sample
        packed_results["inputs"] = inputs

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(keys={self.keys})"
        repr_str += f"(meta_keys={self.meta_keys})"
        return repr_str
