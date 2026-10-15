from typing import Iterator, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.structures import BaseDataElement, InstanceData
from torch import Tensor


class RadarPoints:
    def __init__(
        self,
        tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
        points_dim: int = 3,
        attribute_dims: Optional[dict] = None,
    ) -> None:
        if isinstance(tensor, Tensor):
            device = tensor.device
        else:
            device = torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((-1, points_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == points_dim, (
            "The points dimension must be 2 and the length of the last "
            f"dimension must be {points_dim}, but got points with shape "
            f"{tensor.shape}."
        )

        self.tensor = tensor.clone()
        self.points_dim = points_dim
        self.attribute_dims = attribute_dims
        self.shape = self.tensor.shape

    @property
    def coord(self) -> Tensor:
        return self.tensor[:, :3]

    @coord.setter
    def coord(self, tensor: Union[Tensor, np.ndarray]) -> None:
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):
            raise ValueError(f"got unexpected shape {tensor.shape}")
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        self.tensor[:, :3] = tensor

    @property
    def amp(self) -> Union[Tensor, None]:
        if self.attribute_dims is not None and "amp" in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims["amp"]]
        else:
            return None

    @amp.setter
    def amp(self, tensor: Union[Tensor, np.ndarray]) -> None:
        try:
            tensor = tensor.reshape(self.shape[0], 1)
        except (RuntimeError, ValueError):
            raise ValueError(f"got unexpected shape {tensor.shape}")
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and "amp" in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims["amp"]] = tensor
        else:
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor], dim=1)
            self.attribute_dims.update(dict(amp=[attr_dim]))
            self.points_dim += 1

    @property
    def phase(self) -> Union[Tensor, None]:
        if self.attribute_dims is not None and "phase" in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims["phase"]]
        else:
            return None

    @phase.setter
    def phase(self, tensor: Union[Tensor, np.ndarray]) -> None:
        try:
            tensor = tensor.reshape(self.shape[0], 1)
        except (RuntimeError, ValueError):
            raise ValueError(f"got unexpected shape {tensor.shape}")
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and "phase" in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims["phase"]] = tensor
        else:
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor], dim=1)
            self.attribute_dims.update(dict(phase=[attr_dim]))
            self.points_dim += 1

    def translate(self, trans_vector: Union[Tensor, np.ndarray]) -> None:
        if not isinstance(trans_vector, Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        trans_vector = trans_vector.squeeze(0)
        if trans_vector.dim() == 1:
            assert trans_vector.shape[0] == 3
        elif trans_vector.dim() == 2:
            assert (
                trans_vector.shape[0] == self.tensor.shape[0]
                and trans_vector.shape[1] == 3
            )
        else:
            raise NotImplementedError(
                f"Unsupported translation vector of shape {trans_vector.shape}"
            )
        self.tensor[:, :3] += trans_vector

    def in_range_3d(
        self, point_range: Union[Tensor, np.ndarray, Sequence[float]]
    ) -> Tensor:
        in_range_flags = (
            (self.tensor[:, 0] > point_range[0])
            & (self.tensor[:, 1] > point_range[1])
            & (self.tensor[:, 2] > point_range[2])
            & (self.tensor[:, 0] < point_range[3])
            & (self.tensor[:, 1] < point_range[4])
            & (self.tensor[:, 2] < point_range[5])
        )
        return in_range_flags

    def scale(self, scale_factor: float) -> None:
        self.tensor[:, :3] *= scale_factor

    def __getitem__(
        self, item: Union[int, tuple, slice, np.ndarray, Tensor]
    ) -> "RadarPoints":
        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].view(1, -1),
                points_dim=self.points_dim,
                attribute_dims=self.attribute_dims,
            )
        elif isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[1], slice):
                start = 0 if item[1].start is None else item[1].start
                stop = self.tensor.shape[1] if item[1].stop is None else item[1].stop
                step = 1 if item[1].step is None else item[1].step
                item = list(item)
                item[1] = list(range(start, stop, step))
                item = tuple(item)
            elif isinstance(item[1], int):
                item = list(item)
                item[1] = [item[1]]
                item = tuple(item)
            p = self.tensor[item[0], item[1]]

            keep_dims = list(
                set(item[1]).intersection(set(range(3, self.tensor.shape[1])))
            )
            if self.attribute_dims is not None:
                attribute_dims = self.attribute_dims.copy()
                for key in self.attribute_dims.keys():
                    cur_attribute_dims = attribute_dims[key]
                    if isinstance(cur_attribute_dims, int):
                        cur_attribute_dims = [cur_attribute_dims]
                    intersect_attr = list(
                        set(cur_attribute_dims).intersection(set(keep_dims))
                    )
                    if len(intersect_attr) == 1:
                        attribute_dims[key] = intersect_attr[0]
                    elif len(intersect_attr) > 1:
                        attribute_dims[key] = intersect_attr
                    else:
                        attribute_dims.pop(key)
            else:
                attribute_dims = None
        elif isinstance(item, (slice, np.ndarray, Tensor)):
            p = self.tensor[item]
            attribute_dims = self.attribute_dims
        else:
            raise NotImplementedError(f"Invalid slice {item}!")

        assert p.dim() == 2, (
            f"Indexing on Points with {item} failed to return a matrix!"
        )
        return original_type(p, points_dim=p.shape[1], attribute_dims=attribute_dims)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(\n    " + str(self.tensor) + ")"

    @classmethod
    def cat(cls, points_list: Sequence["RadarPoints"]) -> "RadarPoints":
        assert isinstance(points_list, (list, tuple))
        if len(points_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(points, cls) for points in points_list)

        cat_points = cls(
            torch.cat([p.tensor for p in points_list], dim=0),
            points_dim=points_list[0].points_dim,
            attribute_dims=points_list[0].attribute_dims,
        )
        return cat_points

    def numpy(self) -> np.ndarray:
        return self.tensor.numpy()

    def to(self, device: Union[str, torch.device], *args, **kwargs) -> "RadarPoints":
        original_type = type(self)
        return original_type(
            self.tensor.to(device, *args, **kwargs),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims,
        )

    def cpu(self) -> "RadarPoints":
        original_type = type(self)
        return original_type(
            self.tensor.cpu(),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims,
        )

    def cuda(self, *args, **kwargs) -> "RadarPoints":
        original_type = type(self)
        return original_type(
            self.tensor.cuda(*args, **kwargs),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims,
        )

    def clone(self) -> "RadarPoints":
        original_type = type(self)
        return original_type(
            self.tensor.clone(),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims,
        )

    def detach(self) -> "RadarPoints":
        original_type = type(self)
        return original_type(
            self.tensor.detach(),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims,
        )

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __iter__(self) -> Iterator[Tensor]:
        yield from self.tensor

    def new_point(
        self, data: Union[Tensor, np.ndarray, Sequence[Sequence[float]]]
    ) -> "RadarPoints":
        new_tensor = (
            self.tensor.new_tensor(data)
            if not isinstance(data, Tensor)
            else data.to(self.device)
        )
        original_type = type(self)
        return original_type(
            new_tensor, points_dim=self.points_dim, attribute_dims=self.attribute_dims
        )

    def flip(self, bev_direction: str = "horizontal") -> None:
        assert bev_direction in ("horizontal", "vertical")
        if bev_direction == "horizontal":
            self.tensor[:, 1] = -self.tensor[:, 1]
        elif bev_direction == "vertical":
            self.tensor[:, 0] = -self.tensor[:, 0]


class MmWaveDataSample(BaseDataElement):
    @property
    def gt_instances_3d(self) -> InstanceData:
        return self._gt_instances_3d

    @gt_instances_3d.setter
    def gt_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, "_gt_instances_3d", dtype=InstanceData)

    @gt_instances_3d.deleter
    def gt_instances_3d(self) -> None:
        del self._gt_instances_3d

    @property
    def pred_instances_3d(self) -> InstanceData:
        return self._pred_instances_3d

    @pred_instances_3d.setter
    def pred_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, "_pred_instances_3d", dtype=InstanceData)

    @pred_instances_3d.deleter
    def pred_instances_3d(self) -> None:
        del self._pred_instances_3d

    @property
    def gt_keypoints_3d(self) -> Tensor:
        return self._gt_keypoints_3d

    @gt_keypoints_3d.setter
    def gt_keypoints_3d(self, value: Tensor) -> None:
        self.set_field(value, "_gt_keypoints_3d", dtype=Tensor)

    @gt_keypoints_3d.deleter
    def gt_keypoints_3d(self) -> None:
        del self._gt_keypoints_3d

    @property
    def gt_keypoints_3d_frames(self) -> Tensor:
        return self._gt_keypoints_3d_frames

    @gt_keypoints_3d_frames.setter
    def gt_keypoints_3d_frames(self, value: Tensor) -> None:
        self.set_field(value, "_gt_keypoints_3d_frames", dtype=Tensor)

    @gt_keypoints_3d_frames.deleter
    def gt_keypoints_3d_frames(self) -> None:
        del self._gt_keypoints_3d_frames

    @property
    def pred_keypoints_3d(self) -> Tensor:
        return self._pred_keypoints_3d

    @pred_keypoints_3d.setter
    def pred_keypoints_3d(self, value: Tensor) -> None:
        self.set_field(value, "_pred_keypoints_3d", dtype=Tensor)

    @pred_keypoints_3d.deleter
    def pred_keypoints_3d(self) -> None:
        del self._pred_keypoints_3d

    @property
    def eval_ann_info(self) -> Optional[dict]:
        return self._eval_ann_info

    @eval_ann_info.setter
    def eval_ann_info(self, value: Optional[dict]) -> None:
        self.set_field(value, "_eval_ann_info", dtype=(dict, type(None)))

    @eval_ann_info.deleter
    def eval_ann_info(self) -> None:
        del self._eval_ann_info

    @property
    def dump_name(self) -> Optional[str]:
        return getattr(self, "_dump_name", None)

    @dump_name.setter
    def dump_name(self, value: Optional[str]) -> None:
        self.set_field(value, "_dump_name", dtype=(str, type(None)))

    @dump_name.deleter
    def dump_name(self) -> None:
        if hasattr(self, "_dump_name"):
            del self._dump_name
