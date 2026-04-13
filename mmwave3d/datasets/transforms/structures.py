# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import abstractmethod
from typing import Iterator, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor
from mmengine.structures import BaseDataElement, InstanceData


class RadarPoints:
    """Base class for RadarPoints.
    """

    def __init__(self,
                 tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
                 points_dim: int = 3,
                 attribute_dims: Optional[dict] = None) -> None:
        if isinstance(tensor, Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does
            # not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, points_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == points_dim, \
            ('The points dimension must be 2 and the length of the last '
             f'dimension must be {points_dim}, but got points with shape '
             f'{tensor.shape}.')

        self.tensor = tensor.clone()
        self.points_dim = points_dim
        self.attribute_dims = attribute_dims
        self.shape = self.tensor.shape


    @property
    def coord(self) -> Tensor:
        """Tensor: Coordinates of each point in shape (N, 3)."""
        return self.tensor[:, :3]

    @coord.setter
    def coord(self, tensor: Union[Tensor, np.ndarray]) -> None:
        """Set the coordinates of each point.

        Args:
            tensor (Tensor or np.ndarray): Coordinates of each point with shape
                (N, 3).
        """
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        self.tensor[:, :3] = tensor

    @property
    def amp(self) -> Union[Tensor, None]:
        """Tensor or None: Returns a vector with phase of each point in shape
        (N, 1)."""
        if self.attribute_dims is not None and \
                'amp' in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims['amp']]
        else:
            return None
    @amp.setter
    def amp(self, tensor: Union[Tensor, np.ndarray]) -> None:
        """Set the amplitude of each point.

        Args:
            tensor (Tensor or np.ndarray): amplitude of each point with shape
                (N, 1).
        """
        try:
            tensor = tensor.reshape(self.shape[0], 1)
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and \
                'amp' in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims['amp']] = tensor
        else:
            # add phase attribute
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor], dim=1)
            self.attribute_dims.update(
                dict(amp=[attr_dim]))
            self.points_dim += 1


    @property
    def phase(self) -> Union[Tensor, None]:
        """Tensor or None: Returns a vector with phase of each point in shape
        (N, 1)."""
        if self.attribute_dims is not None and \
                'phase' in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims['phase']]
        else:
            return None

    @phase.setter
    def phase(self, tensor: Union[Tensor, np.ndarray]) -> None:
        """Set the phase of each point.

        Args:
            tensor (Tensor or np.ndarray): phase of each point with shape
                (N, 1).
        """
        try:
            tensor = tensor.reshape(self.shape[0], 1)
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f'got unexpected shape {tensor.shape}')
        if not isinstance(tensor, Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and \
                'phase' in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims['phase']] = tensor
        else:
            # add phase attribute
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor], dim=1)
            self.attribute_dims.update(
                dict(phase=[attr_dim]))
            self.points_dim += 1

    def translate(self, trans_vector: Union[Tensor, np.ndarray]) -> None:
        """Translate points with the given translation vector.

        Args:
            trans_vector (Tensor or np.ndarray): Translation vector of size 3
                or nx3.
        """
        if not isinstance(trans_vector, Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        trans_vector = trans_vector.squeeze(0)
        if trans_vector.dim() == 1:
            assert trans_vector.shape[0] == 3
        elif trans_vector.dim() == 2:
            assert trans_vector.shape[0] == self.tensor.shape[0] and \
                trans_vector.shape[1] == 3
        else:
            raise NotImplementedError(
                f'Unsupported translation vector of shape {trans_vector.shape}'
            )
        self.tensor[:, :3] += trans_vector

    def in_range_3d(
            self, point_range: Union[Tensor, np.ndarray,
                                     Sequence[float]]) -> Tensor:
        """Check whether the points are in the given range.

        Args:
            point_range (Tensor or np.ndarray or Sequence[float]): The range of
                point (x_min, y_min, z_min, x_max, y_max, z_max).

        Note:
            In the original implementation of SECOND, checking whether a box in
            the range checks whether the points are in a convex polygon, we try
            to reduce the burden for simpler cases.

        Returns:
            Tensor: A binary vector indicating whether each point is inside the
            reference range.
        """
        in_range_flags = ((self.tensor[:, 0] > point_range[0])
                          & (self.tensor[:, 1] > point_range[1])
                          & (self.tensor[:, 2] > point_range[2])
                          & (self.tensor[:, 0] < point_range[3])
                          & (self.tensor[:, 1] < point_range[4])
                          & (self.tensor[:, 2] < point_range[5]))
        return in_range_flags

    def scale(self, scale_factor: float) -> None:
        """Scale the points with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the points.
        """
        self.tensor[:, :3] *= scale_factor

    def __getitem__(
            self, item: Union[int, tuple, slice, np.ndarray,
                              Tensor]) -> 'RadarPoints':
        """
        Args:
            item (int or tuple or slice or np.ndarray or Tensor): Index of
                points.

        Note:
            The following usage are allowed:

            1. `new_points = points[3]`: Return a `Points` that contains only
               one point.
            2. `new_points = points[2:10]`: Return a slice of points.
            3. `new_points = points[vector]`: Whether vector is a
               torch.BoolTensor with `length = len(points)`. Nonzero elements
               in the vector will be selected.
            4. `new_points = points[3:11, vector]`: Return a slice of points
               and attribute dims.
            5. `new_points = points[4:12, 2]`: Return a slice of points with
               single attribute.

            Note that the returned Points might share storage with this Points,
            subject to PyTorch's indexing semantics.

        Returns:
            :obj:`BasePoints`: A new object of :class:`BasePoints` after
            indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].view(1, -1),
                points_dim=self.points_dim,
                attribute_dims=self.attribute_dims)
        elif isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[1], slice):
                start = 0 if item[1].start is None else item[1].start
                stop = self.tensor.shape[1] \
                    if item[1].stop is None else item[1].stop
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
                set(item[1]).intersection(set(range(3, self.tensor.shape[1]))))
            if self.attribute_dims is not None:
                attribute_dims = self.attribute_dims.copy()
                for key in self.attribute_dims.keys():
                    cur_attribute_dims = attribute_dims[key]
                    if isinstance(cur_attribute_dims, int):
                        cur_attribute_dims = [cur_attribute_dims]
                    intersect_attr = list(
                        set(cur_attribute_dims).intersection(set(keep_dims)))
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
            raise NotImplementedError(f'Invalid slice {item}!')

        assert p.dim() == 2, \
            f'Indexing on Points with {item} failed to return a matrix!'
        return original_type(
            p, points_dim=p.shape[1], attribute_dims=attribute_dims)

    def __len__(self) -> int:
        """int: Number of points in the current object."""
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        """str: Return a string that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.tensor) + ')'

    @classmethod
    def cat(cls, points_list: Sequence['RadarPoints']) -> 'RadarPoints':
        """Concatenate a list of Points into a single Points.

        Args:
            points_list (Sequence[:obj:`BasePoints`]): List of points.

        Returns:
            :obj:`BasePoints`: The concatenated points.
        """
        assert isinstance(points_list, (list, tuple))
        if len(points_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(points, cls) for points in points_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned points never share storage with input
        cat_points = cls(
            torch.cat([p.tensor for p in points_list], dim=0),
            points_dim=points_list[0].points_dim,
            attribute_dims=points_list[0].attribute_dims)
        return cat_points

    def numpy(self) -> np.ndarray:
        """Reload ``numpy`` from self.tensor."""
        return self.tensor.numpy()

    def to(self, device: Union[str, torch.device], *args,
           **kwargs) -> 'RadarPoints':
        """Convert current points to a specific device.

        Args:
            device (str or :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BasePoints`: A new points object on the specific device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.to(device, *args, **kwargs),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

    def cpu(self) -> 'RadarPoints':
        """Convert current points to cpu device.

        Returns:
            :obj:`BasePoints`: A new points object on the cpu device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.cpu(),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

    def cuda(self, *args, **kwargs) -> 'RadarPoints':
        """Convert current points to cuda device.

        Returns:
            :obj:`BasePoints`: A new points object on the cuda device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.cuda(*args, **kwargs),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

    def clone(self) -> 'RadarPoints':
        """Clone the points.

        Returns:
            :obj:`BasePoints`: Point object with the same properties as self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.clone(),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

    def detach(self) -> 'RadarPoints':
        """Detach the points.

        Returns:
            :obj:`BasePoints`: Point object with the same properties as self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.detach(),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)

    @property
    def device(self) -> torch.device:
        """torch.device: The device of the points are on."""
        return self.tensor.device

    def __iter__(self) -> Iterator[Tensor]:
        """Yield a point as a Tensor at a time.

        Returns:
            Iterator[Tensor]: A point of shape (points_dim, ).
        """
        yield from self.tensor



    def new_point(
        self, data: Union[Tensor, np.ndarray, Sequence[Sequence[float]]]
    ) -> 'RadarPoints':
        """Create a new point object with data.

        The new point and its tensor has the similar properties as self and
        self.tensor, respectively.

        Args:
            data (Tensor or np.ndarray or Sequence[Sequence[float]]): Data to
                be copied.

        Returns:
            :obj:`BasePoints`: A new point object with ``data``, the object's
            other properties are similar to ``self``.
        """
        new_tensor = self.tensor.new_tensor(data) \
            if not isinstance(data, Tensor) else data.to(self.device)
        original_type = type(self)
        return original_type(
            new_tensor,
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims)
    
    def flip(self, bev_direction: str = 'horizontal') -> None:
        """Flip the points along given BEV direction.

        Args:
            bev_direction (str): Flip direction (horizontal or vertical).
                Defaults to 'horizontal'.
        """
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            self.tensor[:, 1] = -self.tensor[:, 1]
        elif bev_direction == 'vertical':
            self.tensor[:, 0] = -self.tensor[:, 0]


class MmWaveDataSample(BaseDataElement):
    """Minimal data sample for mmWave human keypoint detection.

    Only keeps the pieces of ``Det3DDataSample`` that the project uses:

    - ``gt_instances_3d`` (:obj:`InstanceData`)
    - ``pred_instances_3d`` (:obj:`InstanceData`)
    - ``gt_keypoints_3d`` (:obj:`Tensor`)
    - ``pred_keypoints_3d`` (:obj:`Tensor`)
    - ``eval_ann_info`` (dict or ``None``)
    """

    @property
    def gt_instances_3d(self) -> InstanceData:
        return self._gt_instances_3d

    @gt_instances_3d.setter
    def gt_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, '_gt_instances_3d', dtype=InstanceData)

    @gt_instances_3d.deleter
    def gt_instances_3d(self) -> None:
        del self._gt_instances_3d

    @property
    def pred_instances_3d(self) -> InstanceData:
        return self._pred_instances_3d

    @pred_instances_3d.setter
    def pred_instances_3d(self, value: InstanceData) -> None:
        self.set_field(value, '_pred_instances_3d', dtype=InstanceData)

    @pred_instances_3d.deleter
    def pred_instances_3d(self) -> None:
        del self._pred_instances_3d

    @property
    def gt_keypoints_3d(self) -> Tensor:
        return self._gt_keypoints_3d

    @gt_keypoints_3d.setter
    def gt_keypoints_3d(self, value: Tensor) -> None:
        self.set_field(value, '_gt_keypoints_3d', dtype=Tensor)

    @gt_keypoints_3d.deleter
    def gt_keypoints_3d(self) -> None:
        del self._gt_keypoints_3d

    @property
    def pred_keypoints_3d(self) -> Tensor:
        return self._pred_keypoints_3d

    @pred_keypoints_3d.setter
    def pred_keypoints_3d(self, value: Tensor) -> None:
        self.set_field(value, '_pred_keypoints_3d', dtype=Tensor)

    @pred_keypoints_3d.deleter
    def pred_keypoints_3d(self) -> None:
        del self._pred_keypoints_3d

    @property
    def eval_ann_info(self) -> Optional[dict]:
        return self._eval_ann_info

    @eval_ann_info.setter
    def eval_ann_info(self, value: Optional[dict]) -> None:
        self.set_field(value, '_eval_ann_info', dtype=(dict, type(None)))

    @eval_ann_info.deleter
    def eval_ann_info(self) -> None:
        del self._eval_ann_info

    @property
    def dump_name(self) -> Optional[str]:
        return getattr(self, '_dump_name', None)

    @dump_name.setter
    def dump_name(self, value: Optional[str]) -> None:
        self.set_field(value, '_dump_name', dtype=(str, type(None)))

    @dump_name.deleter
    def dump_name(self) -> None:
        if hasattr(self, '_dump_name'):
            del self._dump_name
