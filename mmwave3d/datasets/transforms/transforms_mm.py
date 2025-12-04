import random
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform, Compose, RandomResize, Resize
from mmdet.datasets.transforms import (PhotoMetricDistortion, RandomCrop,
                                       RandomFlip)
from mmengine import is_list_of, is_tuple_of

from mmwave3d.registry import TRANSFORMS
from mmwave3d.datasets.transforms import RadarPoints




@TRANSFORMS.register_module()
class PointSample(BaseTransform):
    """Point sample.

    Sampling data to a certain number.

    Required Keys:

    - points
    - pts_instance_mask (optional)
    - pts_semantic_mask (optional)

    Modified Keys:

    - points
    - pts_instance_mask (optional)
    - pts_semantic_mask (optional)

    Args:
        num_points (int): Number of points to be sampled.
        sample_range (float, optional): The range where to sample points.
            If not None, the points with depth larger than `sample_range` are
            prior to be sampled. Defaults to None.
        replace (bool): Whether the sampling is with or without replacement.
            Defaults to False.
    """

    def __init__(self,
                 num_points: int,
                 sample_range: Optional[float] = None,
                 replace: bool = False) -> None:
        self.num_points = num_points
        self.sample_range = sample_range
        self.replace = replace

    def _points_random_sampling(
        self,
        points: RadarPoints,
        num_samples: Union[int, float],
        sample_range: Optional[float] = None,
        replace: bool = False,
        return_choices: bool = False
    ) -> Union[Tuple[RadarPoints, np.ndarray], RadarPoints]:
        """Points random sampling.

        Sample points to a certain number.

        Args:
            points (:obj:`BasePoints`): 3D Points.
            num_samples (int, float): Number of samples to be sampled. If
                float, we sample random fraction of points from num_points
                to 100%.
            sample_range (float, optional): Indicating the range where the
                points will be sampled. Defaults to None.
            replace (bool): Sampling with or without replacement.
                Defaults to False.
            return_choices (bool): Whether return choice. Defaults to False.

        Returns:
            tuple[:obj:`BasePoints`, np.ndarray] | :obj:`BasePoints`:

                - points (:obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if isinstance(num_samples, float):
            assert num_samples < 1
            num_samples = int(
                np.random.uniform(self.num_points, 1.) * points.shape[0])

        if not replace:
            replace = (points.shape[0] < num_samples)
        point_range = range(len(points))
        if sample_range is not None and not replace:
            # Only sampling the near points when len(points) >= num_samples
            dist = np.linalg.norm(points.coord.numpy(), axis=1)
            far_inds = np.where(dist >= sample_range)[0]
            near_inds = np.where(dist < sample_range)[0]
            # in case there are too many far points
            if len(far_inds) > num_samples:
                far_inds = np.random.choice(
                    far_inds, num_samples, replace=False)
            point_range = near_inds
            num_samples -= len(far_inds)
        choices = np.random.choice(point_range, num_samples, replace=replace)
        if sample_range is not None and not replace:
            choices = np.concatenate((far_inds, choices))
            # Shuffle points after sampling
            np.random.shuffle(choices)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def transform(self, input_dict: dict) -> dict:
        """Transform function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask'
            and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']
        input_dict['points']
        for idx, point in enumerate(points):
            point, choices = self._points_random_sampling(
                point,
                self.num_points,
                self.sample_range,
                self.replace,
                return_choices=True)
            input_dict['points'][idx] = point

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points},'
        repr_str += f' sample_range={self.sample_range},'
        repr_str += f' replace={self.replace})'

        return repr_str