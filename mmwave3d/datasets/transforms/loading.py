import copy
from typing import List, Optional, Union

import mmcv
import mmengine
import numpy as np
from mmcv.transforms import LoadImageFromFile
from mmcv.transforms.base import BaseTransform
# from mmdet.datasets.transforms import LoadAnnotations
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmengine.fileio import get


from mmwave3d.registry import TRANSFORMS
from .structures import RadarPoints


@TRANSFORMS.register_module()
class LoadMmPointsFromFile(BaseTransform):
    """Load Points From File.

    Required Keys:

    - lidar_points (dict)

        - lidar_path (str)

    Added Keys:

    - points (np.float32)

    Args:
    """

    def __init__(self,
                 load_dim: int = 6,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 use_amplitude: bool = False,
                 use_phase: bool = False,
                 norm_intensity: bool = False,
                 backend_args: Optional[dict] = None) -> None:
        self.use_phase = use_phase
        self.use_amplitude = use_amplitude
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'

        self.load_dim = load_dim
        self.use_dim = use_dim
        self.norm_intensity = norm_intensity
        self.backend_args = backend_args

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        try:
            # pts_bytes = get(pts_filename, backend_args=self.backend_args)
            # points = np.frombuffer(pts_bytes, dtype=np.float32)
            radar_data = np.load(pts_filename)
            points_mm = radar_data["spherical"].astype(np.float32)
            amp = radar_data["amplitude"].astype(np.float32)
            phase = radar_data["phase"].astype(np.float32)
            if amp.ndim == 1:
                amp = amp[:, None]
            if phase.ndim == 1:
                phase = phase[:, None]
            # phase = np.stack((real, imag), axis=-1)
            # print(points_mm.shape, amp.shape, phase.shape)
            points = np.concatenate((points_mm, amp, phase), axis=-1)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        # all_points = []
        # frame_ids = []
        pts_file_paths = results['radar_points_paths']
        results['points'] = []
        for fid, pts_file_path in enumerate(pts_file_paths):
            points = self._load_points(pts_file_path)
            # all_points.append(points)
            # frame_ids.append(
            #     np.full((points.shape[0], 1), fid, dtype=np.float32))
        # points = np.concatenate(all_points, axis=0).astype(np.float32)
        # frame_ids = np.concatenate(frame_ids, axis=0)
            points = points.reshape(-1, self.load_dim)
            points = points[:, self.use_dim]
            attribute_dims = None


            if self.use_phase:
                assert len(self.use_dim) >= 5
                if attribute_dims is None:
                    attribute_dims = dict()
                attribute_dims.update(
                    dict(phase = [points.shape[1] - 1 # imag
                                ]
                    ))
                
            if self.use_amplitude:
                assert len(self.use_dim) >= 5
                if attribute_dims is None:
                    attribute_dims = dict()
                attribute_dims.update(
                    dict(amp = [points.shape[1] - 2 # imag
                                ]
                    ))
            
            points = RadarPoints(
                points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        
            results['points'].append(points) 
        # print("Before Pack:", type(results["points"]))

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        repr_str += f'norm_intensity={self.norm_intensity})'
        return repr_str



@TRANSFORMS.register_module()
class LoadAnnotations3D(MMCV_LoadAnnotations):
    """Load Annotations3D.
    """

    def __init__(self,
                 with_keypoints_3d: bool = True,
                 eval: bool = False,
                 **kwargs) -> None:
        super(LoadAnnotations3D, self).__init__(**kwargs)
        self.with_keypoints_3d = with_keypoints_3d
        self.eval = eval
    
    def _load_keypoints_3d(self, results):
        if not self.eval:
            results['gt_keypoints_3d'] = results['ann_info']['gt_keypoints_3d']
        if self.eval:
            results['gt_keypoints_3d'] = results['eval_ann_info']['gt_keypoints_3d']
        return results
        # pass

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
            semantic segmentation annotations.
        """
        if self.with_keypoints_3d:
            results = self._load_keypoints_3d(results)
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_keypoints_3d={self.with_keypoints_3d}, '

        return repr_str