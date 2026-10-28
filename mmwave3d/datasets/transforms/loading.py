from typing import List, Optional, Union

import mmengine
import numpy as np
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms.base import BaseTransform

from mmwave3d.registry import TRANSFORMS

from .structures import RadarPoints


@TRANSFORMS.register_module()
class LoadMmPointsFromFile(BaseTransform):
    def __init__(
        self,
        load_dim: int = 6,
        use_dim: Union[int, List[int]] = [0, 1, 2],
        use_amplitude: bool = False,
        use_phase: bool = False,
        normalize_amplitude: bool = False,
        normalize_phase: bool = False,
        norm_intensity: bool = False,
        backend_args: Optional[dict] = None,
    ) -> None:
        self.use_phase = use_phase
        self.use_amplitude = use_amplitude
        self.normalize_amplitude = normalize_amplitude
        self.normalize_phase = normalize_phase
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, (
            f"Expect all used dimensions < {load_dim}, got {use_dim}"
        )

        self.load_dim = load_dim
        self.use_dim = use_dim
        self.norm_intensity = norm_intensity
        self.backend_args = backend_args

    def _load_points(self, pts_filename: str) -> np.ndarray:
        try:
            radar_data = np.load(pts_filename)
            points_mm = radar_data["spherical"].astype(np.float32)
            amp = radar_data["amplitude"].astype(np.float32)
            phase = radar_data["phase"].astype(np.float32)
            if amp.ndim == 1:
                amp = amp[:, None]
            if phase.ndim == 1:
                phase = phase[:, None]
            if self.normalize_amplitude:
                amp = self._normalize_feature(amp)
            if self.normalize_phase:
                phase = self._normalize_feature(phase)

            points = np.concatenate((points_mm, amp, phase), axis=-1)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith(".npy"):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def _normalize_feature(self, values: np.ndarray) -> np.ndarray:
        min_val = values.min(axis=0, keepdims=True)
        max_val = values.max(axis=0, keepdims=True)
        denom = np.maximum(max_val - min_val, 1e-6)
        return (values - min_val) / denom

    def transform(self, results: dict) -> dict:

        pts_file_paths = results["radar_points_paths"]
        results["points"] = []

        loaded_by_path: dict = {}
        for fid, pts_file_path in enumerate(pts_file_paths):
            if pts_file_path in loaded_by_path:
                points = loaded_by_path[pts_file_path].copy()
            else:
                points = self._load_points(pts_file_path)
                loaded_by_path[pts_file_path] = points.copy()

            points = points.reshape(-1, self.load_dim)
            points = points[:, self.use_dim]
            attribute_dims = None

            if self.use_phase:
                assert len(self.use_dim) >= 5
                if attribute_dims is None:
                    attribute_dims = dict()
                attribute_dims.update(dict(phase=[points.shape[1] - 1]))

            if self.use_amplitude:
                assert len(self.use_dim) >= 5
                if attribute_dims is None:
                    attribute_dims = dict()
                attribute_dims.update(dict(amp=[points.shape[1] - 2]))

            points = RadarPoints(
                points, points_dim=points.shape[-1], attribute_dims=attribute_dims
            )

            results["points"].append(points)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + "("
        repr_str += f"shift_height={self.shift_height}, "
        repr_str += f"use_color={self.use_color}, "
        repr_str += f"backend_args={self.backend_args}, "
        repr_str += f"load_dim={self.load_dim}, "
        repr_str += f"use_dim={self.use_dim})"
        repr_str += f"norm_intensity={self.norm_intensity})"
        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotations3D(MMCV_LoadAnnotations):
    def __init__(
        self, with_keypoints_3d: bool = True, eval: bool = False, **kwargs
    ) -> None:
        super(LoadAnnotations3D, self).__init__(**kwargs)
        self.with_keypoints_3d = with_keypoints_3d
        self.eval = eval

    def _load_keypoints_3d(self, results):
        if not self.eval:
            results["gt_keypoints_3d"] = results["ann_info"]["gt_keypoints_3d"]
            ann_frames = results.get("ann_info_frames", None)
        if self.eval:
            results["gt_keypoints_3d"] = results["eval_ann_info"]["gt_keypoints_3d"]
            ann_frames = results.get("eval_ann_info_frames", None)
        if ann_frames is not None:
            frame_keypoints = []
            for ann_info in ann_frames:
                if ann_info is None:
                    frame_keypoints.append(results["gt_keypoints_3d"][:0])
                else:
                    frame_keypoints.append(ann_info["gt_keypoints_3d"])
            results["gt_keypoints_3d_frames"] = frame_keypoints
        return results

    def transform(self, results: dict) -> dict:
        if self.with_keypoints_3d:
            results = self._load_keypoints_3d(results)
        return results

    def __repr__(self) -> str:
        indent_str = "    "
        repr_str = self.__class__.__name__ + "(\n"
        repr_str += f"{indent_str}with_keypoints_3d={self.with_keypoints_3d}, "

        return repr_str
