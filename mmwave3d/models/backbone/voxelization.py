import math
import time
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor
from mmengine.model.utils import stack_batch
from mmengine.utils import is_seq_of

from mmwave3d.registry import MODELS


@MODELS.register_module()
class MmWaveDataPreprocessor(BaseDataPreprocessor):
    """mmWave data preprocessor with spherical voxelization and BEV encoding."""

    def __init__(self,
                 point_cloud_range: Sequence[float],
                 voxel_size_r: float = 0.12,
                 voxel_angle_size_deg: float = 1.0,
                 frame_id_index: Optional[int] = None,
                 expected_frames: Optional[int] = None,
                 angles_in_degrees: bool = True,
                 voxel_pool_kernel: Optional[Union[int, Sequence[int]]] = None,
                 voxel_pool_stride: Optional[Union[int, Sequence[int]]] = None,
                 voxel_pool_type: str = 'max',
                 voxel_downsample_stages: int = 0,
                 voxel_conv_kernel_size: Union[int, Sequence[int]] = 3,
                 mean: Optional[Sequence[float]] = None,
                 std: Optional[Sequence[float]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: float = 0.0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: bool = False,
                 data_norm: bool = False) -> None:
        super().__init__(non_blocking=non_blocking)
        assert len(point_cloud_range) == 6, 'point_cloud_range must have 6 values'
        assert not (bgr_to_rgb and rgb_to_bgr), (
            'bgr_to_rgb and rgb_to_bgr cannot be both True')
        assert (mean is None) == (std is None), (
            'mean and std should be both None or both provided')
        self.data_norm = data_norm

        if mean is not None:
            assert len(mean) in (1, 3), 'mean should have 1 or 3 values'
            assert len(std) in (1, 3), 'std should have 1 or 3 values'
            self._enable_normalize = True
            self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std', torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False
        self._channel_conversion = bgr_to_rgb or rgb_to_bgr
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

        self.frame_id_index = frame_id_index
        self.expected_frames = expected_frames
        self.angles_in_degrees = angles_in_degrees
        # Token stats for logging percentiles across batches.
        # self._token_stat_batches = 0
        # self._token_stat_samples: List[int] = []
        self.voxel_downsample_stages = voxel_downsample_stages
        if voxel_pool_kernel is not None:
            if isinstance(voxel_pool_kernel, int):
                voxel_pool_kernel = (voxel_pool_kernel,) * 3
            assert len(voxel_pool_kernel[0]) == 3, \
                'voxel_pool_kernel must have 3 elements (r, azimuth, elevation).'
            if voxel_pool_stride is None:
                voxel_pool_stride = voxel_pool_kernel
            if isinstance(voxel_pool_stride, int):
                voxel_pool_stride = (voxel_pool_stride,) * 3
            assert len(voxel_pool_stride[0]) == 3, \
                'voxel_pool_stride must have 3 elements (r, azimuth, elevation).'
            voxel_pool_type = voxel_pool_type.lower()
            assert voxel_pool_type in ('max', 'avg'), \
                'voxel_pool_type must be either "max" or "avg"'
        if self.voxel_downsample_stages > 0:
            assert voxel_pool_kernel is not None, (
                'voxel_pool_kernel must be set when voxel_downsample_stages > 0.')
        self.voxel_pool_kernel = voxel_pool_kernel
        self.voxel_pool_stride = voxel_pool_stride
        self.voxel_pool_type = voxel_pool_type
        if isinstance(voxel_conv_kernel_size, int):
            voxel_conv_kernel_size = (voxel_conv_kernel_size,) * 3
        assert len(voxel_conv_kernel_size) == 3, \
            'voxel_conv_kernel_size must have 3 elements (r, azimuth, elevation).'
        self.voxel_conv_kernel_size = voxel_conv_kernel_size
        conv_kernel = torch.ones(
            (1, 1, *voxel_conv_kernel_size), dtype=torch.float32)
        conv_kernel /= conv_kernel.numel()
        self.register_buffer(
            'voxel_conv_kernel', conv_kernel, persistent=False)
        self.voxel_conv_padding = tuple(k // 2 for k in voxel_conv_kernel_size)

        angle_unit = math.radians if angles_in_degrees else (lambda x: x)
        orig_min = torch.tensor([
            point_cloud_range[0],
            angle_unit(point_cloud_range[1]),
            angle_unit(point_cloud_range[2])
        ],
                                dtype=torch.float32)
        orig_max = torch.tensor([
            point_cloud_range[3],
            angle_unit(point_cloud_range[4]),
            angle_unit(point_cloud_range[5])
        ],
                                dtype=torch.float32)
        orig_span = orig_max - orig_min
        self.register_buffer('orig_range_min', orig_min, persistent=False)
        self.register_buffer('orig_range_max', orig_max, persistent=False)
        self.register_buffer('orig_range_span', orig_span, persistent=False)
        if data_norm:
            range_min = torch.zeros_like(orig_min)
            range_max = torch.ones_like(orig_max)
        else:
            range_min = orig_min
            range_max = orig_max
        self.register_buffer('range_min', range_min, persistent=False)
        self.register_buffer('range_max', range_max, persistent=False)
        voxel_size = torch.tensor([
            voxel_size_r,
            math.radians(voxel_angle_size_deg)
            if angles_in_degrees else voxel_angle_size_deg,
            math.radians(voxel_angle_size_deg)
            if angles_in_degrees else voxel_angle_size_deg
        ],
                                  dtype=torch.float32)
        if data_norm:
            voxel_size = voxel_size / orig_span
        self.register_buffer('voxel_size', voxel_size, persistent=False)
        self.grid_size = torch.ceil(
            (self.range_max - self.range_min) / self.voxel_size).long()
        self.register_buffer(
            'voxel_centers', self._compute_voxel_centers(), persistent=False)


    def forward(self, data: dict, training: bool = False) -> dict:
        inputs = data['inputs']
        pad_shapes = None
        if 'img' in inputs:
            pad_shapes = self._get_pad_shapes(inputs['img'])

        data = self.cast_data(data)
        inputs = data['inputs']
        batch_inputs = dict()

        if 'img' in inputs:
            imgs = self._process_imgs(inputs['img'])
            batch_inputs['imgs'] = imgs
            data_samples = data.get('data_samples')
            if data_samples is not None and pad_shapes is not None:
                batch_input_shape = tuple(imgs.shape[-2:])
                for sample, pad_shape in zip(data_samples, pad_shapes):
                    sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

        if 'points' in inputs:
            points_input = inputs['points']
            # When using `pseudo_collate`, a list of per-frame tensors is
            # transposed to frame-first shape: [num_frames][batch]. Restore
            # to batch-first ([batch][num_frames]) so voxelization reads the
            # batch dimension correctly.
            if (self.expected_frames is not None and isinstance(points_input, list)
                    and len(points_input) == self.expected_frames
                    and points_input
                    and isinstance(points_input[0], (list, tuple))
                    and isinstance(points_input[0][0], torch.Tensor)):
                points_input = [list(frames) for frames in zip(*points_input)]
                inputs['points'] = points_input

            batch_inputs['points'] = inputs['points']
            if isinstance(inputs['points'], (list, tuple)):
                # torch.cuda.synchronize()
                # t_start = time.time()
                voxel_grids = self._build_voxel_batch(inputs['points'])
                # torch.cuda.synchronize()
                # voxel_time = time.time() - t_start

                pool_time = 0.0
                if self.voxel_downsample_stages > 0:
                    # torch.cuda.synchronize()
                    # t_pool = time.time()
                    voxel_grids = self._pool_voxel_grids(voxel_grids)
                    # torch.cuda.synchronize()
                    # pool_time = time.time() - t_pool
                # torch.cuda.synchronize()
                # t_tokens = time.time()
                batch_inputs['mmwave_voxels'] = voxel_grids
                spatial = self._get_spatial_grid_for_shape(
                    voxel_grids.shape[-3:],
                    voxel_grids.device,
                    voxel_grids.dtype)
                spatial = spatial.view(1, 1, *spatial.shape)  # (1,1,r,a,e,3)
                spatial = spatial.expand(
                    voxel_grids.shape[0], voxel_grids.shape[1], -1, -1, -1, -1)
                batch_inputs['mmwave_pos_grid'] = spatial
                # torch.cuda.synchronize()
                # token_time = time.time() - t_tokens

                # logger = MMLogger.get_current_instance()
                # if logger is not None:
                #     logger.info(
                #         'Voxel timings - build: %.4fs  pool: %.4fs  tokens: %.4fs '
                #         'shape: %s',
                #         voxel_time, pool_time, token_time,
                #         tuple(voxel_grids.shape))

        data['inputs'] = batch_inputs
        data.setdefault('data_samples', None)
        return data

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------
    def _process_imgs(self, batch_imgs: Sequence[torch.Tensor]) -> torch.Tensor:
        if is_seq_of(batch_imgs, torch.Tensor):
            img_dim = batch_imgs[0].dim()
            processed = []
            for img in batch_imgs:
                if img_dim == 3:
                    processed.append(self._preprocess_img(img))
                elif img_dim == 4:
                    views = [self._preprocess_img(v) for v in img]
                    processed.append(torch.stack(views, dim=0))
                else:
                    raise ValueError(f'Unsupported image dim {img.dim()}')
            if img_dim == 3:
                return stack_batch(processed, self.pad_size_divisor, self.pad_value)
            else:
                return self._stack_multiview(processed)
        elif isinstance(batch_imgs, torch.Tensor):
            assert batch_imgs.dim() == 4, (
                'Expected NCHW image tensor, got '
                f'{batch_imgs.shape}')
            return self._pad_batch_tensor(batch_imgs)
        else:
            raise TypeError(f'Unsupported image input type {type(batch_imgs)}')

    def _preprocess_img(self, img: torch.Tensor) -> torch.Tensor:
        if self._channel_conversion:
            img = img[[2, 1, 0], ...]
        img = img.float()
        if self._enable_normalize:
            img = (img - self.mean) / self.std
        return img

    def _stack_multiview(self, batch_imgs: Sequence[torch.Tensor]) -> torch.Tensor:
        num_views = batch_imgs[0].shape[0]
        assert all(img.shape[0] == num_views for img in batch_imgs), (
            'All samples must have the same number of views')
        max_h = max(img.shape[-2] for img in batch_imgs)
        max_w = max(img.shape[-1] for img in batch_imgs)
        padded = []
        for img in batch_imgs:
            pad_h = max_h - img.shape[-2]
            pad_w = max_w - img.shape[-1]
            if pad_h > 0 or pad_w > 0:
                img = F.pad(img, (0, pad_w, 0, pad_h), 'constant', self.pad_value)
            padded.append(img)
        return torch.stack(padded, dim=0)

    def _pad_batch_tensor(self, imgs: torch.Tensor) -> torch.Tensor:
        if self._channel_conversion:
            imgs = imgs[:, [2, 1, 0], ...]
        imgs = imgs.float()
        if self._enable_normalize:
            imgs = (imgs - self.mean) / self.std
        h, w = imgs.shape[2:]
        target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
        target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
        pad_h = target_h - h
        pad_w = target_w - w
        if pad_h > 0 or pad_w > 0:
            imgs = F.pad(imgs, (0, pad_w, 0, pad_h), 'constant', self.pad_value)
        return imgs

    def _get_pad_shapes(self, batch_imgs: Sequence[torch.Tensor]) -> List[Tuple[int, int]]:
        pad_shapes: List[Tuple[int, int]] = []
        if is_seq_of(batch_imgs, torch.Tensor):
            first_dim = batch_imgs[0].dim()
            for img in batch_imgs:
                img_ref = img[0] if first_dim == 4 else img
                pad_h = math.ceil(img_ref.shape[-2] / self.pad_size_divisor) * self.pad_size_divisor
                pad_w = math.ceil(img_ref.shape[-1] / self.pad_size_divisor) * self.pad_size_divisor
                pad_shapes.append((pad_h, pad_w))
        elif isinstance(batch_imgs, torch.Tensor):
            assert batch_imgs.dim() == 4, 'Expected NCHW image tensor.'
            pad_h = math.ceil(batch_imgs.shape[2] / self.pad_size_divisor) * self.pad_size_divisor
            pad_w = math.ceil(batch_imgs.shape[3] / self.pad_size_divisor) * self.pad_size_divisor
            pad_shapes = [(pad_h, pad_w)] * batch_imgs.shape[0]
        else:
            raise TypeError(f'Unsupported image input type {type(batch_imgs)}')
        return pad_shapes

    # ------------------------------------------------------------------
    # Voxel encoding
    # ------------------------------------------------------------------
    def _build_voxel_batch(
            self, points_batch: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor:
        encoded_samples = []
        for sample_points in points_batch:
            frames = self._extract_frames(sample_points)
            encoded_frames = [self._voxelize_frame(frame_points) for frame_points in frames]
            encoded_samples.append(torch.stack(encoded_frames, dim=0))
        return torch.stack(encoded_samples, dim=0)

    def _extract_frames(
        self, points: Union[torch.Tensor, Sequence[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Convert raw point inputs to a list of per-frame coordinates."""
        if isinstance(points, (list, tuple)):
            frames: List[torch.Tensor] = []
            for frame in points:
                if frame.numel() == 0:
                    frames.append(frame.new_zeros((0, 3)))
                    continue
                coords = frame[:, :3].clone()
                if self.angles_in_degrees:
                    coords[:, 1:] = torch.deg2rad(coords[:, 1:])
                if self.data_norm:
                    coords = self._normalize_coords(coords)
                frames.append(coords)
            if not frames:
                device = points[0].device if len(points) > 0 else self.range_min.device
                frames = [torch.zeros((0, 3), device=device)]
            return frames
        elif isinstance(points, torch.Tensor):
            return self._split_by_frame(points)
        else:
            raise TypeError(f'Unsupported point container: {type(points)}')

    def _split_by_frame(self, points: torch.Tensor) -> List[torch.Tensor]:
        if points.numel() == 0:
            empty = torch.zeros((0, 3), device=points.device, dtype=points.dtype)
            return [empty]
        coords = points[:, :3].clone()
        if self.angles_in_degrees:
            coords[:, 1:] = torch.deg2rad(coords[:, 1:])
        if self.data_norm:
            coords = self._normalize_coords(coords)
        if self.frame_id_index is not None:
            idx = self.frame_id_index if self.frame_id_index >= 0 else points.shape[1] + self.frame_id_index
            frame_ids = points[:, idx].round().long()
        else:
            # No frame-id column provided; treat the entire tensor as a single frame.
            return [coords]
        frames: List[torch.Tensor] = []
        if self.expected_frames is not None:
            for fid in range(self.expected_frames):
                frames.append(coords[frame_ids == fid])
        else:
            unique_ids = torch.unique(frame_ids, sorted=True)
            for fid in unique_ids:
                frames.append(coords[frame_ids == fid])
        if not frames:
            frames.append(torch.zeros((0, 3), device=points.device, dtype=points.dtype))
        return frames

    def _voxelize_frame(self, coords: torch.Tensor) -> torch.Tensor:
        grid_shape = self.grid_size.tolist()
        voxel_grid = coords.new_zeros(grid_shape)
        if coords.numel() == 0:
            return voxel_grid
        valid_mask = ((coords >= self.range_min) & (coords <= self.range_max)).all(dim=1)
        coords = coords[valid_mask]
        if coords.numel() == 0:
            return voxel_grid
        idx = torch.floor((coords - self.range_min) / self.voxel_size).long()
        idx = torch.stack([
            idx[:, 0].clamp(0, grid_shape[0] - 1),
            idx[:, 1].clamp(0, grid_shape[1] - 1),
            idx[:, 2].clamp(0, grid_shape[2] - 1)
        ], dim=1)
        r_idx, a_idx, e_idx = idx.unbind(dim=1)
        values = torch.ones_like(r_idx, dtype=voxel_grid.dtype)
        voxel_grid.index_put_((r_idx, a_idx, e_idx), values, accumulate=True)
        return voxel_grid

    def _pool_voxel_grids(self, voxel_grids: torch.Tensor) -> torch.Tensor:
        """Downsample voxel grids via conv+pool cascades."""
        if self.voxel_downsample_stages <= 0 or self.voxel_pool_kernel is None:
            return voxel_grids
        b, f, r, a, e = voxel_grids.shape
        grids = voxel_grids.view(b * f, 1, r, a, e)
        kernel = self.voxel_pool_kernel
        stride = self.voxel_pool_stride
        conv_kernel = self.voxel_conv_kernel.to(
            device=grids.device, dtype=grids.dtype)
        padding = self.voxel_conv_padding
        for i in range(self.voxel_downsample_stages):
            grids = F.conv3d(
                grids,
                conv_kernel,
                padding=padding)
            grids = F.relu(grids)
            if self.voxel_pool_type == 'max':
                grids = F.max_pool3d(grids, kernel_size=kernel[i], stride=stride[i])
            else:
                grids = F.avg_pool3d(grids, kernel_size=kernel[i], stride=stride[i])
        grids = grids.view(b, f, grids.shape[-3], grids.shape[-2], grids.shape[-1])
        return grids

    def build_transformer_tokens(
            self, voxel_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flatten voxel grids into tokens plus 4D (x,y,z,t) positions."""
        b, f, r, a, e = voxel_batch.shape
        tokens = voxel_batch.view(b, f * r * a * e, 1)
        spatial = self._get_spatial_grid_for_shape(
            (r, a, e), voxel_batch.device, voxel_batch.dtype)
        spatial = spatial.view(1, 1, r, a, e, 3)
        spatial = spatial.expand(b, f, -1, -1, -1, -1)
        frame_coord = torch.arange(
            f, device=voxel_batch.device, dtype=voxel_batch.dtype)
        frame_coord = frame_coord.view(1, f, 1, 1, 1, 1).expand(b, f, r, a, e, 1)
        pos = torch.cat([spatial, frame_coord], dim=-1).reshape(b, f * r * a * e, 4)
        return tokens, pos

    def _compute_voxel_centers(self) -> torch.Tensor:
        """Pre-compute voxel center coordinates for positional encoding."""
        axes = []
        for axis in range(3):
            num = int(self.grid_size[axis].item())
            centers = torch.arange(num, dtype=torch.float32) + 0.5
            centers = centers * self.voxel_size[axis] + self.range_min[axis]
            axes.append(centers)
        rr, aa, ee = torch.meshgrid(axes[0], axes[1], axes[2], indexing='ij')
        return torch.stack([rr, aa, ee], dim=-1)

    def _get_spatial_grid_for_shape(self, shape: Tuple[int, int, int],
                                    device: torch.device,
                                    dtype: torch.dtype) -> torch.Tensor:
        """Return voxel centers that match the (r, a, e) shape of the grid."""
        if tuple(self.voxel_centers.shape[:3]) == shape:
            return self.voxel_centers.to(device=device, dtype=dtype)
        if self.voxel_pool_kernel is None or self.voxel_pool_stride is None:
            raise RuntimeError(
                'Voxel center grid does not match voxel_batch shape and no '
                'pooling parameters are available to reconcile them.')
        centers = self.voxel_centers.permute(3, 0, 1, 2).unsqueeze(0)
        centers = centers.to(device=device, dtype=dtype)
        pooled = centers
        stages = self.voxel_downsample_stages if self.voxel_downsample_stages > 0 else 1
        for i in range(stages):
            pooled = F.avg_pool3d(
                pooled,
                kernel_size=self.voxel_pool_kernel[i],
                stride=self.voxel_pool_stride[i])
        pooled = pooled.squeeze(0).permute(1, 2, 3, 0)
        if tuple(pooled.shape[:3]) != shape:
            raise RuntimeError(
                'Pooled voxel center grid shape does not match voxel_batch, '
                f'expected {shape} but got {tuple(pooled.shape[:3])}.')
        return pooled

    def _normalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalize spherical coords to [0,1] using the original range."""
        span = self.orig_range_span.to(device=coords.device, dtype=coords.dtype)
        minv = self.orig_range_min.to(device=coords.device, dtype=coords.dtype)
        norm = (coords - minv) / span.clamp(min=1e-6)
        return norm


@MODELS.register_module()
class SparseVoxelization(BaseDataPreprocessor):
    """Sparse voxelization that outputs per-frame sparse tokens + positions."""

    def __init__(self,
                 point_cloud_range: Sequence[float],
                 voxel_size_r: float = 0.12,
                 voxel_angle_size_deg: float = 1.0,
                 frame_id_index: Optional[int] = None,
                 expected_frames: Optional[int] = None,
                 angles_in_degrees: bool = True,
                 include_point_features: bool = True,
                 count_log_scale: bool = True,
                 max_tokens_per_sample: Optional[int] = None,
                 mean: Optional[Sequence[float]] = None,
                 std: Optional[Sequence[float]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: float = 0.0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: bool = False,
                 data_norm: bool = False) -> None:
        super().__init__(non_blocking=non_blocking)
        assert len(point_cloud_range) == 6, 'point_cloud_range must have 6 values'
        assert not (bgr_to_rgb and rgb_to_bgr), (
            'bgr_to_rgb and rgb_to_bgr cannot be both True')
        assert (mean is None) == (std is None), (
            'mean and std should be both None or both provided')
        self.data_norm = data_norm
        self.include_point_features = include_point_features
        self.count_log_scale = count_log_scale
        self.max_tokens_per_sample = max_tokens_per_sample

        if mean is not None:
            assert len(mean) in (1, 3), 'mean should have 1 or 3 values'
            assert len(std) in (1, 3), 'std should have 1 or 3 values'
            self._enable_normalize = True
            self.register_buffer('mean', torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std', torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False
        self._channel_conversion = bgr_to_rgb or rgb_to_bgr
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

        self.frame_id_index = frame_id_index
        self.expected_frames = expected_frames
        self.angles_in_degrees = angles_in_degrees

        angle_unit = math.radians if angles_in_degrees else (lambda x: x)
        orig_min = torch.tensor([
            point_cloud_range[0],
            angle_unit(point_cloud_range[1]),
            angle_unit(point_cloud_range[2])
        ],
                                dtype=torch.float32)
        orig_max = torch.tensor([
            point_cloud_range[3],
            angle_unit(point_cloud_range[4]),
            angle_unit(point_cloud_range[5])
        ],
                                dtype=torch.float32)
        orig_span = orig_max - orig_min
        self.register_buffer('orig_range_min', orig_min, persistent=False)
        self.register_buffer('orig_range_max', orig_max, persistent=False)
        self.register_buffer('orig_range_span', orig_span, persistent=False)
        if data_norm:
            range_min = torch.zeros_like(orig_min)
            range_max = torch.ones_like(orig_max)
        else:
            range_min = orig_min
            range_max = orig_max
        self.register_buffer('range_min', range_min, persistent=False)
        self.register_buffer('range_max', range_max, persistent=False)
        voxel_size = torch.tensor([
            voxel_size_r,
            math.radians(voxel_angle_size_deg)
            if angles_in_degrees else voxel_angle_size_deg,
            math.radians(voxel_angle_size_deg)
            if angles_in_degrees else voxel_angle_size_deg
        ],
                                  dtype=torch.float32)
        if data_norm:
            voxel_size = voxel_size / orig_span
        self.register_buffer('voxel_size', voxel_size, persistent=False)
        self.grid_size = torch.ceil(
            (self.range_max - self.range_min) / self.voxel_size).long()

    def forward(self, data: dict, training: bool = False) -> dict:
        inputs = data['inputs']
        pad_shapes = None
        if 'img' in inputs:
            pad_shapes = self._get_pad_shapes(inputs['img'])

        data = self.cast_data(data)
        inputs = data['inputs']
        batch_inputs = dict()

        if 'img' in inputs:
            imgs = self._process_imgs(inputs['img'])
            batch_inputs['imgs'] = imgs
            data_samples = data.get('data_samples')
            if data_samples is not None and pad_shapes is not None:
                batch_input_shape = tuple(imgs.shape[-2:])
                for sample, pad_shape in zip(data_samples, pad_shapes):
                    sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

        if 'points' in inputs:
            points_input = inputs['points']
            if (self.expected_frames is not None and isinstance(points_input, list)
                    and len(points_input) == self.expected_frames
                    and points_input
                    and isinstance(points_input[0], (list, tuple))
                    and isinstance(points_input[0][0], torch.Tensor)):
                points_input = [list(frames) for frames in zip(*points_input)]
                inputs['points'] = points_input

            batch_inputs['points'] = inputs['points']
            if isinstance(inputs['points'], (list, tuple)):
                tokens, pos, mask, idx, lin, shape = self._build_sparse_batch(inputs['points'])
                batch_inputs['mmwave_sparse_tokens'] = tokens
                batch_inputs['mmwave_sparse_pos'] = pos
                batch_inputs['mmwave_sparse_mask'] = mask
                batch_inputs['mmwave_sparse_indices'] = idx
                batch_inputs['mmwave_sparse_lin'] = lin
                batch_inputs['mmwave_sparse_shape'] = shape

        data['inputs'] = batch_inputs
        data.setdefault('data_samples', None)
        return data

    # ------------------------------------------------------------------
    # Image helpers (same as dense preprocessor)
    # ------------------------------------------------------------------
    def _process_imgs(self, batch_imgs: Sequence[torch.Tensor]) -> torch.Tensor:
        if is_seq_of(batch_imgs, torch.Tensor):
            img_dim = batch_imgs[0].dim()
            processed = []
            for img in batch_imgs:
                if img_dim == 3:
                    processed.append(self._preprocess_img(img))
                elif img_dim == 4:
                    views = [self._preprocess_img(v) for v in img]
                    processed.append(torch.stack(views, dim=0))
                else:
                    raise ValueError(f'Unsupported image dim {img.dim()}')
            if img_dim == 3:
                return stack_batch(processed, self.pad_size_divisor, self.pad_value)
            else:
                return self._stack_multiview(processed)
        elif isinstance(batch_imgs, torch.Tensor):
            assert batch_imgs.dim() == 4, (
                'Expected NCHW image tensor, got '
                f'{batch_imgs.shape}')
            return self._pad_batch_tensor(batch_imgs)
        else:
            raise TypeError(f'Unsupported image input type {type(batch_imgs)}')

    def _preprocess_img(self, img: torch.Tensor) -> torch.Tensor:
        if self._channel_conversion:
            img = img[[2, 1, 0], ...]
        img = img.float()
        if self._enable_normalize:
            img = (img - self.mean) / self.std
        return img

    def _stack_multiview(self, batch_imgs: Sequence[torch.Tensor]) -> torch.Tensor:
        num_views = batch_imgs[0].shape[0]
        assert all(img.shape[0] == num_views for img in batch_imgs), (
            'All samples must have the same number of views')
        max_h = max(img.shape[-2] for img in batch_imgs)
        max_w = max(img.shape[-1] for img in batch_imgs)
        padded = []
        for img in batch_imgs:
            pad_h = max_h - img.shape[-2]
            pad_w = max_w - img.shape[-1]
            if pad_h > 0 or pad_w > 0:
                img = F.pad(img, (0, pad_w, 0, pad_h), 'constant', self.pad_value)
            padded.append(img)
        return torch.stack(padded, dim=0)

    def _pad_batch_tensor(self, imgs: torch.Tensor) -> torch.Tensor:
        if self._channel_conversion:
            imgs = imgs[:, [2, 1, 0], ...]
        imgs = imgs.float()
        if self._enable_normalize:
            imgs = (imgs - self.mean) / self.std
        h, w = imgs.shape[2:]
        target_h = math.ceil(h / self.pad_size_divisor) * self.pad_size_divisor
        target_w = math.ceil(w / self.pad_size_divisor) * self.pad_size_divisor
        pad_h = target_h - h
        pad_w = target_w - w
        if pad_h > 0 or pad_w > 0:
            imgs = F.pad(imgs, (0, pad_w, 0, pad_h), 'constant', self.pad_value)
        return imgs

    def _get_pad_shapes(self, batch_imgs: Sequence[torch.Tensor]) -> List[Tuple[int, int]]:
        pad_shapes: List[Tuple[int, int]] = []
        if is_seq_of(batch_imgs, torch.Tensor):
            first_dim = batch_imgs[0].dim()
            for img in batch_imgs:
                img_ref = img[0] if first_dim == 4 else img
                pad_h = math.ceil(img_ref.shape[-2] / self.pad_size_divisor) * self.pad_size_divisor
                pad_w = math.ceil(img_ref.shape[-1] / self.pad_size_divisor) * self.pad_size_divisor
                pad_shapes.append((pad_h, pad_w))
        elif isinstance(batch_imgs, torch.Tensor):
            assert batch_imgs.dim() == 4, 'Expected NCHW image tensor.'
            pad_h = math.ceil(batch_imgs.shape[2] / self.pad_size_divisor) * self.pad_size_divisor
            pad_w = math.ceil(batch_imgs.shape[3] / self.pad_size_divisor) * self.pad_size_divisor
            pad_shapes = [(pad_h, pad_w)] * batch_imgs.shape[0]
        else:
            raise TypeError(f'Unsupported image input type {type(batch_imgs)}')
        return pad_shapes

    # ------------------------------------------------------------------
    # Sparse voxel encoding
    # ------------------------------------------------------------------
    def _build_sparse_batch(
            self, points_batch: Sequence[Sequence[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens_per_sample = []
        pos_per_sample = []
        mask_per_sample = []
        idx_per_sample = []
        lin_per_sample = []
        batch_tokens = []
        batch_pos = []
        batch_mask = []
        batch_idx = []
        batch_lin = []
        max_tokens = 0
        for sample_points in points_batch:
            frames = self._extract_frames(sample_points)
            frame_voxels = [self._voxelize_frame_sparse(frame_points) for frame_points in frames]
            num_tokens_per_frame = [lin_idx.numel() for lin_idx, _, _ in frame_voxels]
            if self.max_tokens_per_sample is not None:
                capped_tokens_per_frame = [
                    min(n, self.max_tokens_per_sample) for n in num_tokens_per_frame
                ]
            else:
                capped_tokens_per_frame = num_tokens_per_frame
            max_tokens_sample = max(capped_tokens_per_frame) if capped_tokens_per_frame else 0
            all_empty = max_tokens_sample == 0
            if max_tokens_sample == 0:
                max_tokens_sample = 1
            tokens_frames = []
            pos_frames = []
            mask_frames = []
            idx_frames = []
            lin_frames = []
            for lin_idx, feats, centers in frame_voxels:
                if lin_idx.numel() == 0:
                    if all_empty:
                        tokens_f = centers.new_zeros((1, feats.shape[1]))
                        pos_f = centers.new_zeros((1, 3))
                        mask_f = torch.ones((1,), device=centers.device, dtype=torch.bool)
                        idx_f = centers.new_zeros((1, 3), dtype=torch.long)
                        lin_f = centers.new_zeros((1,), dtype=torch.long)
                    else:
                        tokens_f = centers.new_zeros((0, feats.shape[1]))
                        pos_f = centers.new_zeros((0, 3))
                        mask_f = torch.zeros((0,), device=centers.device, dtype=torch.bool)
                        idx_f = centers.new_zeros((0, 3), dtype=torch.long)
                        lin_f = centers.new_zeros((0,), dtype=torch.long)
                else:
                    lin_f = lin_idx
                    if (self.max_tokens_per_sample is not None
                            and lin_f.numel() > self.max_tokens_per_sample):
                        perm = torch.randperm(lin_f.numel(), device=lin_f.device)
                        lin_f = lin_f[perm[:self.max_tokens_per_sample]].sort().values
                        keep = torch.searchsorted(lin_idx, lin_f)
                        feats = feats[keep]
                        centers = centers[keep]
                    r_idx, a_idx, e_idx = self._unravel_lin(lin_f)
                    idx_f = torch.stack([r_idx, a_idx, e_idx], dim=1)
                    tokens_f = feats
                    pos_f = centers
                    mask_f = torch.ones((lin_f.numel(),), device=centers.device, dtype=torch.bool)

                pad_tokens = max_tokens_sample - tokens_f.shape[0]
                if pad_tokens > 0:
                    pad_value = int(self.grid_size[0].item()
                                    * self.grid_size[1].item()
                                    * self.grid_size[2].item())
                    tokens_f = F.pad(tokens_f, (0, 0, 0, pad_tokens))
                    pos_f = F.pad(pos_f, (0, 0, 0, pad_tokens))
                    mask_f = F.pad(mask_f, (0, pad_tokens))
                    idx_f = F.pad(idx_f, (0, 0, 0, pad_tokens))
                    lin_f = F.pad(lin_f, (0, pad_tokens), value=pad_value)

                tokens_frames.append(tokens_f)
                pos_frames.append(pos_f)
                mask_frames.append(mask_f)
                idx_frames.append(idx_f)
                lin_frames.append(lin_f)

            tokens = torch.stack(tokens_frames, dim=0)
            pos = torch.stack(pos_frames, dim=0)
            mask = torch.stack(mask_frames, dim=0)
            idx = torch.stack(idx_frames, dim=0)
            lin = torch.stack(lin_frames, dim=0)

            tokens_per_sample.append(tokens)
            pos_per_sample.append(pos)
            mask_per_sample.append(mask)
            idx_per_sample.append(idx)
            lin_per_sample.append(lin)
            max_tokens = max(max_tokens, max_tokens_sample)

        if max_tokens == 0:
            raise RuntimeError('Sparse voxelization produced no tokens.')

        for tokens, pos, mask, idx, lin in zip(
                tokens_per_sample, pos_per_sample, mask_per_sample, idx_per_sample, lin_per_sample):
            pad_tokens = max_tokens - tokens.shape[1]
            if pad_tokens > 0:
                pad_value = int(self.grid_size[0].item()
                                * self.grid_size[1].item()
                                * self.grid_size[2].item())
                tokens = F.pad(tokens, (0, 0, 0, pad_tokens))
                pos = F.pad(pos, (0, 0, 0, pad_tokens))
                mask = F.pad(mask, (0, pad_tokens))
                idx = F.pad(idx, (0, 0, 0, pad_tokens))
                lin = F.pad(lin, (0, pad_tokens), value=pad_value)
            batch_tokens.append(tokens)
            batch_pos.append(pos)
            batch_mask.append(mask)
            batch_idx.append(idx)
            batch_lin.append(lin)

        return (torch.stack(batch_tokens, dim=0),
                torch.stack(batch_pos, dim=0),
                torch.stack(batch_mask, dim=0),
                torch.stack(batch_idx, dim=0),
                torch.stack(batch_lin, dim=0),
                self.grid_size.clone())

    def _extract_frames(
        self, points: Union[torch.Tensor, Sequence[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """Convert raw point inputs to a list of per-frame tensors."""
        if isinstance(points, (list, tuple)):
            frames: List[torch.Tensor] = []
            for frame in points:
                frames.append(self._process_frame_points(frame))
            if self.expected_frames is not None and len(frames) < self.expected_frames:
                device = points[0].device if len(points) > 0 else self.range_min.device
                dim = points[0].shape[1] if len(points) > 0 else 3
                for _ in range(self.expected_frames - len(frames)):
                    frames.append(torch.zeros((0, dim), device=device))
            if not frames:
                device = points[0].device if len(points) > 0 else self.range_min.device
                frames = [torch.zeros((0, 3), device=device)]
            return frames
        if isinstance(points, torch.Tensor):
            return self._split_by_frame(points)
        raise TypeError(f'Unsupported point container: {type(points)}')

    def _split_by_frame(self, points: torch.Tensor) -> List[torch.Tensor]:
        if points.numel() == 0:
            empty = torch.zeros((0, points.shape[1]), device=points.device, dtype=points.dtype)
            if self.expected_frames is not None:
                return [empty for _ in range(self.expected_frames)]
            return [empty]
        if self.frame_id_index is not None:
            idx = self.frame_id_index if self.frame_id_index >= 0 else points.shape[1] + self.frame_id_index
            frame_ids = points[:, idx].round().long()
        else:
            return [self._process_frame_points(points)]
        frames: List[torch.Tensor] = []
        if self.expected_frames is not None:
            for fid in range(self.expected_frames):
                frames.append(self._process_frame_points(points[frame_ids == fid]))
        else:
            unique_ids = torch.unique(frame_ids, sorted=True)
            for fid in unique_ids:
                frames.append(self._process_frame_points(points[frame_ids == fid]))
        if not frames:
            frames.append(torch.zeros((0, points.shape[1]), device=points.device, dtype=points.dtype))
        return frames

    def _process_frame_points(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.numel() == 0:
            return frame.new_zeros((0, frame.shape[1]))
        coords = frame[:, :3].clone()
        if self.angles_in_degrees:
            coords[:, 1:] = torch.deg2rad(coords[:, 1:])
        if self.data_norm:
            coords = self._normalize_coords(coords)
        frame = frame.clone()
        frame[:, :3] = coords
        return frame

    def _voxelize_frame_sparse(
            self, points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if points.numel() == 0:
            empty = self.range_min.new_zeros((0,), dtype=torch.long)
            feat_dim = max(points.shape[1] - 3, 0) if self.include_point_features else 0
            feats = self.range_min.new_zeros((0, 1 + feat_dim))
            centers = self.range_min.new_zeros((0, 3))
            return empty, feats, centers

        coords = points[:, :3]
        feats = points[:, 3:] if points.shape[1] > 3 and self.include_point_features else None
        valid_mask = ((coords >= self.range_min) & (coords <= self.range_max)).all(dim=1)
        coords = coords[valid_mask]
        if feats is not None:
            feats = feats[valid_mask]
        if coords.numel() == 0:
            empty = self.range_min.new_zeros((0,), dtype=torch.long)
            feat_dim = max(points.shape[1] - 3, 0) if feats is not None else 0
            feats = self.range_min.new_zeros((0, 1 + feat_dim))
            centers = self.range_min.new_zeros((0, 3))
            return empty, feats, centers

        idx = torch.floor((coords - self.range_min) / self.voxel_size).long()
        idx = torch.stack([
            idx[:, 0].clamp(0, self.grid_size[0] - 1),
            idx[:, 1].clamp(0, self.grid_size[1] - 1),
            idx[:, 2].clamp(0, self.grid_size[2] - 1)
        ], dim=1)

        lin = self._linearize_idx(idx)
        uniq_lin, inverse = torch.unique(lin, return_inverse=True, sorted=True)
        counts = torch.bincount(inverse, minlength=uniq_lin.numel()).to(coords.dtype)
        if self.count_log_scale:
            count_feat = torch.log1p(counts)
        else:
            count_feat = counts
        count_feat = count_feat.unsqueeze(1)

        if feats is not None and feats.numel() > 0:
            feat_dim = feats.shape[1]
            sum_feats = feats.new_zeros((uniq_lin.numel(), feat_dim))
            sum_feats.index_add_(0, inverse, feats)
            mean_feats = sum_feats / counts.clamp(min=1.0).unsqueeze(1)
            voxel_feats = torch.cat([count_feat, mean_feats], dim=1)
        else:
            voxel_feats = count_feat

        r_idx, a_idx, e_idx = self._unravel_lin(uniq_lin)
        idx = torch.stack([r_idx, a_idx, e_idx], dim=1).to(coords.dtype)
        centers = (idx + 0.5) * self.voxel_size + self.range_min
        return uniq_lin, voxel_feats, centers

    def _linearize_idx(self, idx: torch.Tensor) -> torch.Tensor:
        a_stride = int(self.grid_size[1].item() * self.grid_size[2].item())
        e_stride = int(self.grid_size[2].item())
        return idx[:, 0] * a_stride + idx[:, 1] * e_stride + idx[:, 2]

    def _unravel_lin(self, lin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_stride = int(self.grid_size[1].item() * self.grid_size[2].item())
        e_stride = int(self.grid_size[2].item())
        r_idx = lin // a_stride
        rem = lin % a_stride
        a_idx = rem // e_stride
        e_idx = rem % e_stride
        return r_idx, a_idx, e_idx

    def _normalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        span = self.orig_range_span.to(device=coords.device, dtype=coords.dtype)
        minv = self.orig_range_min.to(device=coords.device, dtype=coords.dtype)
        norm = (coords - minv) / span.clamp(min=1e-6)
        return norm
