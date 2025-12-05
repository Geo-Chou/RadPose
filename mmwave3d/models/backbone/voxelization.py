import math
import time
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor
from mmengine.model.utils import stack_batch
from mmengine.utils import is_seq_of
from mmengine.logging import MMLogger

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
                 non_blocking: bool = False) -> None:
        super().__init__(non_blocking=non_blocking)
        assert len(point_cloud_range) == 6, 'point_cloud_range must have 6 values'
        assert not (bgr_to_rgb and rgb_to_bgr), (
            'bgr_to_rgb and rgb_to_bgr cannot be both True')
        assert (mean is None) == (std is None), (
            'mean and std should be both None or both provided')

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
        self.voxel_downsample_stages = voxel_downsample_stages
        if voxel_pool_kernel is not None:
            if isinstance(voxel_pool_kernel, int):
                voxel_pool_kernel = (voxel_pool_kernel,) * 3
            assert len(voxel_pool_kernel) == 3, \
                'voxel_pool_kernel must have 3 elements (r, azimuth, elevation).'
            if voxel_pool_stride is None:
                voxel_pool_stride = voxel_pool_kernel
            if isinstance(voxel_pool_stride, int):
                voxel_pool_stride = (voxel_pool_stride,) * 3
            assert len(voxel_pool_stride) == 3, \
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
        self.register_buffer(
            'range_min',
            torch.tensor([
                point_cloud_range[0],
                angle_unit(point_cloud_range[1]),
                angle_unit(point_cloud_range[2])
            ], dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            'range_max',
            torch.tensor([
                point_cloud_range[3],
                angle_unit(point_cloud_range[4]),
                angle_unit(point_cloud_range[5])
            ], dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            'voxel_size',
            torch.tensor([
                voxel_size_r,
                math.radians(voxel_angle_size_deg)
                if angles_in_degrees else voxel_angle_size_deg,
                math.radians(voxel_angle_size_deg)
                if angles_in_degrees else voxel_angle_size_deg
            ], dtype=torch.float32),
            persistent=False)
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
                tokens, pos = self.build_transformer_tokens(voxel_grids)
                batch_inputs['mmwave_tokens'] = tokens
                batch_inputs['mmwave_pos'] = pos
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
        for _ in range(self.voxel_downsample_stages):
            grids = F.conv3d(
                grids,
                conv_kernel,
                padding=padding)
            grids = F.relu(grids)
            if self.voxel_pool_type == 'max':
                grids = F.max_pool3d(grids, kernel_size=kernel, stride=stride)
            else:
                grids = F.avg_pool3d(grids, kernel_size=kernel, stride=stride)
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
        for _ in range(stages):
            pooled = F.avg_pool3d(
                pooled,
                kernel_size=self.voxel_pool_kernel,
                stride=self.voxel_pool_stride)
        pooled = pooled.squeeze(0).permute(1, 2, 3, 0)
        if tuple(pooled.shape[:3]) != shape:
            raise RuntimeError(
                'Pooled voxel center grid shape does not match voxel_batch, '
                f'expected {shape} but got {tuple(pooled.shape[:3])}.')
        return pooled
