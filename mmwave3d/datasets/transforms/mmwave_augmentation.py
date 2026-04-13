"""mmWave-specific augmentation for per-frame jitter and noise."""
import numpy as np
import torch
from mmcv.transforms import BaseTransform

from mmwave3d.registry import TRANSFORMS
from .structures import RadarPoints


@TRANSFORMS.register_module()
class MmWaveJitterAugmentation(BaseTransform):
    """Apply per-frame jitter/noise to mmWave spherical points.

    This transform operates per frame and preserves the input format.
    """

    def __init__(self,
                 range_jitter=(0.12, 0.36),
                 angle_jitter_deg=(1.0, 3.0),
                 phase_noise_deg=(5.0, 15.0),
                 amp_noise_pct=(0.05, 0.15),
                 amp_scale=(0.7, 1.3),
                 angles_in_degrees=False,
                 phase_in_degrees=True,
                 phase_normalized=False,
                 phase_range_degrees=360.0,
                 phase_range_radians=2 * np.pi,
                 clip_to_original=True,
                 apply_prob=1.0):
        self.range_jitter = range_jitter
        self.angle_jitter_deg = angle_jitter_deg
        self.phase_noise_deg = phase_noise_deg
        self.amp_noise_pct = amp_noise_pct
        self.amp_scale = amp_scale
        self.angles_in_degrees = angles_in_degrees
        self.phase_in_degrees = phase_in_degrees
        self.phase_normalized = phase_normalized
        self.phase_range_degrees = phase_range_degrees
        self.phase_range_radians = phase_range_radians
        self.clip_to_original = clip_to_original
        self.apply_prob = apply_prob

    def transform(self, results):
        points_list = results.get('points')

        if not isinstance(points_list, (list, tuple)):
            points_list = [points_list]
            single_frame = True
        else:
            single_frame = False

        if len(points_list) == 0:
            return results

        attribute_dims = None
        points_dim = 3
        if isinstance(points_list[0], RadarPoints):
            attribute_dims = points_list[0].attribute_dims
            points_dim = points_list[0].points_dim

        augmented_frames = []
        for points in points_list:
            if isinstance(points, RadarPoints):
                frame = points.tensor.cpu().numpy()
            else:
                frame = np.asarray(points)

            if frame.size == 0:
                augmented_frames.append(frame)
                continue

            if self.apply_prob < 1.0 and np.random.rand() > self.apply_prob:
                augmented_frames.append(frame)
                continue

            frame_aug = frame.copy()
            min_vals = frame.min(axis=0)
            max_vals = frame.max(axis=0)

            # Range jitter (r)
            if self.range_jitter is not None and frame_aug.shape[1] >= 1:
                r_low, r_high = self.range_jitter
                r_mag = np.random.uniform(r_low, r_high, size=(len(frame_aug), 1))
                r_sign = np.random.choice([-1.0, 1.0], size=(len(frame_aug), 1))
                frame_aug[:, 0:1] += r_sign * r_mag

            # Angle jitter (azimuth + elevation)
            if self.angle_jitter_deg is not None and frame_aug.shape[1] >= 3:
                a_low, a_high = self.angle_jitter_deg
                a_mag = np.random.uniform(a_low, a_high, size=(len(frame_aug), 1))
                a_sign = np.random.choice([-1.0, 1.0], size=(len(frame_aug), 1))
                if self.angles_in_degrees:
                    a_delta = a_sign * a_mag
                else:
                    a_delta = a_sign * np.deg2rad(a_mag)
                frame_aug[:, 1:2] += a_delta
                frame_aug[:, 2:3] += a_delta

            amp_idx, phase_idx = self._feature_indices(points, frame_aug)

            # Phase noise
            if self.phase_noise_deg is not None and phase_idx is not None and phase_idx < frame_aug.shape[1]:
                p_low, p_high = self.phase_noise_deg
                p_mag = np.random.uniform(p_low, p_high, size=(len(frame_aug), 1))
                p_sign = np.random.choice([-1.0, 1.0], size=(len(frame_aug), 1))
                if self.phase_in_degrees:
                    p_delta = p_sign * p_mag
                    if self.phase_normalized:
                        p_delta = p_delta / max(self.phase_range_degrees, 1e-6)
                else:
                    p_delta = p_sign * np.deg2rad(p_mag)
                    if self.phase_normalized:
                        p_delta = p_delta / max(self.phase_range_radians, 1e-6)
                frame_aug[:, phase_idx:phase_idx + 1] += p_delta

            # Amplitude noise + scaling
            if amp_idx is not None and amp_idx < frame_aug.shape[1]:
                amp = frame_aug[:, amp_idx:amp_idx + 1]
                if self.amp_noise_pct is not None:
                    n_low, n_high = self.amp_noise_pct
                    noise_mag = np.random.uniform(n_low, n_high, size=(len(frame_aug), 1))
                    noise_sign = np.random.choice([-1.0, 1.0], size=(len(frame_aug), 1))
                    amp = amp + amp * (noise_sign * noise_mag)
                if self.amp_scale is not None:
                    s_low, s_high = self.amp_scale
                    amp = amp * np.random.uniform(s_low, s_high, size=(len(frame_aug), 1))
                frame_aug[:, amp_idx:amp_idx + 1] = amp

            if self.clip_to_original:
                frame_aug = np.clip(frame_aug, min_vals, max_vals)

            augmented_frames.append(frame_aug)

        if single_frame:
            if isinstance(results['points'], RadarPoints):
                results['points'] = RadarPoints(augmented_frames[0], points_dim=points_dim,
                                                attribute_dims=attribute_dims)
            else:
                results['points'] = augmented_frames[0]
        else:
            if isinstance(points_list[0], RadarPoints):
                results['points'] = [RadarPoints(frame, points_dim=points_dim,
                                                 attribute_dims=attribute_dims)
                                    for frame in augmented_frames]
            else:
                results['points'] = augmented_frames
        return results

    def _feature_indices(self, sample_points, np_points):
        amp_idx, phase_idx = None, None
        if isinstance(sample_points, RadarPoints) and sample_points.attribute_dims:
            dims = sample_points.attribute_dims
            if 'amp' in dims and dims['amp']:
                amp_idx = dims['amp'][0]
            if 'phase' in dims and dims['phase']:
                phase_idx = dims['phase'][0]
        if amp_idx is None and np_points.shape[1] >= 4:
            amp_idx = 3
        if phase_idx is None and np_points.shape[1] >= 5:
            phase_idx = 4
        return amp_idx, phase_idx


@TRANSFORMS.register_module()
class MmWaveRangeShift(BaseTransform):
    """Shift entire point cloud range (r) and sync GT range labels."""

    def __init__(self,
                 range_shift=(0.05, 0.25),
                 apply_prob=1.0,
                 point_cloud_range=None,
                 clip_to_range=True,
                 shared_across_frames=True):
        self.range_shift = range_shift
        self.apply_prob = apply_prob
        self.point_cloud_range = point_cloud_range
        self.clip_to_range = clip_to_range
        self.shared_across_frames = shared_across_frames

    def transform(self, results):
        points_list = results.get('points')

        if not isinstance(points_list, (list, tuple)):
            points_list = [points_list]
            single_frame = True
        else:
            single_frame = False

        if len(points_list) == 0:
            return results

        apply_aug = True
        if self.apply_prob < 1.0:
            apply_aug = (np.random.rand() <= self.apply_prob)
        if not apply_aug:
            return results

        attribute_dims = None
        points_dim = 3
        if isinstance(points_list[0], RadarPoints):
            attribute_dims = points_list[0].attribute_dims
            points_dim = points_list[0].points_dim

        range_delta = self._sample_delta()
        augmented_frames = []
        for points in points_list:
            if isinstance(points, RadarPoints):
                frame = points.tensor.cpu().numpy()
            else:
                frame = np.asarray(points)

            if frame.size == 0 or frame.shape[1] < 1:
                augmented_frames.append(frame)
                continue

            frame_aug = frame.copy()
            delta = range_delta if self.shared_across_frames else self._sample_delta()
            frame_aug[:, 0:1] = frame_aug[:, 0:1] + delta

            if self.clip_to_range:
                if self.point_cloud_range is not None and len(self.point_cloud_range) >= 6:
                    r_min = self.point_cloud_range[0]
                    r_max = self.point_cloud_range[3]
                else:
                    r_min = frame[:, 0].min()
                    r_max = frame[:, 0].max()
                frame_aug[:, 0:1] = np.clip(frame_aug[:, 0:1], r_min, r_max)

            augmented_frames.append(frame_aug)

        results = self._apply_gt(results, range_delta)

        if single_frame:
            if isinstance(results['points'], RadarPoints):
                results['points'] = RadarPoints(augmented_frames[0], points_dim=points_dim,
                                                attribute_dims=attribute_dims)
            else:
                results['points'] = augmented_frames[0]
        else:
            if isinstance(points_list[0], RadarPoints):
                results['points'] = [RadarPoints(frame, points_dim=points_dim,
                                                 attribute_dims=attribute_dims)
                                    for frame in augmented_frames]
            else:
                results['points'] = augmented_frames

        return results

    def _sample_delta(self):
        low, high = self.range_shift
        mag = np.random.uniform(low, high)
        sign = np.random.choice([-1.0, 1.0])
        return sign * mag

    def _apply_gt(self, results, delta):
        gt = results.get('gt_keypoints_3d', None)
        if gt is None:
            return results

        is_torch = False
        if hasattr(gt, 'detach'):
            gt_np = gt.detach().cpu().numpy()
            is_torch = True
        else:
            gt_np = np.asarray(gt)

        if gt_np.size == 0 or gt_np.shape[-1] < 1:
            return results

        gt_aug = gt_np.copy()
        r = gt_aug[..., 0]
        valid = ~np.isnan(r)
        r = r + delta

        if self.clip_to_range:
            if self.point_cloud_range is not None and len(self.point_cloud_range) >= 6:
                r_min = self.point_cloud_range[0]
                r_max = self.point_cloud_range[3]
            else:
                r_min = np.nanmin(gt_np[..., 0])
                r_max = np.nanmax(gt_np[..., 0])
            r = np.clip(r, r_min, r_max)

        gt_aug[..., 0] = r
        if valid is not None:
            gt_aug[..., 0][~valid] = gt_np[..., 0][~valid]

        if is_torch:
            results['gt_keypoints_3d'] = torch.as_tensor(gt_aug, device=gt.device, dtype=gt.dtype)
        else:
            results['gt_keypoints_3d'] = gt_aug

        return results


@TRANSFORMS.register_module()
class MmWaveYawRotation(BaseTransform):
    """Random yaw rotation (azimuth) applied consistently to points and GT."""

    def __init__(self,
                 yaw_deg=(10.0, 20.0),
                 angles_in_degrees=False,
                 apply_prob=1.0,
                 point_cloud_range=None,
                 clip_to_original=True,
                 shared_across_frames=True):
        self.yaw_deg = yaw_deg
        self.angles_in_degrees = angles_in_degrees
        self.apply_prob = apply_prob
        self.point_cloud_range = point_cloud_range
        self.clip_to_original = clip_to_original
        self.shared_across_frames = shared_across_frames

    def transform(self, results):
        points_list = results.get('points')

        if not isinstance(points_list, (list, tuple)):
            points_list = [points_list]
            single_frame = True
        else:
            single_frame = False

        if len(points_list) == 0:
            return results

        apply_aug = True
        if self.apply_prob < 1.0:
            apply_aug = (np.random.rand() <= self.apply_prob)
        if not apply_aug:
            return results

        attribute_dims = None
        points_dim = 3
        if isinstance(points_list[0], RadarPoints):
            attribute_dims = points_list[0].attribute_dims
            points_dim = points_list[0].points_dim

        yaw_delta = self._sample_delta()
        augmented_frames = []
        for points in points_list:
            if isinstance(points, RadarPoints):
                frame = points.tensor.cpu().numpy()
            else:
                frame = np.asarray(points)

            if frame.size == 0 or frame.shape[1] < 2:
                augmented_frames.append(frame)
                continue

            frame_aug = frame.copy()
            min_vals = frame.min(axis=0)
            max_vals = frame.max(axis=0)

            delta = yaw_delta if self.shared_across_frames else self._sample_delta()
            frame_aug[:, 1:2] = frame_aug[:, 1:2] + delta

            if self.clip_to_original:
                frame_aug = np.clip(frame_aug, min_vals, max_vals)

            augmented_frames.append(frame_aug)

        results = self._apply_gt(results, yaw_delta)

        if single_frame:
            if isinstance(results['points'], RadarPoints):
                results['points'] = RadarPoints(augmented_frames[0], points_dim=points_dim,
                                                attribute_dims=attribute_dims)
            else:
                results['points'] = augmented_frames[0]
        else:
            if isinstance(points_list[0], RadarPoints):
                results['points'] = [RadarPoints(frame, points_dim=points_dim,
                                                 attribute_dims=attribute_dims)
                                    for frame in augmented_frames]
            else:
                results['points'] = augmented_frames

        return results

    def _sample_delta(self):
        low, high = self.yaw_deg
        mag = np.random.uniform(low, high)
        sign = np.random.choice([-1.0, 1.0])
        if self.angles_in_degrees:
            return sign * mag
        return sign * np.deg2rad(mag)

    def _apply_gt(self, results, delta):
        gt = results.get('gt_keypoints_3d', None)
        if gt is None:
            return results

        is_torch = False
        if hasattr(gt, 'detach'):
            gt_np = gt.detach().cpu().numpy()
            is_torch = True
        else:
            gt_np = np.asarray(gt)

        if gt_np.size == 0 or gt_np.shape[-1] < 2:
            return results

        gt_aug = gt_np.copy()
        az = gt_aug[..., 1]
        valid = ~np.isnan(az)
        az = az + delta

        if self.point_cloud_range is not None and len(self.point_cloud_range) >= 6:
            az_min = self.point_cloud_range[1]
            az_max = self.point_cloud_range[4]
            az = np.clip(az, az_min, az_max)

        gt_aug[..., 1] = az
        if valid is not None:
            gt_aug[..., 1][~valid] = gt_np[..., 1][~valid]

        if is_torch:
            results['gt_keypoints_3d'] = torch.as_tensor(gt_aug, device=gt.device, dtype=gt.dtype)
        else:
            results['gt_keypoints_3d'] = gt_aug

        return results
