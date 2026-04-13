"""mmWave-specific filtering based on temporal phase variance and amplitude."""
from collections import deque
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmwave3d.registry import TRANSFORMS
from .structures import RadarPoints


def spherical_to_cartesian(points_sp: np.ndarray) -> np.ndarray:
    """Convert spherical coords [r, a, e] to Cartesian."""
    r = points_sp[:, 0]
    a = points_sp[:, 1]
    e = points_sp[:, 2]
    x = r * np.cos(a) * np.cos(e)
    y = r * np.sin(e)
    z = r * np.sin(a) * np.cos(e)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def voxel_phase_stats(points, phase, amplitude, voxel_size):
    """Aggregate mean phasor and amplitude per voxel."""
    vox = np.floor(points / voxel_size).astype(np.int32)
    keys = [tuple(v) for v in vox]
    unique, inverse = np.unique(vox, axis=0, return_inverse=True)
    phasor = np.exp(1j * phase.astype(np.float64))
    mean_phase = np.zeros(len(unique), dtype=np.complex128)
    mean_amp = np.zeros(len(unique), dtype=np.float32)
    counts = np.zeros(len(unique), dtype=np.int32)
    np.add.at(mean_phase, inverse, phasor)
    np.add.at(mean_amp, inverse, amplitude.astype(np.float32))
    np.add.at(counts, inverse, 1)
    counts_safe = np.maximum(counts, 1)
    mean_phase /= counts_safe
    mean_amp /= counts_safe
    stats = {tuple(unique[i]): (mean_phase[i], mean_amp[i], counts[i]) for i in range(len(unique))}
    return stats, keys


@TRANSFORMS.register_module()
class TemporalPhaseFiltering(BaseTransform):
    
    def __init__(self,
                 amp_threshold: float = 1.5,
                 phase_delta: float = 0.4,
                 phase_voxel: float = 0.35,
                 amp_relax: float = 0.5,
                 static_delta: float = 0.4,
                 static_persist: int = 3,
                 motion_decay: float = 0.7,
                 motion_keep: float = 1.5,
                 history_size: int = 3,
                 use_phase: bool = True):
        self.amp_threshold = amp_threshold
        self.phase_delta = phase_delta
        self.phase_voxel = phase_voxel
        self.amp_relax = amp_relax
        self.static_delta = static_delta
        self.static_persist = static_persist
        self.motion_decay = motion_decay
        self.motion_keep = motion_keep
        self.history_size = history_size
        self.use_phase = use_phase
    
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
        
        frames_data = []
        for points in points_list:
            if isinstance(points, RadarPoints):
                coords = points.tensor.numpy()
            else:
                coords = points
            frames_data.append(coords)
        
        amp_idx, phase_idx = self._feature_indices(points_list[0], frames_data[0])
        filtered_frames = self._filter_sequence(frames_data, amp_idx, phase_idx)
        
        if isinstance(points_list[0], RadarPoints):
            wrapped = [RadarPoints(frame, points_dim=points_dim,
                                   attribute_dims=attribute_dims)
                       for frame in filtered_frames]
        else:
            wrapped = filtered_frames

        points_count = sum(len(frame) for frame in filtered_frames)
        results['points_count_after_temporal_filter'] = int(points_count)
        
        # # Print filtering statistics
        # total_before = sum(len(frame) for frame in frames_data)
        # print(f"[TemporalPhaseFiltering] Points: {total_before} -> {points_count} "
        #       f"({points_count/max(total_before, 1)*100:.1f}% remaining)")
        
        if single_frame:
            results['points'] = wrapped[0]
        else:
            results['points'] = wrapped
        return results
    
    def _feature_indices(self, sample_points, np_points):
        """Locate amplitude/phase columns."""
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
    
    def _filter_sequence(self, frames_data, amp_idx, phase_idx):
        """Apply motion-aware filtering across frames."""
        filtered_frames = []
        prev_stats = deque(maxlen=self.history_size)
        static_map = {}
        motion_map = {}
        
        for frame in frames_data:
            if len(frame) == 0 or amp_idx is None or amp_idx >= frame.shape[1]:
                filtered_frames.append(frame)
                continue
            
            mask = self._filter_frame(frame, amp_idx, phase_idx,
                                      prev_stats, static_map, motion_map)
            filtered_frames.append(frame[mask])
        return filtered_frames
    
    def _filter_frame(self, frame, amp_idx, phase_idx, prev_stats, static_map, motion_map):
        amp = frame[:, amp_idx].astype(np.float32)
        phase = None
        if self.use_phase and phase_idx is not None and phase_idx < frame.shape[1]:
            phase = frame[:, phase_idx].astype(np.float32)
        
        xyz = spherical_to_cartesian(frame[:, :3])
        mask_amp = amp >= self.amp_threshold
        phase_dynamic = np.zeros(len(frame), dtype=bool)
        phase_static = np.zeros(len(frame), dtype=bool)
        
        stats = None
        if phase is not None:
            stats, keys = voxel_phase_stats(xyz, phase, amp, self.phase_voxel)
            latest_stats = prev_stats[-1] if len(prev_stats) > 0 else None
            
            if latest_stats is not None:
                for idx, key in enumerate(keys):
                    if key not in latest_stats:
                        continue
                    prev_mean_phase, _, _ = latest_stats[key]
                    cur_mean_phase, _, _ = stats.get(key, (0j, 0.0, 0))
                    if prev_mean_phase == 0j or cur_mean_phase == 0j:
                        continue
                    phase_delta = np.abs(np.angle(cur_mean_phase * np.conj(prev_mean_phase))) / np.pi
                    
                    motion_score = motion_map.get(key, 0.0) * self.motion_decay
                    if phase_delta >= self.phase_delta and amp[idx] >= self.amp_threshold * self.amp_relax:
                        motion_score += 1.0
                    motion_map[key] = motion_score
                    
                    if motion_score >= self.motion_keep:
                        phase_dynamic[idx] = True
                    
                    if phase_delta < self.static_delta:
                        static_map[key] = static_map.get(key, 0) + 1
                    else:
                        static_map[key] = 0
                    if static_map.get(key, 0) >= self.static_persist:
                        phase_static[idx] = True
            
            if stats is not None:
                prev_stats.append(stats)
                if len(prev_stats) > self.history_size:
                    prev_stats.popleft()
        
        mask = (mask_amp | phase_dynamic) & (~phase_static)
        return mask


@TRANSFORMS.register_module()
class MultiFrameAmplitudeFiltering(BaseTransform):
    def __init__(self,
                 amplitude_percentile: float = 30.0,
                 apply_per_frame: bool = True):
        self.amplitude_percentile = amplitude_percentile
        self.apply_per_frame = apply_per_frame
    
    def transform(self, results):
        points_list = results.get('points')
        
        if not isinstance(points_list, (list, tuple)):
            points_list = [points_list]
            single_frame = True
        else:
            single_frame = False
        
        filtered_frames = []
        
        if self.apply_per_frame:
            for points in points_list:
                if isinstance(points, RadarPoints):
                    coords = points.tensor.numpy()
                else:
                    coords = points
                
                if coords.shape[1] < 4:
                    filtered_frames.append(coords)
                    continue
                
                amp = coords[:, 3]
                threshold = np.percentile(amp, self.amplitude_percentile)
                mask = amp >= threshold
                
                filtered_frames.append(coords[mask])
        else:
            all_amps = []
            for points in points_list:
                if isinstance(points, RadarPoints):
                    coords = points.tensor.numpy()
                else:
                    coords = points
                if coords.shape[1] >= 4:
                    all_amps.append(coords[:, 3])
            
            if all_amps:
                global_threshold = np.percentile(np.concatenate(all_amps), self.amplitude_percentile)
                
                for points in points_list:
                    if isinstance(points, RadarPoints):
                        coords = points.tensor.numpy()
                    else:
                        coords = points
                    
                    if coords.shape[1] >= 4:
                        mask = coords[:, 3] >= global_threshold
                        filtered_frames.append(coords[mask])
                    else:
                        filtered_frames.append(coords)
            else:
                filtered_frames = [points.tensor.numpy() if isinstance(points, RadarPoints) else points 
                                   for points in points_list]
        
        attribute_dims = None
        points_dim = 3
        if len(points_list) > 0 and isinstance(points_list[0], RadarPoints):
            attribute_dims = points_list[0].attribute_dims
            points_dim = points_list[0].points_dim
        
        if single_frame:
            if isinstance(results['points'], RadarPoints):
                results['points'] = RadarPoints(filtered_frames[0], points_dim=points_dim,
                                               attribute_dims=attribute_dims)
            else:
                results['points'] = filtered_frames[0]
        else:
            if isinstance(points_list[0], RadarPoints):
                results['points'] = [RadarPoints(frame, points_dim=points_dim,
                                                 attribute_dims=attribute_dims) 
                                    for frame in filtered_frames]
            else:
                results['points'] = filtered_frames
        
        return results


@TRANSFORMS.register_module()
class UniformSampling(BaseTransform):
    
    def __init__(self,
                 num_points_per_frame: int = 50000,
                 sampling_method: str = 'random',
                 sampling_ratio: float = None,
                 min_points: int = 1):
        self.num_points_per_frame = num_points_per_frame
        self.sampling_method = sampling_method
        self.sampling_ratio = sampling_ratio
        self.min_points = min_points
        assert sampling_method in ['random', 'fps'], f"Unsupported sampling method: {sampling_method}"
        if sampling_ratio is not None:
            assert 0 < sampling_ratio <= 1, "sampling_ratio must be in (0, 1]"
    
    def transform(self, results):
        points_list = results.get('points')
        
        if not isinstance(points_list, (list, tuple)):
            points_list = [points_list]
            single_frame = True
        else:
            single_frame = False
        
        # Count total points before sampling
        total_points_before = 0
        for points in points_list:
            if isinstance(points, RadarPoints):
                total_points_before += len(points.tensor)
            else:
                total_points_before += len(points)
        
        # print(f"[UniformSampling] Points before sampling: {total_points_before} "
        #       f"(target: {self.num_points_per_frame * len(points_list)} = "
        #       f"{self.num_points_per_frame} × {len(points_list)} frames)")
        
        # 提取attribute_dims
        attribute_dims = None
        points_dim = 3
        if len(points_list) > 0 and isinstance(points_list[0], RadarPoints):
            attribute_dims = points_list[0].attribute_dims
            points_dim = points_list[0].points_dim
        
        sampled_frames = []
        
        for points in points_list:
            if isinstance(points, RadarPoints):
                coords = points.tensor.numpy()
            else:
                coords = points

            num_points = len(coords)
            if self.sampling_ratio is not None:
                target_points = max(self.min_points, int(num_points * self.sampling_ratio))
            elif isinstance(self.num_points_per_frame, float) and 0 < self.num_points_per_frame <= 1:
                target_points = max(self.min_points, int(num_points * self.num_points_per_frame))
            else:
                target_points = int(self.num_points_per_frame)

            if num_points <= target_points:
                sampled_frames.append(coords)
            else:
                if self.sampling_method == 'random':
                    indices = np.random.choice(num_points, target_points, replace=False)
                    sampled = coords[indices]
                else:
                    sampled = self._fps_sampling(coords, target_points)
                
                sampled_frames.append(sampled)
        
        if single_frame:
            if isinstance(results['points'], RadarPoints):
                results['points'] = RadarPoints(sampled_frames[0], points_dim=points_dim,
                                               attribute_dims=attribute_dims)
            else:
                results['points'] = sampled_frames[0]
        else:
            if isinstance(points_list[0], RadarPoints):
                results['points'] = [RadarPoints(frame, points_dim=points_dim,
                                                 attribute_dims=attribute_dims) 
                                    for frame in sampled_frames]
            else:
                results['points'] = sampled_frames
        
        return results

    
    def _fps_sampling(self, points, num_samples):
        coords = points[:, :3]
        
        r, theta, phi = coords[:, 0], coords[:, 1], coords[:, 2]
        x = r * np.cos(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.cos(phi)
        z = r * np.sin(phi)
        xyz = np.stack([x, y, z], axis=1)
        
        n_points = len(xyz)
        selected_indices = np.zeros(num_samples, dtype=int)
        distances = np.ones(n_points) * 1e10
        
        farthest = np.random.randint(0, n_points)
        
        for i in range(num_samples):
            selected_indices[i] = farthest
            centroid = xyz[farthest]
            dist = np.sum((xyz - centroid) ** 2, axis=1)
            distances = np.minimum(distances, dist)
            farthest = np.argmax(distances)
        
        return points[selected_indices]
