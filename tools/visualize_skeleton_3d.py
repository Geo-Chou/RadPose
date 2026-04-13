#!/usr/bin/env python
"""
3D Skeleton Visualization Tool for MatchedPredictionDump outputs.

Usage:
    python tools/visualize_skeleton_3d.py \
        --input work_dirs/matched_predictions_ik/ \
        --output visualizations/ik_skeletons/ \
        --max-samples 100 \
        --show-errors
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# COCO-17 skeleton definition (16 joints after removing nose)
# Joint order: [0右脚踝, 1右膝, 2右髋, 3左髋, 4左膝, 5左脚踝,
#              6骨盆, 7脊柱, 8脖子, 9头, 10右手腕, 11右肘, 12右肩,
#              13左肩, 14左肘, 15左手腕]
SKELETON_EDGES = [
    (0, 1),   # 右脚踝 -> 右膝
    (1, 2),   # 右膝 -> 右髋
    (2, 6),   # 右髋 -> 骨盆
    (3, 6),   # 左髋 -> 骨盆
    (4, 3),   # 左膝 -> 左髋
    (5, 4),   # 左脚踝 -> 左膝
    (6, 7),   # 骨盆 -> 脊柱
    (7, 8),   # 脊柱 -> 脖子
    (8, 9),   # 脖子 -> 头
    (10, 11), # 右手腕 -> 右肘
    (11, 12), # 右肘 -> 右肩
    (12, 7),  # 右肩 -> 脊柱
    (13, 7),  # 左肩 -> 脊柱
    (14, 13), # 左肘 -> 左肩
    (15, 14), # 左手腕 -> 左肘
]

JOINT_NAMES = [
    'R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle',
    'Pelvis', 'Spine', 'Neck', 'Head',
    'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist'
]


def spherical_to_cartesian(coords: np.ndarray) -> np.ndarray:
    """Convert spherical [r, a, e] to Cartesian [x, y, z] with dataset convention.
    
    Args:
        coords: (..., 3) array with [r, a, e]
    
    Returns:
        (..., 3) array with [x, y, z]
    """
    r = coords[..., 0]
    a = coords[..., 1]  # azimuth
    e = coords[..., 2]  # elevation
    
    x = r * np.cos(a) * np.cos(e)
    y = r * np.sin(e)
    z = r * np.sin(a) * np.cos(e)
    
    return np.stack([x, y, z], axis=-1)


def load_prediction(json_path: Path) -> Any:
    """Load a single prediction JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def _extract_field(data: Any, key: str) -> Optional[List]:
    """Extract a field from either a dict record or a list of dict records."""
    if isinstance(data, dict):
        return data.get(key, None)
    if isinstance(data, list):
        collected = []
        found = False
        for item in data:
            if not isinstance(item, dict) or key not in item:
                continue
            found = True
            value = np.array(item[key])
            if value.ndim <= 2:
                collected.append(value.tolist())
            else:
                collected.extend(value.tolist())
        return collected if found else None
    raise TypeError(f'Unsupported JSON root type: {type(data).__name__}')


def parse_keypoints(data: Any, use_spherical: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Parse keypoints from prediction data.
    
    Args:
        data: Dictionary containing prediction results
        use_spherical: If True, use spherical coordinates, else use Cartesian
    
    Returns:
        pred_kpts: (N, J, 3) predicted keypoints in Cartesian coordinates
        gt_kpts: (N, J, 3) ground truth keypoints or None
    """
    if use_spherical:
        pred_key = 'keypoints_sp'
        gt_key = 'gt_keypoints_sp'
    else:
        pred_key = 'keypoints_3d'
        gt_key = 'gt_keypoints_3d'
    
    # Parse predictions
    pred_list = _extract_field(data, pred_key) or []
    if not pred_list:
        raise ValueError(f"No {pred_key} found in data")
    
    pred_kpts = np.array(pred_list)  # (N, J, 4) or (J, 4)
    if pred_kpts.ndim == 2:
        pred_kpts = pred_kpts[np.newaxis, ...]  # (1, J, 4)
    pred_kpts = pred_kpts[..., :3]  # (N, J, 3)
    
    # Convert spherical to Cartesian if needed
    if use_spherical:
        pred_kpts = spherical_to_cartesian(pred_kpts)
    
    # Parse ground truth if available
    gt_kpts = None
    gt_list = _extract_field(data, gt_key)
    if gt_list:
        gt_kpts = np.array(gt_list)
        if gt_kpts.ndim == 2:
            gt_kpts = gt_kpts[np.newaxis, ...]
        gt_kpts = gt_kpts[..., :3]
        if use_spherical:
            gt_kpts = spherical_to_cartesian(gt_kpts)
    
    return pred_kpts, gt_kpts


def plot_skeleton_3d(ax: Axes3D,
                     keypoints: np.ndarray,
                     edges: List[Tuple[int, int]],
                     color: str = 'blue',
                     label: str = 'Skeleton',
                     alpha: float = 1.0,
                     linewidth: float = 2.0) -> None:
    """Plot a single 3D skeleton.
    
    Args:
        ax: 3D axes to plot on
        keypoints: (J, 3) joint positions
        edges: List of (child, parent) tuples
        color: Color for skeleton
        label: Label for legend
        alpha: Transparency
        linewidth: Line width for bones
    """
    # Filter out invalid joints (all zeros)
    valid_mask = np.abs(keypoints).sum(axis=-1) > 1e-6
    
    # Plot joints
    valid_kpts = keypoints[valid_mask]
    if len(valid_kpts) > 0:
        ax.scatter(valid_kpts[:, 0], valid_kpts[:, 1], valid_kpts[:, 2],
                   c=color, s=50, alpha=alpha, label=label, marker='o')
    
    # Plot bones
    for child, parent in edges:
        if valid_mask[child] and valid_mask[parent]:
            x = [keypoints[child, 0], keypoints[parent, 0]]
            y = [keypoints[child, 1], keypoints[parent, 1]]
            z = [keypoints[child, 2], keypoints[parent, 2]]
            ax.plot(x, y, z, color=color, alpha=alpha, linewidth=linewidth)


def visualize_sample(pred_kpts: np.ndarray,
                     gt_kpts: Optional[np.ndarray],
                     errors: Optional[np.ndarray],
                     sample_name: str,
                     output_path: Path,
                     show_errors: bool = False,
                     show_gt: bool = True) -> None:
    """Visualize a single sample with multiple persons.
    
    Args:
        pred_kpts: (N, J, 3) predicted keypoints
        gt_kpts: (N, J, 3) ground truth keypoints or None
        errors: (N, J) per-joint errors or None
        sample_name: Sample name for title and filename
        output_path: Output directory
        show_errors: Whether to show error colormap
        show_gt: Whether to render ground-truth skeleton
    """
    num_person = pred_kpts.shape[0]
    
    # Create figure with subplots for each person
    fig = plt.figure(figsize=(6 * num_person, 6))
    
    for person_idx in range(num_person):
        ax = fig.add_subplot(1, num_person, person_idx + 1, projection='3d')
        
        pred = pred_kpts[person_idx]
        # Display transform to plotting coords:
        # dataset [x(right), y(down), z(forward)] -> plot [x, z, -y]
        # so plot axes are X(right), Z(forward), Y(down)
        pred_display = pred[:, [0, 2, 1]].copy()
        pred_display[:, 2] *= -1
        
        # Check if person is valid (not all zeros)
        if np.abs(pred).sum() < 1e-6:
            ax.text(0, 0, 0, 'No detection', ha='center', va='center', fontsize=14)
            # ax.set_title(f'Person {person_idx + 1}')
            continue
        
        # Plot ground truth
        if show_gt and gt_kpts is not None:
            gt = gt_kpts[person_idx]
            if np.abs(gt).sum() > 1e-6:
                gt_display = gt[:, [0, 2, 1]].copy()
                gt_display[:, 2] *= -1
                plot_skeleton_3d(ax, gt_display, SKELETON_EDGES,
                               color='green', label='Ground Truth',
                               alpha=0.6, linewidth=2.5)
        
        # Plot prediction
        if show_errors and errors is not None:
            # Color joints by error magnitude
            err = errors[person_idx]  # (J,)
            valid_mask = np.abs(pred).sum(axis=-1) > 1e-6
            valid_err = err[valid_mask]
            
            if len(valid_err) > 0:
                norm = plt.Normalize(vmin=0, vmax=np.percentile(valid_err, 95))
                cmap = cm.get_cmap('hot')
                
                # Plot joints with error colormap
                for j in range(len(pred)):
                    if valid_mask[j]:
                        c = cmap(norm(err[j]))
                        ax.scatter(pred_display[j, 0], pred_display[j, 1], pred_display[j, 2],
                                 c=[c], s=100, marker='o', edgecolors='black', linewidths=1.5)
                
                # Plot bones
                for child, parent in SKELETON_EDGES:
                    if valid_mask[child] and valid_mask[parent]:
                        avg_err = (err[child] + err[parent]) / 2
                        c = cmap(norm(avg_err))
                        x = [pred_display[child, 0], pred_display[parent, 0]]
                        y = [pred_display[child, 1], pred_display[parent, 1]]
                        z = [pred_display[child, 2], pred_display[parent, 2]]
                        ax.plot(x, y, z, color=c, linewidth=3.0)
                
                # Add colorbar
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
                cbar.set_label('Error (m)', rotation=270, labelpad=15)
        else:
            plot_skeleton_3d(ax, pred_display, SKELETON_EDGES,
                           color='blue', label='Prediction',
                           alpha=0.8, linewidth=2.5)
        
        # Set equal aspect ratio using both prediction and GT (if available),
        # so GT won't be clipped when far from prediction.
        points_for_bounds = [pred_display]
        if show_gt and gt_kpts is not None and person_idx < gt_kpts.shape[0]:
            gt = gt_kpts[person_idx]
            if np.abs(gt).sum() > 1e-6:
                gt_display = gt[:, [0, 2, 1]].copy()
                gt_display[:, 2] *= -1
                points_for_bounds.append(gt_display)

        all_points = np.concatenate(points_for_bounds, axis=0)
        valid_points = all_points[np.abs(all_points).sum(axis=-1) > 1e-6]
        if valid_points.shape[0] == 0:
            valid_points = all_points

        min_xyz = valid_points.min(axis=0)
        max_xyz = valid_points.max(axis=0)
        ranges = max_xyz - min_xyz
        max_range = max(ranges.max() / 2.0, 1e-3)
        mid_x, mid_y, mid_z = (min_xyz + max_xyz) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Z (m)', fontsize=10)
        ax.set_zlabel('Y (down, m)', fontsize=10)
        # ax.set_title(f'Person {person_idx + 1}', fontsize=12)
        
        # Add legend
        if not show_errors or errors is None:
            ax.legend(loc='upper right', fontsize=8)
        
        # Initial view: look toward +X and +Z (dataset) at 45 degrees.
        # In plot coords, dataset Z is matplotlib Y.
        ax.view_init(elev=10, azim=-45)
    
    # plt.suptitle(f'Sample {sample_name}', fontsize=14, y=0.98)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / f'{sample_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Saved: {output_file}')


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D skeletons from MatchedPredictionDump')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing prediction JSON files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for visualization images')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to visualize')
    parser.add_argument('--start-file', type=str, default=None,
                       help='Start from this file (name or path) and process consecutive samples')
    parser.add_argument('--use-spherical', action='store_true',
                       help='Use spherical coordinates instead of Cartesian')
    parser.add_argument('--show-errors', action='store_true',
                       help='Show per-joint errors with colormap (requires include_errors=True)')
    parser.add_argument('--pred-only', action='store_true',
                       help='Render prediction only, do not draw ground truth')
    parser.add_argument('--pattern', type=str, default='*.json',
                       help='File pattern to match (default: *.json)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all prediction files
    pred_files = sorted(input_dir.glob(args.pattern))
    
    if not pred_files:
        print(f'No files found matching {args.pattern} in {input_dir}')
        return
    
    total_files = len(pred_files)
    print(f'Found {total_files} prediction files')
    
    # Process consecutive files from a specified start file if provided.
    # Default to 50 samples for this mode unless max_samples is explicitly set.
    if args.start_file is not None:
        start_name = Path(args.start_file).name
        start_idx = next((i for i, p in enumerate(pred_files) if p.name == start_name), -1)
        if start_idx < 0:
            print(f'Start file not found: {start_name}')
            return
        sample_count = args.max_samples if args.max_samples is not None else 50
        end_idx = min(start_idx + sample_count, total_files)
        pred_files = pred_files[start_idx:end_idx]
        print(f'Processing consecutive {len(pred_files)} samples from {start_name} '
              f'(index {start_idx} to {end_idx - 1})')
    # Otherwise randomly sample files if max_samples is specified
    elif args.max_samples is not None:
        sample_count = min(args.max_samples, total_files)
        pred_files = random.sample(pred_files, sample_count)
        print(f'Randomly selected {len(pred_files)} samples from {total_files} files')
    
    # Process each file
    for idx, pred_file in enumerate(pred_files):
        try:
            data = load_prediction(pred_file)
            sample_name = pred_file.stem
            
            # Parse keypoints
            pred_kpts, gt_kpts = parse_keypoints(data, use_spherical=args.use_spherical)
            
            # Parse errors if available
            errors = None
            if args.show_errors:
                if args.use_spherical:
                    errors_data = _extract_field(data, 'errors_sp')
                else:
                    errors_data = _extract_field(data, 'errors_3d')
                
                if errors_data is not None:
                    errors = np.array(errors_data)
                    if errors.ndim == 1:
                        errors = errors[np.newaxis, :]
                else:
                    print(f'Warning: No error data in {pred_file.name}, skipping error visualization')
            
            # Visualize
            visualize_sample(
                pred_kpts, gt_kpts, errors, sample_name, output_dir,
                show_errors=args.show_errors,
                show_gt=not args.pred_only)
            
        except Exception as e:
            print(f'Error processing {pred_file.name}: {e}')
            continue
    
    print(f'\nVisualization complete! Images saved to {output_dir}')
    print(f'Total: {len(list(output_dir.glob("*.png")))} images')


if __name__ == '__main__':
    main()
