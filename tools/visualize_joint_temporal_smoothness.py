#!/usr/bin/env python
"""Visualize temporal smoothness for one 3D joint from prediction JSON files.

Example:
    python tools/visualize_joint_temporal_smoothness.py \
        --input work_dirs/predictions_openarea/predictions_w_sp_ik \
        --joint R_Wrist \
        --output work_dirs/predictions_openarea/r_wrist_temporal_smoothness.png \
        --show-gt
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


JOINT_NAMES = [
    'R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle',
    'Pelvis', 'Spine', 'Neck', 'Head',
    'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder', 'L_Elbow', 'L_Wrist'
]

JOINT_ALIASES = {
    'right_wrist': 'R_Wrist',
    'r_wrist': 'R_Wrist',
    'rwrist': 'R_Wrist',
    'left_wrist': 'L_Wrist',
    'l_wrist': 'L_Wrist',
    'lwrist': 'L_Wrist',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot x/y/z temporal continuity for one 3D joint.')
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Directory containing per-frame prediction JSON files.')
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output image path. Defaults to <input>/<joint>_temporal_smoothness.png.')
    parser.add_argument(
        '--joint',
        default='R_Wrist',
        help='Joint name or index. Common wrist choices: R_Wrist/index 10, L_Wrist/index 15.')
    parser.add_argument(
        '--person',
        type=int,
        default=0,
        help='Person index inside each JSON record. Default: 0.')
    parser.add_argument(
        '--key',
        default='keypoints_3d',
        help='Prediction key to read. Default: keypoints_3d.')
    parser.add_argument(
        '--gt-key',
        default='gt_keypoints_3d',
        help='Ground-truth key to read when --show-gt is set.')
    parser.add_argument(
        '--show-gt',
        action='store_true',
        help='Overlay ground-truth x/y/z curves when available.')
    parser.add_argument(
        '--frame-start',
        type=int,
        default=None,
        help='Start index in the sorted JSON file list, inclusive. Example: 50.')
    parser.add_argument(
        '--frame-end',
        type=int,
        default=None,
        help='End index in the sorted JSON file list, inclusive. Example: 100.')
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Limit number of sorted JSON files to plot.')
    parser.add_argument(
        '--dpi',
        type=int,
        default=180,
        help='Output image DPI.')
    return parser.parse_args()


def resolve_joint(joint: str) -> Tuple[int, str]:
    joint_text = str(joint).strip()
    if joint_text.isdigit():
        joint_idx = int(joint_text)
        if joint_idx < 0 or joint_idx >= len(JOINT_NAMES):
            raise ValueError(f'Joint index must be in [0, {len(JOINT_NAMES) - 1}], got {joint_idx}')
        return joint_idx, JOINT_NAMES[joint_idx]

    normalized = joint_text.lower().replace('-', '_')
    joint_name = JOINT_ALIASES.get(normalized, joint_text)
    if joint_name not in JOINT_NAMES:
        choices = ', '.join(f'{idx}:{name}' for idx, name in enumerate(JOINT_NAMES))
        raise ValueError(f'Unknown joint {joint!r}. Choices: {choices}')
    return JOINT_NAMES.index(joint_name), joint_name


def frame_sort_key(path: Path) -> Tuple[int, Any]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


def iter_json_files(
    input_dir: Path,
    frame_start: Optional[int],
    frame_end: Optional[int],
    max_frames: Optional[int],
) -> Iterable[Tuple[int, Path]]:
    files = sorted(input_dir.glob('*.json'), key=frame_sort_key)
    start = 0 if frame_start is None else frame_start
    end = len(files) - 1 if frame_end is None else frame_end
    if start < 0:
        raise ValueError(f'--frame-start must be >= 0, got {start}')
    if end < start:
        raise ValueError(f'--frame-end must be >= --frame-start, got {end} < {start}')
    indexed_files = list(enumerate(files))
    indexed_files = indexed_files[start:end + 1]
    if max_frames is not None:
        indexed_files = indexed_files[:max_frames]
    return indexed_files


def read_records(path: Path) -> List[Dict[str, Any]]:
    with path.open('r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise TypeError(f'{path} has unsupported root type: {type(data).__name__}')


def extract_joint(record: Dict[str, Any], key: str, joint_idx: int) -> Optional[np.ndarray]:
    if key not in record:
        return None
    arr = np.asarray(record[key], dtype=float)
    if arr.ndim != 2 or arr.shape[0] <= joint_idx or arr.shape[1] < 3:
        return None
    xyz = arr[joint_idx, :3]
    if not np.isfinite(xyz).all():
        return None
    return xyz


def load_series(
    input_dir: Path,
    joint_idx: int,
    person: int,
    pred_key: str,
    gt_key: str,
    show_gt: bool,
    frame_start: Optional[int],
    frame_end: Optional[int],
    max_frames: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    frames: List[int] = []
    pred_xyz: List[np.ndarray] = []
    gt_xyz: List[np.ndarray] = []
    frame_names: List[str] = []

    for frame_idx, path in iter_json_files(input_dir, frame_start, frame_end, max_frames):
        records = read_records(path)
        if person >= len(records):
            continue
        pred = extract_joint(records[person], pred_key, joint_idx)
        if pred is None:
            continue
        frames.append(frame_idx)
        pred_xyz.append(pred)
        frame_names.append(path.stem)

        if show_gt:
            gt = extract_joint(records[person], gt_key, joint_idx)
            if gt is None:
                gt_xyz.append(np.array([np.nan, np.nan, np.nan], dtype=float))
            else:
                gt_xyz.append(gt)

    if not pred_xyz:
        raise ValueError(f'No valid {pred_key} joint data found in {input_dir}')

    gt_arr = np.vstack(gt_xyz) if show_gt and gt_xyz else None
    return np.asarray(frames), np.vstack(pred_xyz), gt_arr, frame_names


def finite_diff(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, axis=0)


def path_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    diffs = np.diff(xy, axis=0)
    valid = np.isfinite(diffs).all(axis=1)
    return float(np.linalg.norm(diffs[valid], axis=1).sum())


def mean_abs_acceleration(xy: np.ndarray) -> float:
    if len(xy) < 3:
        return 0.0
    accel = np.diff(xy, n=2, axis=0)
    valid = np.isfinite(accel).all(axis=1)
    if not valid.any():
        return float('nan')
    return float(np.linalg.norm(accel[valid], axis=1).mean())


def apply_common_style(ax: plt.Axes) -> None:
    ax.grid(True, color='#d8dee9', linewidth=0.7, alpha=0.85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_smoothness(
    frames: np.ndarray,
    pred_xyz: np.ndarray,
    gt_xyz: Optional[np.ndarray],
    joint_name: str,
    output: Path,
    dpi: int,
) -> None:
    fig, ax_series = plt.subplots(figsize=(12, 4.8), constrained_layout=True)

    pred_color_x = '#1f77b4'
    pred_color_y = '#d62728'
    pred_color_z = '#2ca02c'
    gt_color_x = '#7aa6d8'
    gt_color_y = '#e08a8a'
    gt_color_z = '#8fd18f'

    ax_series.plot(frames, pred_xyz[:, 0], color=pred_color_x, linewidth=2.0, label='Pred X')
    ax_series.plot(frames, pred_xyz[:, 1], color=pred_color_y, linewidth=2.0, label='Pred Y')
    ax_series.plot(frames, pred_xyz[:, 2], color=pred_color_z, linewidth=2.0, label='Pred Z')
    if gt_xyz is not None:
        ax_series.plot(frames, gt_xyz[:, 0], color=gt_color_x, linewidth=1.6, linestyle='--', label='GT X')
        ax_series.plot(frames, gt_xyz[:, 1], color=gt_color_y, linewidth=1.6, linestyle='--', label='GT Y')
        ax_series.plot(frames, gt_xyz[:, 2], color=gt_color_z, linewidth=1.6, linestyle='--', label='GT Z')
    ax_series.set_xlabel('Frame index')
    ax_series.set_ylabel('Coordinate value')
    ax_series.legend(ncol=6 if gt_xyz is not None else 3, frameon=False, loc='upper right')
    apply_common_style(ax_series)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    joint_idx, joint_name = resolve_joint(args.joint)
    output = args.output
    if output is None:
        output = args.input / f'{joint_name.lower()}_temporal_smoothness.png'

    frames, pred_xyz, gt_xyz, _ = load_series(
        args.input,
        joint_idx,
        args.person,
        args.key,
        args.gt_key,
        args.show_gt,
        args.frame_start,
        args.frame_end,
        args.max_frames,
    )
    plot_smoothness(frames, pred_xyz, gt_xyz, joint_name, output, args.dpi)
    print(f'Saved {joint_name} temporal smoothness plot to {output}')


if __name__ == '__main__':
    main()
