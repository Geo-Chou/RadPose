import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


DEFAULT_POINT_CLOUD_RANGE = [
    0.12,
    (30 / 180) * math.pi,
    -(30 / 180) * math.pi,
    4.,
    (150 / 180) * math.pi,
    (30 / 180) * math.pi,
]


def get_args():
    parser = argparse.ArgumentParser(
        description='Remove mmWave samples whose keypoints_sp fall outside point_cloud_range')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument(
        '--source',
        type=str,
        default='auto',
        choices=['auto', 'metainfo', 'dirs'],
        help='Prefer metainfo cleanup, or scan split directories directly')
    parser.add_argument(
        '--point-cloud-range',
        type=float,
        nargs=6,
        default=DEFAULT_POINT_CLOUD_RANGE,
        metavar=('R_MIN', 'AZ_MIN', 'EL_MIN', 'R_MAX', 'AZ_MAX', 'EL_MAX'))
    parser.add_argument(
        '--splits',
        type=str,
        nargs='*',
        default=['train', 'val', 'test'],
        help='Splits to scan when source=dirs or metainfo entries include them')
    parser.add_argument('--dry-run', action='store_true', help='Report only, do not delete files')
    return parser.parse_args()


def load_json(path: Path):
    with path.open('r') as f:
        return json.load(f)


def dump_json(path: Path, data) -> None:
    with path.open('w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_finite_number(value) -> bool:
    if value is None:
        return False
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value)


def joint_in_range(joint: List[float], point_cloud_range: List[float]) -> Optional[bool]:
    if not isinstance(joint, list) or len(joint) < 3:
        return None
    coords = joint[:3]
    if not all(is_finite_number(v) for v in coords):
        return None
    r_min, az_min, el_min, r_max, az_max, el_max = point_cloud_range
    r, az, el = [float(v) for v in coords]
    return (r_min <= r <= r_max and
            az_min <= az <= az_max and
            el_min <= el <= el_max)


def annotation_out_of_range(ann, point_cloud_range: List[float]) -> Tuple[bool, str]:
    if not isinstance(ann, list):
        return True, 'annotation_not_list'

    for person in ann:
        joints = person.get('keypoints_sp')
        if not isinstance(joints, list):
            return True, 'missing_keypoints_sp'
        for joint in joints:
            in_range = joint_in_range(joint, point_cloud_range)
            if in_range is None:
                continue
            if not in_range:
                return True, 'joint_out_of_range'
    return False, 'ok'


def safe_unlink(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if path.exists():
        path.unlink()


def infer_radar_path(data_root: Path, ann_rel: str, radar_rel: Optional[str]) -> Path:
    if radar_rel:
        return data_root / radar_rel
    ann_path = data_root / ann_rel
    split = ann_path.parent.parent.name
    stem = ann_path.stem
    return data_root / split / 'radar_points' / f'{stem}.npz'


def filter_by_metainfo(data_root: Path,
                       point_cloud_range: List[float],
                       dry_run: bool,
                       splits: List[str]) -> None:
    metainfo_path = data_root / 'metainfo.json'
    if not metainfo_path.exists():
        raise FileNotFoundError(f'metainfo.json not found: {metainfo_path}')

    metainfo = load_json(metainfo_path)
    if not isinstance(metainfo, list):
        raise TypeError('metainfo.json must be a list')

    kept_entries = []
    removed_counts: Dict[str, int] = {}
    removed_reasons: Dict[str, int] = {}

    for entry in tqdm(metainfo, desc='Filtering metainfo'):
        split = entry.get('split', 'unknown')
        ann_rel = entry.get('annotation_file')
        radar_rel = entry.get('radar_file')

        if splits and split not in splits:
            kept_entries.append(entry)
            continue

        if not ann_rel:
            removed_counts[split] = removed_counts.get(split, 0) + 1
            removed_reasons['missing_annotation_file'] = (
                removed_reasons.get('missing_annotation_file', 0) + 1)
            continue

        ann_path = data_root / ann_rel
        radar_path = infer_radar_path(data_root, ann_rel, radar_rel)
        if not ann_path.exists():
            removed_counts[split] = removed_counts.get(split, 0) + 1
            removed_reasons['annotation_missing'] = (
                removed_reasons.get('annotation_missing', 0) + 1)
            continue

        ann = load_json(ann_path)
        should_remove, reason = annotation_out_of_range(ann, point_cloud_range)
        if should_remove:
            removed_counts[split] = removed_counts.get(split, 0) + 1
            removed_reasons[reason] = removed_reasons.get(reason, 0) + 1
            safe_unlink(ann_path, dry_run)
            safe_unlink(radar_path, dry_run)
            continue

        kept_entries.append(entry)

    if not dry_run:
        dump_json(metainfo_path, kept_entries)

    print('\nRemoved samples by split:')
    for split in sorted(removed_counts):
        print(f'{split}: {removed_counts[split]}')
    print('\nRemoved reasons:')
    for reason in sorted(removed_reasons):
        print(f'{reason}: {removed_reasons[reason]}')
    print(f'\nKept entries: {len(kept_entries)}')
    print(f'Removed entries: {len(metainfo) - len(kept_entries)}')


def filter_by_dirs(data_root: Path,
                   point_cloud_range: List[float],
                   dry_run: bool,
                   splits: List[str]) -> None:
    removed_counts: Dict[str, int] = {}
    removed_reasons: Dict[str, int] = {}
    removed_ann_rels = set()

    for split in splits:
        ann_dir = data_root / split / 'annotations'
        radar_dir = data_root / split / 'radar_points'
        if not ann_dir.exists():
            continue

        ann_files = sorted(ann_dir.glob('*.json'))
        for ann_path in tqdm(ann_files, desc=f'Filtering {split}'):
            ann = load_json(ann_path)
            should_remove, reason = annotation_out_of_range(ann, point_cloud_range)
            if not should_remove:
                continue

            removed_counts[split] = removed_counts.get(split, 0) + 1
            removed_reasons[reason] = removed_reasons.get(reason, 0) + 1
            removed_ann_rels.add(str(ann_path.relative_to(data_root)))
            safe_unlink(ann_path, dry_run)
            safe_unlink(radar_dir / f'{ann_path.stem}.npz', dry_run)

    metainfo_path = data_root / 'metainfo.json'
    if metainfo_path.exists():
        metainfo = load_json(metainfo_path)
        if not isinstance(metainfo, list):
            raise TypeError('metainfo.json must be a list')
        kept_entries = []
        for entry in metainfo:
            ann_rel = entry.get('annotation_file')
            radar_rel = entry.get('radar_file')
            radar_path = data_root / radar_rel if radar_rel else None
            ann_path = data_root / ann_rel if ann_rel else None
            if (ann_rel in removed_ann_rels or ann_path is None or not ann_path.exists() or
                    (radar_path is not None and not radar_path.exists())):
                continue
            kept_entries.append(entry)
        if not dry_run:
            dump_json(metainfo_path, kept_entries)
        print(f'\nKept entries: {len(kept_entries)}')
        print(f'Removed entries from metainfo: {len(metainfo) - len(kept_entries)}')
    else:
        print('\nmetainfo.json not found; no metainfo update performed.')

    print('\nRemoved samples by split:')
    for split in sorted(removed_counts):
        print(f'{split}: {removed_counts[split]}')
    print('\nRemoved reasons:')
    for reason in sorted(removed_reasons):
        print(f'{reason}: {removed_reasons[reason]}')


def main():
    args = get_args()
    data_root = Path(args.data_root)
    metainfo_path = data_root / 'metainfo.json'

    source = args.source
    if source == 'auto':
        source = 'metainfo' if metainfo_path.exists() else 'dirs'

    if source == 'metainfo':
        filter_by_metainfo(
            data_root=data_root,
            point_cloud_range=args.point_cloud_range,
            dry_run=args.dry_run,
            splits=args.splits)
    else:
        filter_by_dirs(
            data_root=data_root,
            point_cloud_range=args.point_cloud_range,
            dry_run=args.dry_run,
            splits=args.splits)


if __name__ == '__main__':
    main()
