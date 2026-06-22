import argparse
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

SKELETON_EDGES_16 = [
    (0, 1), (1, 2), (2, 6),        # left leg
    (5, 4), (4, 3), (3, 6),        # right leg
    (6, 7),                       # body
    (10, 11), (11, 12), (12, 7),   # left arm
    (15, 14), (14, 13), (13, 7),   # right arm
    (7, 8), (8, 9)                # head and neck
]


def get_args():
    parser = argparse.ArgumentParser(
        description='Clean annotations with null keypoints and compute skeleton length stats')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--dry-run', action='store_true', help='Only report, no file deletion')
    return parser.parse_args()


def is_invalid_keypoints(kpts):
    if not isinstance(kpts, list) or len(kpts) == 0:
        return True
    for kp in kpts:
        if not isinstance(kp, list) or len(kp) < 3:
            return True
        for v in kp[:3]:
            if v is None:
                return True
            try:
                if isinstance(v, float) and math.isnan(v):
                    return True
            except Exception:
                return True
    return False


def load_json(path: Path):
    with path.open('r') as f:
        return json.load(f)


def compute_stats(entries, data_root: Path):
    sum_len = np.zeros(len(SKELETON_EDGES_16), dtype=np.float64)
    sum_sq = np.zeros(len(SKELETON_EDGES_16), dtype=np.float64)
    count = np.zeros(len(SKELETON_EDGES_16), dtype=np.int64)

    for entry in tqdm(entries, desc='Computing stats'):
        ann_path = data_root / entry['annotation_file']
        if not ann_path.exists():
            continue
        ann = load_json(ann_path)
        for person in ann:
            kpts = person.get('keypoints_3d')
            if not isinstance(kpts, list):
                continue
            for i, (a, b) in enumerate(SKELETON_EDGES_16):
                if a >= len(kpts) or b >= len(kpts):
                    continue
                pa = kpts[a]
                pb = kpts[b]
                if (not isinstance(pa, list)) or (not isinstance(pb, list)):
                    continue
                if len(pa) < 3 or len(pb) < 3:
                    continue
                ax, ay, az = pa[:3]
                bx, by, bz = pb[:3]
                if None in (ax, ay, az, bx, by, bz):
                    continue
                try:
                    if any(math.isnan(v) for v in (ax, ay, az, bx, by, bz)):
                        continue
                except Exception:
                    continue
                dx = ax - bx
                dy = ay - by
                dz = az - bz
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                sum_len[i] += dist
                sum_sq[i] += dist * dist
                count[i] += 1

    mean_len = np.zeros_like(sum_len)
    var_len = np.zeros_like(sum_len)
    valid = count > 0
    mean_len[valid] = sum_len[valid] / count[valid]
    var_len[valid] = (sum_sq[valid] / count[valid]) - (mean_len[valid] ** 2)
    var_len = np.maximum(var_len, 0.0)
    return mean_len, var_len, count


def main():
    args = get_args()
    data_root = Path(args.data_root)
    metainfo_path = data_root / 'metainfo.json'

    if not metainfo_path.exists():
        raise FileNotFoundError(f'metainfo.json not found: {metainfo_path}')

    metainfo = load_json(metainfo_path)
    if not isinstance(metainfo, list):
        raise TypeError('metainfo.json must be a list')

    kept_entries = []
    removed_entries = []
    removed_counts = {'train': 0, 'val': 0}

    for entry in tqdm(metainfo, desc='Scanning annotations'):
        ann_rel = entry.get('annotation_file')
        radar_rel = entry.get('radar_file')
        split = entry.get('split')
        if not ann_rel or not radar_rel:
            removed_entries.append(entry)
            if split in removed_counts:
                removed_counts[split] += 1
            continue

        ann_path = data_root / ann_rel
        radar_path = data_root / radar_rel
        if not ann_path.exists():
            removed_entries.append(entry)
            if split in removed_counts:
                removed_counts[split] += 1
            continue

        ann = load_json(ann_path)
        invalid = False
        for person in ann:
            kpts = person.get('keypoints_3d')
            if is_invalid_keypoints(kpts):
                invalid = True
                break

        if invalid:
            removed_entries.append(entry)
            if split in removed_counts:
                removed_counts[split] += 1
            if not args.dry_run:
                if ann_path.exists():
                    ann_path.unlink()
                if radar_path.exists():
                    radar_path.unlink()
        else:
            kept_entries.append(entry)

    if not args.dry_run:
        with metainfo_path.open('w') as f:
            json.dump(kept_entries, f, indent=2, ensure_ascii=False)

    mean_len, var_len, counts = compute_stats(kept_entries, data_root)

    print('\nRemoved samples:')
    print(f"train: {removed_counts['train']}")
    print(f"val:   {removed_counts['val']}")
    print('\nSkeleton length stats (meters):')
    for i, (edge, mean, var, cnt) in enumerate(zip(SKELETON_EDGES_16, mean_len, var_len, counts)):
        print(f"edge {i} {edge}: mean={mean:.6f} var={var:.6f} count={int(cnt)}")


if __name__ == '__main__':
    main()
