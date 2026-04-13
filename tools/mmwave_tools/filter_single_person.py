import argparse
import json
from pathlib import Path

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(
        description='Filter samples where annotation is not exactly one person')
    parser.add_argument('--data-root', type=str, default='/workspace/mmpose/data')
    parser.add_argument(
        '--source',
        type=str,
        default='metainfo',
        choices=['metainfo', 'dirs'],
        help='Filter by metainfo.json or by scanning train/val directories')
    parser.add_argument('--dry-run', action='store_true', help='Only report, no file deletion')
    return parser.parse_args()


def load_json(path: Path):
    with path.open('r') as f:
        return json.load(f)


def is_single_person(ann):
    if not isinstance(ann, list):
        return False
    return len(ann) == 1


def remove_files(ann_path: Path, radar_path: Path, dry_run: bool):
    if dry_run:
        return
    if ann_path.exists():
        ann_path.unlink()
    if radar_path.exists():
        radar_path.unlink()


def filter_by_metainfo(data_root: Path, dry_run: bool):
    metainfo_path = data_root / 'metainfo.json'
    if not metainfo_path.exists():
        raise FileNotFoundError(f'metainfo.json not found: {metainfo_path}')

    metainfo = load_json(metainfo_path)
    if not isinstance(metainfo, list):
        raise TypeError('metainfo.json must be a list')

    kept_entries = []
    removed_entries = []
    removed_counts = {'train': 0, 'val': 0, 'other': 0}
    removed_reasons = {'missing': 0, 'not_single': 0, 'invalid': 0}

    for entry in tqdm(metainfo, desc='Filtering by metainfo'):
        ann_rel = entry.get('annotation_file')
        radar_rel = entry.get('radar_file')
        split = entry.get('split', 'other')
        if split not in removed_counts:
            split = 'other'

        if not ann_rel or not radar_rel:
            removed_entries.append(entry)
            removed_counts[split] += 1
            removed_reasons['missing'] += 1
            continue

        ann_path = data_root / ann_rel
        radar_path = data_root / radar_rel
        if not ann_path.exists() or not radar_path.exists():
            removed_entries.append(entry)
            removed_counts[split] += 1
            removed_reasons['missing'] += 1
            continue

        try:
            ann = load_json(ann_path)
        except Exception:
            removed_entries.append(entry)
            removed_counts[split] += 1
            removed_reasons['invalid'] += 1
            remove_files(ann_path, radar_path, dry_run)
            continue

        if not is_single_person(ann):
            removed_entries.append(entry)
            removed_counts[split] += 1
            removed_reasons['not_single'] += 1
            remove_files(ann_path, radar_path, dry_run)
            continue

        kept_entries.append(entry)

    if not dry_run:
        with metainfo_path.open('w') as f:
            json.dump(kept_entries, f, indent=2, ensure_ascii=False)

    return removed_counts, removed_reasons, len(kept_entries), len(removed_entries)


def filter_by_dirs(data_root: Path, dry_run: bool):
    metainfo_path = data_root / 'metainfo.json'
    metainfo = None
    if metainfo_path.exists():
        metainfo = load_json(metainfo_path)
        if not isinstance(metainfo, list):
            raise TypeError('metainfo.json must be a list')

    removed_counts = {'train': 0, 'val': 0}
    removed_ann_rels = set()

    for split in ('train', 'val'):
        ann_dir = data_root / split / 'annotations'
        radar_dir = data_root / split / 'radar_points'
        if not ann_dir.exists():
            continue

        ann_files = sorted(ann_dir.glob('*.json'))
        for ann_path in tqdm(ann_files, desc=f'Filtering {split}'):
            radar_path = radar_dir / ann_path.name
            try:
                ann = load_json(ann_path)
            except Exception:
                removed_counts[split] += 1
                removed_ann_rels.add(str(ann_path.relative_to(data_root)))
                remove_files(ann_path, radar_path, dry_run)
                continue

            if not is_single_person(ann):
                removed_counts[split] += 1
                removed_ann_rels.add(str(ann_path.relative_to(data_root)))
                remove_files(ann_path, radar_path, dry_run)

    removed_entries = 0
    kept_entries = 0
    if metainfo is not None:
        kept = []
        for entry in metainfo:
            ann_rel = entry.get('annotation_file')
            radar_rel = entry.get('radar_file')
            if not ann_rel or not radar_rel:
                removed_entries += 1
                continue
            ann_path = data_root / ann_rel
            radar_path = data_root / radar_rel
            if (ann_rel in removed_ann_rels) or (not ann_path.exists()) or (not radar_path.exists()):
                removed_entries += 1
                continue
            kept.append(entry)
        kept_entries = len(kept)
        if not dry_run:
            with metainfo_path.open('w') as f:
                json.dump(kept, f, indent=2, ensure_ascii=False)

    return removed_counts, len(kept_entries), removed_entries, metainfo is not None


def main():
    args = get_args()
    data_root = Path(args.data_root)

    if args.source == 'metainfo':
        removed_counts, removed_reasons, kept, removed = filter_by_metainfo(
            data_root, args.dry_run)
        print('\nRemoved samples:')
        print(f"train: {removed_counts['train']}")
        print(f"val:   {removed_counts['val']}")
        if removed_counts['other'] > 0:
            print(f"other: {removed_counts['other']}")
        print('\nRemoved reasons:')
        for k, v in removed_reasons.items():
            print(f'{k}: {v}')
        print(f'\nKept entries: {kept}')
        print(f'Removed entries: {removed}')
    else:
        removed_counts, kept, removed, has_metainfo = filter_by_dirs(
            data_root, args.dry_run)
        print('\nRemoved samples (dirs scan):')
        print(f"train: {removed_counts['train']}")
        print(f"val:   {removed_counts['val']}")
        if has_metainfo:
            print(f'\nKept entries: {kept}')
            print(f'Removed entries from metainfo: {removed}')
        else:
            print('\nmetainfo.json not found; no metainfo update performed.')


if __name__ == '__main__':
    main()
