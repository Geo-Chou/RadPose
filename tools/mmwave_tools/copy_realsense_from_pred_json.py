#!/usr/bin/env python3
"""Copy RealSense images by mapping prediction sample names through metainfo.

Mapping chain:
pred json file (sample_name) -> metainfo.data_name -> metainfo.annotation_file
stem -> dataproc/<scene>/<session>/realsense/<annotation_stem>.png
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Top-level scene folder presets under dataproc root.
SCENE_PRESETS: Dict[str, List[str]] = {
    "all_main": ["G", "J", "L", "P", "Q"],
    "unseen": ["Unseen"],
    "foam": ["foam"],
    "paper": ["paper"],
    "openarea": ["openarea"],
    "dim": ["dim"]

}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy realsense images for prediction json files, using metainfo.json "
            "to map radar sample names to annotation/realsense image ids."
        ))
    parser.add_argument(
        "--pred-json-roots",
        type=Path,
        nargs="+",
        required=True,
        help="One or more folders that contain prediction json files.")
    parser.add_argument(
        "--metainfo",
        type=Path,
        required=True,
        help="Path to metainfo.json used by the dataset.")
    parser.add_argument(
        "--dataproc-root",
        type=Path,
        default=Path("/workspace/dataproc"),
        help="Root of dataproc folders.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination root for copied images.")
    parser.add_argument(
        "--scene-preset",
        type=str,
        default=None,
        help=f"Scene preset key. Available: {list(SCENE_PRESETS.keys())}")
    parser.add_argument(
        "--scene-folders",
        nargs="+",
        default=None,
        help="Explicit scene folders under dataproc root. Overrides --scene-preset.")
    parser.add_argument(
        "--realsense-dirname",
        type=str,
        default="realsense",
        help="Realsense directory name under dataproc scene/session.")
    parser.add_argument(
        "--img-exts",
        nargs="+",
        default=["png", "jpg"],
        help="Image extensions to copy (without dot).")
    parser.add_argument(
        "--copy-depth",
        action="store_true",
        help="Also copy depth npy with the same stem.")
    parser.add_argument(
        "--keep-subdirs",
        action="store_true",
        help="Keep scene/session subdirectories under output root.")
    parser.add_argument(
        "--on-conflict",
        type=str,
        choices=["overwrite", "skip", "rename"],
        default="rename",
        help=(
            "When output filename already exists: overwrite, skip, or rename. "
            "Default is rename."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print operations without copying.")
    return parser.parse_args()


def choose_scene_folders(args: argparse.Namespace) -> Optional[List[str]]:
    if args.scene_folders:
        return list(args.scene_folders)
    if args.scene_preset:
        if args.scene_preset not in SCENE_PRESETS:
            raise ValueError(
                f"Unknown scene preset '{args.scene_preset}'. "
                f"Available: {list(SCENE_PRESETS.keys())}")
        return SCENE_PRESETS[args.scene_preset]
    return None


def parse_sample_name_from_json(json_path: Path) -> str:
    try:
        with json_path.open("r") as f:
            content = json.load(f)
        if isinstance(content, dict):
            sample_name = content.get("sample_name")
            if isinstance(sample_name, str) and sample_name:
                return sample_name
    except Exception:
        pass
    return json_path.stem


def collect_target_samples(pred_json_roots: Sequence[Path]) -> List[str]:
    samples: List[str] = []
    seen = set()
    for root in pred_json_roots:
        if root.is_file() and root.suffix == ".json":
            sample = parse_sample_name_from_json(root)
            if sample not in seen:
                samples.append(sample)
                seen.add(sample)
            continue
        if not root.exists():
            print(f"[WARN] pred root missing: {root}")
            continue
        for path in sorted(root.glob("*.json")):
            sample = parse_sample_name_from_json(path)
            if sample not in seen:
                samples.append(sample)
                seen.add(sample)
    return samples


def load_metainfo_index(
    metainfo_path: Path,
) -> Dict[str, List[dict]]:
    with metainfo_path.open("r") as f:
        content = json.load(f)
    if not isinstance(content, list):
        raise TypeError(f"metainfo must be a list: {metainfo_path}")

    by_name: Dict[str, List[dict]] = defaultdict(list)
    for item in content:
        if not isinstance(item, dict):
            continue
        data_name = item.get("data_name")
        if not data_name:
            continue
        by_name[str(data_name)].append(item)
    return by_name


def canonical_session_id(source_folder: Optional[str]) -> Optional[str]:
    if not source_folder:
        return None
    # source_folder often looks like "<session_id>_<group_id>".
    m = re.match(r"^(\d+)_\d+$", source_folder)
    if m:
        return m.group(1)
    return source_folder


def build_session_index(
    dataproc_root: Path,
    realsense_dirname: str,
    scene_folders: Optional[Sequence[str]],
) -> Dict[str, List[Tuple[str, Path]]]:
    index: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)
    if scene_folders is None:
        scenes = sorted([p.name for p in dataproc_root.iterdir() if p.is_dir()])
    else:
        scenes = list(scene_folders)

    for scene in scenes:
        scene_dir = dataproc_root / scene
        if not scene_dir.is_dir():
            continue
        for session_dir in sorted(scene_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            if not (session_dir / realsense_dirname).is_dir():
                continue
            index[session_dir.name].append((scene, session_dir))
    return index


def pick_session_dir(
    candidates: List[Tuple[str, Path]],
    preferred_scenes: Optional[Sequence[str]],
) -> Optional[Tuple[str, Path]]:
    if not candidates:
        return None
    if preferred_scenes:
        rank = {name: i for i, name in enumerate(preferred_scenes)}
        candidates = sorted(
            candidates,
            key=lambda x: (rank.get(x[0], 10**9), str(x[1])),
        )
    return candidates[0]


def find_realsense_stem(entry: dict, sample_name: str) -> List[str]:
    stems: List[str] = []
    ann_rel = entry.get("annotation_file")
    if isinstance(ann_rel, str) and ann_rel:
        stems.append(Path(ann_rel).stem)
    if sample_name not in stems:
        stems.append(sample_name)
    return stems


def ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def resolve_dst_path(
    dst_dir: Path,
    filename: str,
    on_conflict: str,
    scene: str,
    session_id: str,
) -> Optional[Path]:
    base = dst_dir / filename
    if not base.exists() or on_conflict == "overwrite":
        return base
    if on_conflict == "skip":
        return None

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = dst_dir / f"{stem}__{scene}_{session_id}{suffix}"
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        candidate = dst_dir / f"{stem}__{scene}_{session_id}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def main() -> None:
    args = parse_args()
    scene_folders = choose_scene_folders(args)
    image_exts = [ext.lower().lstrip(".") for ext in args.img_exts]

    samples = collect_target_samples(args.pred_json_roots)
    print(f"[INFO] collected samples from pred json: {len(samples)}")

    by_name = load_metainfo_index(args.metainfo)
    session_index = build_session_index(
        dataproc_root=args.dataproc_root,
        realsense_dirname=args.realsense_dirname,
        scene_folders=scene_folders,
    )

    ensure_dir(args.output_root, args.dry_run)

    copied = 0
    missing_in_metainfo = 0
    missing_session = 0
    missing_image = 0

    for sample_name in samples:
        entries = by_name.get(sample_name, [])
        if not entries:
            print(f"[MISS] sample not in metainfo: {sample_name}")
            missing_in_metainfo += 1
            continue
        entry = entries[0]

        source_folder = entry.get("source_folder")
        session_id = canonical_session_id(source_folder)
        if not session_id:
            print(f"[MISS] source_folder missing for sample: {sample_name}")
            missing_session += 1
            continue

        session_candidates = session_index.get(session_id, [])
        picked = pick_session_dir(session_candidates, scene_folders)
        if picked is None:
            print(f"[MISS] session dir not found for session_id={session_id} sample={sample_name}")
            missing_session += 1
            continue
        scene, session_dir = picked
        realsense_dir = session_dir / args.realsense_dirname

        stems = find_realsense_stem(entry, sample_name)
        found_image = False
        for stem in stems:
            copied_this_stem = False
            for ext in image_exts:
                src = realsense_dir / f"{stem}.{ext}"
                if not src.exists():
                    continue

                if args.keep_subdirs:
                    dst_dir = args.output_root / scene / session_id
                    dst_name = src.name
                else:
                    dst_dir = args.output_root
                    dst_name = src.name
                ensure_dir(dst_dir, args.dry_run)
                dst = resolve_dst_path(
                    dst_dir=dst_dir,
                    filename=dst_name,
                    on_conflict=args.on_conflict,
                    scene=scene,
                    session_id=session_id,
                )
                if dst is None:
                    print(f"[SKIP] conflict: {dst_dir / dst_name}")
                    continue
                print(f"[COPY] {src} -> {dst}")
                if not args.dry_run:
                    shutil.copy2(src, dst)
                copied += 1
                copied_this_stem = True
                found_image = True

            if copied_this_stem and args.copy_depth:
                depth_src = realsense_dir / f"{stem}.npy"
                if depth_src.exists():
                    if args.keep_subdirs:
                        depth_dst_dir = args.output_root / scene / session_id
                        depth_dst_name = depth_src.name
                    else:
                        depth_dst_dir = args.output_root
                        depth_dst_name = depth_src.name
                    ensure_dir(depth_dst_dir, args.dry_run)
                    depth_dst = resolve_dst_path(
                        dst_dir=depth_dst_dir,
                        filename=depth_dst_name,
                        on_conflict=args.on_conflict,
                        scene=scene,
                        session_id=session_id,
                    )
                    if depth_dst is None:
                        print(f"[SKIP] conflict: {depth_dst_dir / depth_dst_name}")
                        continue
                    print(f"[COPY] {depth_src} -> {depth_dst}")
                    if not args.dry_run:
                        shutil.copy2(depth_src, depth_dst)
                    copied += 1

            if found_image:
                break

        if not found_image:
            print(
                f"[MISS] image not found for sample={sample_name} "
                f"(candidate_stems={stems}, realsense_dir={realsense_dir})")
            missing_image += 1

    print("\nDone.")
    print(f"samples_total: {len(samples)}")
    print(f"copied_files: {copied}")
    print(f"missing_in_metainfo: {missing_in_metainfo}")
    print(f"missing_session_dir: {missing_session}")
    print(f"missing_realsense_image: {missing_image}")


if __name__ == "__main__":
    main()
