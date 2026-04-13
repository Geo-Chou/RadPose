#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

from mmengine.config import Config


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _get_radar_paths(dataset, index: int):
    if (not dataset.index_to_folder_pos or
            len(dataset.index_to_folder_pos) != len(dataset.data_list)):
        dataset._build_source_folder_cache()
    folder, folder_pos = dataset.index_to_folder_pos[index]
    folder_indices = dataset.folder_to_indices.get(folder, [index])
    max_frame_offset = int(dataset.num_frames / 2)
    frame_step = dataset.frame_interval + 1
    paths = []
    for i in range(dataset.num_frames):
        relative_pos = (i - max_frame_offset) * frame_step
        target_pos = folder_pos + relative_pos
        if target_pos < 0:
            target_pos = 0
        elif target_pos >= len(folder_indices):
            target_pos = len(folder_indices) - 1
        curr_index = folder_indices[target_pos]
        curr_info = dataset.get_data_info(curr_index)
        paths.append(curr_info["radar_points_path"])
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan val dataset for unusually slow samples.")
    parser.add_argument("--config", required=True, help="Config path.")
    parser.add_argument("--threshold-sec", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--every", type=int, default=50,
                        help="Progress print interval.")
    args = parser.parse_args()

    _ensure_repo_on_path()
    import mmwave3d  # noqa: F401
    from mmwave3d.registry import DATASETS

    cfg = Config.fromfile(args.config)
    dataset_cfg = cfg.val_dataloader.dataset
    dataset = DATASETS.build(dataset_cfg)

    total = len(dataset)
    start = max(0, args.start)
    end = total if args.end < 0 else min(args.end, total)

    timings = []
    slow = []
    for idx in range(start, end):
        t0 = time.perf_counter()
        try:
            _ = dataset[idx]
            ok = True
        except Exception as exc:
            ok = False
            err = exc
        t1 = time.perf_counter()
        elapsed = t1 - t0
        timings.append(elapsed)
        if elapsed >= args.threshold_sec:
            paths = _get_radar_paths(dataset, idx)
            sample_name = Path(paths[len(paths) // 2]).stem if paths else "n/a"
            slow.append((elapsed, idx, sample_name, paths))
        if not ok:
            print(f"ERROR idx={idx} time={elapsed:.3f}s err={err}")
        if args.every > 0 and (idx - start + 1) % args.every == 0:
            print(f"PROGRESS {idx - start + 1}/{end - start}")

    slow.sort(key=lambda x: x[0], reverse=True)
    print(f"Scanned {end - start} samples. Threshold={args.threshold_sec:.3f}s")
    if slow:
        print(f"Top {min(args.top_k, len(slow))} slow samples:")
        for elapsed, idx, sample_name, paths in slow[:args.top_k]:
            print(f"SLOW idx={idx} time={elapsed:.3f}s sample={sample_name}")
            for p in paths:
                print(f"  {p}")
    else:
        print("No slow samples found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
