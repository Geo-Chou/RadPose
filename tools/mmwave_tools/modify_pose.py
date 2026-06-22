from mmpose.apis import MMPoseInferencer
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
from tqdm import tqdm


DEFAULT_MODEL = 'rtmpose-m_8xb64-210e_mpii-256x256'
DEFAULT_DATAPROC_ROOT = 'data/raw'
DEFAULT_MMPOSE_ROOT = 'data'
DEFAULT_PRED_DIRNAME = 'predictions_mpii'
DEFAULT_CONF_THRESH = 0.5


def get_args():
    parser = argparse.ArgumentParser(
        description='Replace mmpose annotations with MPII keypoints from Realsense + depth')
    parser.add_argument('--dataproc-root', type=str, default=DEFAULT_DATAPROC_ROOT)
    parser.add_argument('--mmpose-root', type=str, default=DEFAULT_MMPOSE_ROOT)
    parser.add_argument('--pose2d', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--pred-dirname', type=str, default=DEFAULT_PRED_DIRNAME)
    parser.add_argument('--conf-thresh', type=float, default=DEFAULT_CONF_THRESH)
    parser.add_argument('--infer-batch-size', type=int, default=64)
    parser.add_argument('--use-existing-pred', action='store_true',
                        help='Skip inference if prediction json exists')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing annotation json files')
    return parser.parse_args()


def build_intrinsics(intr_data):
    intr = rs.intrinsics()
    intr.width = int(intr_data['width'])
    intr.height = int(intr_data['height'])
    intr.ppx = float(intr_data['ppx'])
    intr.ppy = float(intr_data['ppy'])
    intr.fx = float(intr_data['fx'])
    intr.fy = float(intr_data['fy'])
    intr.model = rs.distortion(int(intr_data['model']))
    intr.coeffs = intr_data['coeffs']
    return intr


def chunked(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def compute_annotations(pred_data, depth_map, intr, conf_thresh):
    result_data = []
    for pred in pred_data:
        keypoints = np.array(pred['keypoints'])
        score = np.array(pred['keypoint_scores'])
        result_dict = dict()
        keypoints_3d = []
        keypoints_sp = []
        for i, (u, v) in enumerate(keypoints):
            if u < 0 or v < 0 or u >= depth_map.shape[1] or v >= depth_map.shape[0]:
                keypoints_3d.append([np.nan, np.nan, np.nan, 0.01])
                keypoints_sp.append([np.nan, np.nan, np.nan, 0.01])
                continue
            Z = depth_map[int(v), int(u)]
            if Z <= 0:
                keypoints_3d.append([np.nan, np.nan, np.nan, 0.01])
                keypoints_sp.append([np.nan, np.nan, np.nan, 0.01])
                continue
            X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [u, v], Z)
            X, Y, Z = X / 1000.0, Y / 1000.0, Z / 1000.0
            c = float(score[i])
            keypoints_3d.append([X, Y, Z, c])
            r = np.sqrt(X**2 + Y**2 + Z**2)
            az = np.arctan2(Z, X)
            el = np.arcsin(Y / r)
            keypoints_sp.append([r, az, el, c])
        result_dict['keypoints_3d'] = keypoints_3d
        result_dict['keypoints_sp'] = keypoints_sp

        keypoints_3d_np = np.array(keypoints_3d)
        mask_conf = keypoints_3d_np[:, 3] > conf_thresh
        mask_valid = ~np.isnan(keypoints_3d_np[:, :3]).any(axis=1)
        valid_mask = mask_conf & mask_valid
        pts = keypoints_3d_np[valid_mask, :3]

        result_dict['keypoints_3d'] = [
            [None if (isinstance(x, float) and np.isnan(x)) else x for x in row]
            for row in result_dict['keypoints_3d']
        ]
        result_dict['keypoints_sp'] = [
            [None if (isinstance(x, float) and np.isnan(x)) else x for x in row]
            for row in result_dict['keypoints_sp']
        ]

        if pts.shape[0] < 2:
            result_dict['bbox_3d'] = None
        else:
            mins = np.nanmin(pts, axis=0)
            maxs = np.nanmax(pts, axis=0)
            margin = 0.8
            center = (mins + maxs) / 2.0
            size = (maxs - mins) * (1 + margin)
            bbox3d = np.concatenate([center, size])
            result_dict['bbox_3d'] = bbox3d.tolist()

        result_data.append(result_dict)

    return result_data


def main():
    args = get_args()

    mmpose_root = Path(args.mmpose_root)
    dataproc_root = Path(args.dataproc_root)

    metainfo_path = mmpose_root / 'metainfo.json'
    if not metainfo_path.exists():
        raise FileNotFoundError(f'metainfo.json not found at {metainfo_path}')

    with metainfo_path.open('r') as f:
        metainfo = json.load(f)

    entries_by_source = {}
    for entry in metainfo:
        source_folder = entry.get('source_folder')
        ann_rel = entry.get('annotation_file')
        if not source_folder or not ann_rel:
            continue
        ann_abs = mmpose_root / ann_rel
        img_id = Path(ann_rel).stem
        split = Path(ann_rel).parts[0]
        entries_by_source.setdefault(source_folder, []).append(
            dict(img_id=img_id, ann_path=ann_abs, split=split)
        )

    inferencer = MMPoseInferencer(pose2d=args.pose2d, show_progress=True)

    for source_folder, items in entries_by_source.items():
        realsense_dir = dataproc_root / source_folder / 'realsense'
        intr_path = dataproc_root / source_folder / 'intrinsics.npz'
        pred_dir = dataproc_root / source_folder / args.pred_dirname

        if not realsense_dir.exists():
            print(f'[WARN] missing realsense dir: {realsense_dir}, skip')
            continue
        if not intr_path.exists():
            print(f'[WARN] missing intrinsics: {intr_path}, skip')
            continue

        intr_data = np.load(intr_path)
        intr = build_intrinsics(intr_data)

        img_paths = []
        for item in items:
            img_path = realsense_dir / f"{item['img_id']}.png"
            if img_path.exists():
                img_paths.append(img_path)
            else:
                print(f"[WARN] image missing: {img_path}")

        if not img_paths:
            continue

        os.makedirs(pred_dir, exist_ok=True)

        if not args.use_existing_pred:
            for batch in chunked(img_paths, args.infer_batch_size):
                _ = list(inferencer(
                    [str(p) for p in batch],
                    pred_out_dir=str(pred_dir),
                    show=False
                ))
        else:
            for img_path in img_paths:
                pred_path = pred_dir / f"{img_path.stem}.json"
                if not pred_path.exists():
                    print(f"[WARN] missing pred json: {pred_path}, will run infer")
                    _ = list(inferencer(
                        [str(img_path)],
                        pred_out_dir=str(pred_dir),
                        show=False
                    ))

        for item in tqdm(items, desc=f'Writing {source_folder}'):
            ann_path = item['ann_path']
            if ann_path.exists() and not args.force:
                print(f"[SKIP] {ann_path} exists (use --force to overwrite)")
                continue

            depth_path = realsense_dir / f"{item['img_id']}.npy"
            pred_path = pred_dir / f"{item['img_id']}.json"

            if not depth_path.exists():
                print(f"[WARN] missing depth: {depth_path}")
                continue
            if not pred_path.exists():
                print(f"[WARN] missing prediction: {pred_path}")
                continue

            depth_map = np.load(depth_path)
            with pred_path.open('r') as f:
                pred_data = json.load(f)

            result_data = compute_annotations(pred_data, depth_map, intr, args.conf_thresh)

            ann_path.parent.mkdir(parents=True, exist_ok=True)
            with ann_path.open('w') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
