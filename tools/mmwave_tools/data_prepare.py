from mmpose.apis import MMPoseInferencer
import os
import argparse
from tqdm import tqdm
import numpy as np
import pyrealsense2 as rs
import json
import sys
import shutil
import tempfile
sys.modules['numpy._core'] = np

conf_thresh = 0.5

def get_args():
    parser = argparse.ArgumentParser(description="data preparation selection")

    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--output_path', type=str, default='unfiltered_outdoor/', 
                        help='Path to the output directory')
    parser.add_argument('--data_dirs', type=str, default=None, help='Specify single data directory to process, ' \
    'else will process all directories in data_path')
    parser.add_argument('--skip_processed', action='store_true', 
                        help='Skip directories that have already been processed')
    args = parser.parse_args()
    return args

def depth_to_points_rs(depth_map, intr):
    h, w = depth_map.shape
    points = []
    for y in range(h):
        for x in range(w):
            d = depth_map[y, x]
            if d > 0:
                X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [x, y], d)
                points.append([X, Y, Z])
    return np.array(points)

def load_realsense_intrinsics(data_dir):
    intr_data = np.load(os.path.join(data_dir, "intrinsics.npz"))
    intr = rs.intrinsics()
    intr.width = int(intr_data["width"])
    intr.height = int(intr_data["height"])
    intr.ppx = float(intr_data["ppx"])
    intr.ppy = float(intr_data["ppy"])
    intr.fx = float(intr_data["fx"])
    intr.fy = float(intr_data["fy"])
    intr.model = rs.distortion(int(intr_data["model"]))
    intr.coeffs = intr_data["coeffs"].tolist()
    return intr

def timestamp_from_name(filename):
    return int(os.path.splitext(os.path.basename(filename))[0])

def align_radar_realsense(radar_dir, realsense_dir, max_diff=50):
    """
    params:
        radar_dir (str): radar 
        realsense_dir (str): directory containing rgb images and depth maps
        max_diff: maximum allowed timestamp difference, in the same unit as filenames
    """
    radar_files = sorted(os.listdir(radar_dir))
    image_files = sorted(
        f for f in os.listdir(realsense_dir)
        if f.endswith('.png') and
        os.path.exists(os.path.join(realsense_dir, f'{os.path.splitext(f)[0]}.npy'))
    )

    radar_ts = np.array([timestamp_from_name(f) for f in radar_files])
    image_ts = np.array([timestamp_from_name(f) for f in image_files])

    matched_pairs = []
    used_image_idx = set()

    for i, rt in enumerate(radar_ts):
        if len(image_ts) == 0:
            break
        diff = np.abs(image_ts - rt)
        for used_idx in used_image_idx:
            diff[used_idx] = np.iinfo(diff.dtype).max
        j = int(np.argmin(diff))
        if diff[j] <= max_diff:
            image_file = image_files[j]
            depth_file = f'{os.path.splitext(image_file)[0]}.npy'
            matched_pairs.append(dict(
                radar_file=radar_files[i],
                image_file=image_file,
                depth_file=depth_file,
                radar_ts=int(rt),
                image_ts=int(image_ts[j]),
                diff=int(diff[j]),
            ))
            used_image_idx.add(j)

    print(f"Matched {len(matched_pairs)} radar/realsense pairs within ±{max_diff}")
    return matched_pairs

def load_predictions(pred_path):
    with open(pred_path, 'r') as f:
        pred_data = json.load(f)
    if isinstance(pred_data, dict):
        if 'predictions' in pred_data:
            pred_data = pred_data['predictions']
        elif 'instance_info' in pred_data:
            pred_data = pred_data['instance_info']
    if isinstance(pred_data, list) and len(pred_data) == 1 and isinstance(pred_data[0], list):
        pred_data = pred_data[0]
    return pred_data

def build_annotation(pred_data, depth_map, intr):
    result_data = list()
    for pred in pred_data:
        keypoints = np.array(pred["keypoints"])
        score = np.array(pred["keypoint_scores"])
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
            c = score[i]
            keypoints_3d.append([X, Y, Z, c])
            r = np.sqrt(X**2 + Y**2 + Z**2)
            az = np.arctan2(Z, X)
            el = np.arcsin(Y / r) if r > 0 else np.nan
            keypoints_sp.append([r, az, el, c])
        result_dict['keypoints_3d'] = keypoints_3d
        result_dict['keypoints_sp'] = keypoints_sp
        keypoints_3d = np.array(keypoints_3d)
        mask_conf = keypoints_3d[:, 3] > conf_thresh
        mask_valid = ~np.isnan(keypoints_3d[:, :3]).any(axis=1)
        valid_mask = mask_conf & mask_valid
        pts = keypoints_3d[valid_mask, :3]
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

def is_directory_processed(output_path, dir_name):
    """
    Check if a directory has already been processed.
    Returns True if the output directory exists and contains expected files.
    """
    output_dir = os.path.join(output_path, dir_name)
    annotations_dir = os.path.join(output_dir, 'annotations')
    radar_dir = os.path.join(output_dir, 'radar_points')
    
    if not os.path.exists(output_dir):
        return False
    
    # Check if both annotations and radar_points directories exist and contain files
    has_annotations = os.path.exists(annotations_dir) and len(os.listdir(annotations_dir)) > 0
    has_radar = os.path.exists(radar_dir) and len(os.listdir(radar_dir)) > 0
    
    return has_annotations and has_radar

if __name__ == '__main__':
    args = get_args()
    data_path = args.data_path
    output_path = args.output_path

    # process 2D pose estimation
    inferencer = MMPoseInferencer(pose2d = 'rtmpose-m_8xb64-210e_mpii-256x256', show_progress=True)
    
    if args.data_dirs is not None:
        data_dirs = [args.data_dirs]
    else:
        data_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    for dir in data_dirs:
        if args.skip_processed and is_directory_processed(output_path, dir):
            print(f'Skipping {dir}: Already processed')
            print('---------------------------------------')
            continue
        
        cur_data_path = os.path.join(data_path, dir)
        realsense_dir = os.path.join(cur_data_path, 'realsense')
        radar_dir = os.path.join(cur_data_path, 'radar_points')
        out_radar_dir = os.path.join(output_path, dir, 'radar_points')
        out_gt_dir = os.path.join(output_path, dir, 'annotations')

        print(f'Processing data directory: {cur_data_path}')
        print('---------------------------------------')
        print('Radar/realsense alignment is processing')

        matched_pairs = align_radar_realsense(
            radar_dir=radar_dir,
            realsense_dir=realsense_dir,
            max_diff=5e7 # 50ms if timestamps are in ns
        )
        if not matched_pairs:
            print(f'No matched radar/realsense pairs found in {dir}')
            print('---------------------------------------')
            continue

        intr = load_realsense_intrinsics(cur_data_path)
        os.makedirs(out_radar_dir, exist_ok=True)
        os.makedirs(out_gt_dir, exist_ok=True)

        imgs = [
            os.path.join(realsense_dir, pair['image_file'])
            for pair in matched_pairs
        ]
        print(f'Running MMPose on {len(imgs)} aligned images.')
        with tempfile.TemporaryDirectory() as pred_dir:
            result_generator = inferencer(imgs, pred_out_dir=pred_dir, show=False)
            for _ in result_generator:
                pass

            print('Processing matched depth maps to 3D annotations...')
            for pair in tqdm(matched_pairs):
                radar_src = os.path.join(radar_dir, pair['radar_file'])
                radar_dst = os.path.join(out_radar_dir, pair['radar_file'])
                shutil.copy(radar_src, radar_dst)

                image_idx = os.path.splitext(pair['image_file'])[0]
                radar_idx = os.path.splitext(pair['radar_file'])[0]
                depth_map = np.load(os.path.join(realsense_dir, pair['depth_file']))
                pred_data = load_predictions(os.path.join(pred_dir, f'{image_idx}.json'))
                result_data = build_annotation(pred_data, depth_map, intr)
                with open(os.path.join(out_gt_dir, f'{radar_idx}.json'), 'w') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f'Finished processing data directory: {cur_data_path}')
        print('---------------------------------------')
