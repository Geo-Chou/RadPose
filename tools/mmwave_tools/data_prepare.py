from mmpose.apis import MMPoseInferencer
import os
import argparse
from tqdm import tqdm
import numpy as np
import pyrealsense2 as rs
import json
import sys
import shutil
sys.modules['numpy._core'] = np
import pickle

data_path = '/workspace/dataproc/data/'
conf_thresh = 0.2
output_path = '/workspace/ikmmpose/data_unfiltered/'

def get_args():
    parser = argparse.ArgumentParser(description="data preparation selection")

    parser.add_argument('--data_dirs', type=str, default=None, help='Specify single data directory to process, ' \
    'else will process all directories in data_path')
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

def align_and_clean(radar_dir, gt_dir, out_radar_dir=None, out_gt_dir=None, max_diff=50):
    """
    params:
        radar_dir (str): radar 
        gt_dir (str): groundtruth 
        out_radar_dir (str)
        out_gt_dir (str)
        max_diff (int) ms
    """
    radar_files = sorted(os.listdir(radar_dir))
    gt_files = sorted(os.listdir(gt_dir))

    radar_ts = np.array([int(os.path.splitext(f)[0]) for f in radar_files])
    gt_ts = np.array([int(os.path.splitext(f)[0]) for f in gt_files])

    matched_pairs = []
    used_gt_idx = set()

    for i, rt in enumerate(radar_ts):
        diff = np.abs(gt_ts - rt)
        j = np.argmin(diff)
        if diff[j] <= max_diff:
            matched_pairs.append((radar_files[i], gt_files[j]))
            used_gt_idx.add(j)

    print(f"Matched {len(matched_pairs)} pairs within ±{max_diff} ms")

    def safe_copy(src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
    os.makedirs(out_radar_dir, exist_ok=True)
    os.makedirs(out_gt_dir, exist_ok=True)
    if out_radar_dir and out_gt_dir:
        for rf, gf in tqdm(matched_pairs):
            safe_copy(os.path.join(radar_dir, rf), os.path.join(out_radar_dir, rf))
            safe_copy(os.path.join(gt_dir, gf), os.path.join(out_gt_dir, gf))

    return matched_pairs

if __name__ == '__main__':
    args = get_args()

    ## process 2D pose estimation
    inferencer = MMPoseInferencer(pose2d = 'human', show_progress=True)
    
    if args.data_dirs is not None:
        data_dirs = [args.data_dirs]
    else:
        data_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    for dir in data_dirs:
        
        cur_data_path = os.path.join(data_path, dir, 'realsense')
        print(f'Processing data directory: {cur_data_path}')
        imgs = [os.path.join(cur_data_path, f) for f in os.listdir(cur_data_path) if f.endswith('.png')]
        print(f'Found {len(imgs)} images to process.')
        result_generator = inferencer(imgs, pred_out_dir=os.path.join(data_path, dir, 'predictions'), show=False)
        results = [result for result in result_generator]

        intr_data = np.load(os.path.join(data_path,dir,"intrinsics.npz"))
        # with open(os.path.join(data_path,dir,"intrinsics.npy"), "rb") as f:
        #      intr_data = pickle.load(f)
        intr = rs.intrinsics()
        intr.width = intr_data["width"]
        intr.height = intr_data["height"]
        intr.ppx = intr_data["ppx"]
        intr.ppy = intr_data["ppy"]
        intr.fx = intr_data["fx"]
        intr.fy = intr_data["fy"]
        intr.model = rs.distortion(int(intr_data["model"]))
        intr.coeffs = intr_data["coeffs"]

        ## process depth to 3D points
        print('Processing depth maps to 3D points...')
        os.makedirs(os.path.join(data_path, dir, 'annotations'), exist_ok=True)
        for img in tqdm(imgs):
            img_name = os.path.basename(img)
            img_idx = os.path.splitext(img_name)[0]
            depth_map = np.load(os.path.join(cur_data_path, f'{img_idx}.npy'))
            pred_data = json.load(open(os.path.join(data_path, dir, 'predictions', f'{img_idx}.json')))
            result_data = list()
            for pred in pred_data:
                keypoints = np.array(pred["keypoints"])
                score = np.array(pred["keypoint_scores"])
                result_dict = dict()
                keypoints_3d = []
                for i, (u, v) in enumerate(keypoints):
                    if u < 0 or v < 0 or u >= depth_map.shape[1] or v >= depth_map.shape[0]:
                        keypoints_3d.append([np.nan, np.nan, np.nan, 0.01])
                        continue
                    Z = depth_map[int(v), int(u)]
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [u, v], Z)
                    X, Y, Z = X / 1000.0, Y / 1000.0, Z / 1000.0  # convert mm to meters
                    c = score[i]
                    keypoints_3d.append([X, Y, Z, c])
                result_dict['keypoints_3d'] = keypoints_3d
                keypoints_3d = np.array(keypoints_3d)
                mask_conf = keypoints_3d[:, 3] > conf_thresh
                mask_valid = ~np.isnan(keypoints_3d[:, :3]).any(axis=1)
                valid_mask = mask_conf & mask_valid
                pts = keypoints_3d[valid_mask, :3]
                result_dict['keypoints_3d'] = [
                    [None if (isinstance(x, float) and np.isnan(x)) else x for x in row]
                    for row in result_dict['keypoints_3d']
                ]
                if pts.shape[0] < 2:
                    result_dict['bbox_3d'] = None
                    
                else:
                    min_xyz = np.min(pts, axis=0)
                    max_xyz = np.max(pts, axis=0)
                    mins = np.nanmin(pts, axis=0)
                    maxs = np.nanmax(pts, axis=0)
                    margin = 0.8
                    center = (mins + maxs) / 2.0
                    size = (maxs - mins) * (1 + margin)  
                    bbox3d = np.concatenate([center, size])
                    result_dict['bbox_3d'] = bbox3d.tolist()
                result_data.append(result_dict)
            with open(os.path.join(data_path, dir, 'annotations', f'{img_idx}.json'), 'w') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f'Finished processing data directory: {cur_data_path}')
        print('---------------------------------------')
        print('Coarse data alignment is processing')
        align_and_clean(
            radar_dir=os.path.join(data_path, dir, 'radar_points'),
            gt_dir=os.path.join(data_path, dir, 'annotations'),
            out_radar_dir=os.path.join(output_path, dir, 'radar_points'),
            out_gt_dir=os.path.join(output_path, dir, 'annotations'),
            max_diff=5e7 # 50ms
        )