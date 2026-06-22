import os
import shutil
import json
import glob
import re
from tqdm import tqdm

data_root = "unfiltered_outdoor/"
target_data_root = "data_outdoor/"
group_size = 100
train_group_ratio = 0.8
max_point_radius = 6.0
# concatenate all data into one data path list
data_folders = sorted(os.listdir(data_root))


def build_existing_index(metainfo, root):
    index = {}
    cleaned = []
    for item in metainfo:
        if not isinstance(item, dict):
            continue
        radar_rel = item.get('radar_file')
        ann_rel = item.get('annotation_file')
        if not radar_rel or not ann_rel:
            continue
        radar_path = os.path.join(root, radar_rel)
        ann_path = os.path.join(root, ann_rel)
        if not (os.path.exists(radar_path) and os.path.exists(ann_path)):
            continue
        source_folder = item.get('source_folder', item.get('origin_dir'))
        key = (item.get('data_name'), source_folder)
        index[key] = len(cleaned)
        cleaned.append(item)
    return cleaned, index


def extract_timestamp(path):
    """Extract sortable timestamp key from filename; fallback to stem string."""
    stem = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r'\d+', stem)
    if nums:
        return tuple(int(x) for x in nums)
    return (stem,)


def is_valid_annotation(objs):
    if not isinstance(objs, list):
        return False
    for obj in objs:
        bbox_3d = obj.get('bbox_3d')
        if bbox_3d is None:
            return False
        if bbox_3d and (bbox_3d[3] > 1.9 or bbox_3d[5] > 1.9):
            return False
        keypoints_3d = obj.get('keypoints_3d')
        if not isinstance(keypoints_3d, list):
            return False
        for keypoint in keypoints_3d:
            if not isinstance(keypoint, list) or len(keypoint) < 3:
                return False
            if keypoint[0] is None or keypoint[1] is None or keypoint[2] is None:
                return False
            radius = (
                keypoint[0] ** 2 +
                keypoint[1] ** 2 +
                keypoint[2] ** 2
            ) ** 0.5
            if radius > max_point_radius:
                return False
    return True


os.makedirs(f'{target_data_root}/train/radar_points', exist_ok=True)
os.makedirs(f'{target_data_root}/train/annotations', exist_ok=True)
os.makedirs(f'{target_data_root}/val/radar_points', exist_ok=True)
os.makedirs(f'{target_data_root}/val/annotations', exist_ok=True)


metainfo_path = os.path.join(target_data_root, 'metainfo.json')
existing_metainfo = []
if os.path.exists(metainfo_path):
    with open(metainfo_path, 'r') as f:
        try:
            content = json.load(f)
            if isinstance(content, list):
                existing_metainfo = content
        except json.JSONDecodeError:
            existing_metainfo = []
existing_metainfo, existing_index = build_existing_index(existing_metainfo, target_data_root)


for folder in data_folders:
    folder_radar = glob.glob(os.path.join(data_root, folder, 'radar_points', '*'))
    folder_gt = glob.glob(os.path.join(data_root, folder, 'annotations', '*'))
    if len(folder_radar) != len(folder_gt):
        raise ValueError(
            f'Mismatched radar/annotation count in {folder}: '
            f'{len(folder_radar)} vs {len(folder_gt)}')
    radar_data_list = sorted(folder_radar, key=extract_timestamp)
    gt_data_list = sorted(folder_gt, key=extract_timestamp)
    radar_source_dirs = [folder] * len(radar_data_list)

    data_num = len(radar_data_list)

    metadata_entries = []
    for i in tqdm(range(data_num)):
        group_id = i // group_size
        group_start = group_id * group_size
        group_end = min(group_start + group_size, data_num)
        group_len = group_end - group_start
        idx_in_group = i - group_start
        # If the final group is shorter than group_size, keep it all in train.
        if group_len < group_size:
            split = 'train'
        else:
            train_len = int(group_len * train_group_ratio)
            split = 'train' if idx_in_group < train_len else 'val'
        radar_name = os.path.basename(radar_data_list[i])
        ann_name = os.path.basename(gt_data_list[i])
        radar_dest = os.path.join(
            target_data_root, split, 'radar_points', radar_name)
        ann_dest = os.path.join(
            target_data_root, split, 'annotations', ann_name)

        with open(gt_data_list[i], "r") as f:
            objs = json.load(f)

        if not is_valid_annotation(objs):
            continue

        shutil.copyfile(radar_data_list[i], radar_dest)
        shutil.copyfile(gt_data_list[i], ann_dest)

        entry = dict(
            data_name=os.path.splitext(radar_name)[0],
            radar_file=os.path.relpath(radar_dest, target_data_root),
            annotation_file=os.path.relpath(ann_dest, target_data_root),
            split=split,
            source_folder=f"{radar_source_dirs[i]}_{group_id}")
        metadata_entries.append(entry)

    for entry in metadata_entries:
        key = (entry['data_name'], entry.get('source_folder'))
        if key in existing_index:
            existing_metainfo[existing_index[key]] = entry
        else:
            existing_index[key] = len(existing_metainfo)
            existing_metainfo.append(entry)

with open(metainfo_path, 'w') as f:
    json.dump(existing_metainfo, f, indent=2)
