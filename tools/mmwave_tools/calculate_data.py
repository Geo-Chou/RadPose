import os
import shutil
import json
import glob
from tqdm import tqdm

data_root = "/workspace/ikmmpose/data_unfiltered/"
target_data_root = "/workspace/ikmmpose/data/mmwave/"
sets = ['train', 'val']
# concatenate all data into one data path list
data_folders = os.listdir(data_root)
radar_data_list = sorted([
        file
        for folder in data_folders
        for file in glob.glob(os.path.join(data_root, folder, 'radar_points', '*'))
    ])
gt_data_list = sorted([
        file
        for folder in data_folders
        for file in glob.glob(os.path.join(data_root, folder, 'annotations', '*'))
    ])

data_num  = len(radar_data_list)
train_max_num = int(data_num * 0.8)
# p = 0.8

os.makedirs(f'{target_data_root}/train/radar_points', exist_ok=True)
os.makedirs(f'{target_data_root}/train/annotations', exist_ok=True)
os.makedirs(f'{target_data_root}/val/radar_points', exist_ok=True)
os.makedirs(f'{target_data_root}/val/annotations', exist_ok=True)


for i in tqdm(range(data_num)):
    if i < train_max_num:
        
        shutil.copy(radar_data_list[i], f'{target_data_root}/train/radar_points/{os.path.basename(radar_data_list[i])}')
        shutil.copy(gt_data_list[i], f'{target_data_root}/train/annotations/{os.path.basename(gt_data_list[i])}')
    else:
        
        shutil.copy(radar_data_list[i], f'{target_data_root}/val/radar_points/{os.path.basename(radar_data_list[i])}')
        shutil.copy(gt_data_list[i], f'{target_data_root}/val/annotations/{os.path.basename(gt_data_list[i])}')
for split in sets:
    radar_data_list = sorted(os.listdir(os.path.join(target_data_root, split, 'radar_points')))
    gt_data_list = sorted(os.listdir(os.path.join(target_data_root, split, 'annotations')))

    for i, data in tqdm(enumerate(gt_data_list)):
        radar_path = os.path.join(target_data_root, split, 'radar_points', radar_data_list[i])
        gt_path = os.path.join(target_data_root, split, 'annotations', data)

        with open(gt_path, "r") as f:
            objs = json.load(f)

        valid_objs = []
        break_flag = False
        for obj in objs:
            if obj['bbox_3d'] == None:
                os.remove(radar_path)
                os.remove(gt_path)
                break
            elif obj['bbox_3d'] and (obj['bbox_3d'][3]>1.9 or obj['bbox_3d'][5]>1.9):
                os.remove(radar_path)
                os.remove(gt_path)
                break
            for keypoint in obj['keypoints_3d']:
                if keypoint[0]==None or keypoint[1]==None or keypoint[2]==None:
                    os.remove(radar_path)
                    os.remove(gt_path)
                    break_flag = True
                    break
            if break_flag:
                break_flag = False
                break

