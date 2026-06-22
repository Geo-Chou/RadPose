from mmengine.fileio import join_path, load, list_dir_or_file
from mmengine.logging import print_log
from mmengine.utils import is_abs
from mmengine.dataset import BaseDataset

import numpy as np
from pathlib import Path
from typing import Callable, List, Optional, Union, Dict, Tuple
from os import path as osp
import os
import copy

from mmwave3d.registry import DATASETS

@DATASETS.register_module()
class MMWaveDataset(BaseDataset):
    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 data_prefix: dict = dict(pts='velodyne', img=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 load_eval_anns: bool = True,
                 backend_args: Optional[dict] = None,
                 num_frames: int = 3,
                 frame_interval: int = 0,
                 metainfo_file: str = 'metainfo.json',
                 source_folders: Optional[Union[str, List[str]]] = None,
                 **kwargs)->None:
        self.num_frames = num_frames
        self.frame_interval = max(frame_interval, 0)
        self.metainfo_file = metainfo_file
        if source_folders is None:
            self.source_folders = None
        elif isinstance(source_folders, str):
            self.source_folders = {source_folders}
        else:
            self.source_folders = set(source_folders)
        self.backend_args = backend_args
        self.load_eval_anns = load_eval_anns
        self.folder_to_indices = dict()
        self.index_to_folder_pos: List[Tuple[str, int]] = []
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)
        self._build_source_folder_cache()
    
    def load_data_list(self) -> List[dict]:
        raw_data_list: List[dict] = []
        split = 'val' if self.test_mode else 'train'
        metainfo_path = self.metainfo_file
        if not is_abs(metainfo_path):
            metainfo_path = osp.join(self.data_root, metainfo_path)
        metainfo_entries = load(metainfo_path)
        if not isinstance(metainfo_entries, list):
            raise TypeError('metainfo file must provide a list of entries.')
        folder_counters: Dict[str, int] = dict()
        for entry in metainfo_entries:
            if entry.get('split') != split:
                continue
            source_folder = entry.get('source_folder')
            if self.source_folders is not None and \
                    source_folder not in self.source_folders:
                continue
            radar_rel = entry.get('radar_file')
            ann_rel = entry.get('annotation_file')
            if radar_rel is None or ann_rel is None:
                continue
            radar_path = osp.join(self.data_root, radar_rel)
            ann_path = osp.join(self.data_root, ann_rel)
            if not osp.exists(radar_path) or not osp.exists(ann_path):
                continue
            ann = load(ann_path)
            frame_order = folder_counters.get(source_folder, 0)
            folder_counters[source_folder] = frame_order + 1
            raw_data_list.append(
                dict(
                    radar_points_path=radar_path,
                    instances=[{
                        'keypoints_3d': obj['keypoints_sp']
                    } for obj in ann],
                    source_folder=source_folder,
                    frame_order=frame_order))
        if len(raw_data_list) == 0:
            raise ValueError(f'No valid annotation file found in '
                             f"{metainfo_path}")

        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For image tasks, `data_info` should information if single
                data_list.append(data_info)
            elif isinstance(data_info, list):
                # For video tasks, `data_info` could contain image
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')

        return data_list
    

    def parse_data_info(self, info: dict) -> dict:
        if 'radar_points_path' not in info:
            info['radar_points_path'] = osp.join(
                self.data_root, 'val' if self.test_mode else 'train',
                info['radar_path'])
        if 'frame_order' not in info:
            info['frame_order'] = 0

        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_ann_info(info)

        return info
    
    def parse_ann_info(self, info):
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
        """
        name_mapping = {
            'keypoints_3d': 'gt_keypoints_3d'
        }
        instances = info['instances']
        # empty gt
        if len(instances) == 0:
            return None
        else:
            keys = list(instances[0].keys())
            ann_info = dict()
            for ann_name in keys:
                temp_anns = [item[ann_name] for item in instances]
                # map the original dataset label to training label
                if ann_name in name_mapping:
                    mapped_ann_name = name_mapping[ann_name]
                else:
                    mapped_ann_name = ann_name

                if ann_name in name_mapping:
                    temp_anns = np.array(temp_anns).astype(np.float32)
                else:
                    temp_anns = np.array(temp_anns)

                ann_info[mapped_ann_name] = temp_anns
            ann_info['instances'] = info['instances']

        # return ann_info
        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info['gt_keypoints_3d'] = np.zeros((0, 17), dtype=np.float32)

        return ann_info
    
    def prepare_data(self, index: int) -> Union[dict, None]:
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        ori_input_dict = self.get_data_info(index)

        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(ori_input_dict)
        real_path_list = []
        ann_info_frames = []
        eval_ann_info_frames = []
        if (not self.index_to_folder_pos or
                len(self.index_to_folder_pos) != len(self.data_list)):
            self._build_source_folder_cache()
        folder, folder_pos = self.index_to_folder_pos[index]
        folder_indices = self.folder_to_indices.get(folder, [index])
        max_frame_offset = int(self.num_frames / 2)
        frame_step = self.frame_interval + 1
        for i in range(self.num_frames):
            relative_pos = (i - max_frame_offset) * frame_step
            target_pos = folder_pos + relative_pos
            if target_pos < 0:
                target_pos = 0
            elif target_pos >= len(folder_indices):
                target_pos = len(folder_indices) - 1
            curr_index = folder_indices[target_pos]
            curr_info = self.get_data_info(curr_index)
            real_path_list.append(curr_info['radar_points_path'])
            if 'ann_info' in curr_info:
                ann_info_frames.append(copy.deepcopy(curr_info['ann_info']))
            if 'eval_ann_info' in curr_info:
                eval_ann_info_frames.append(copy.deepcopy(curr_info['eval_ann_info']))

        input_dict['radar_points_paths'] = real_path_list
        if ann_info_frames:
            input_dict['ann_info_frames'] = ann_info_frames
        if eval_ann_info_frames:
            input_dict['eval_ann_info_frames'] = eval_ann_info_frames
        if real_path_list:
            center_idx = len(real_path_list) // 2
            input_dict['sample_name'] = Path(
                real_path_list[center_idx]).stem
        input_dict['num_frames'] = self.num_frames
        input_dict['source_folder'] = folder


        example = self.pipeline(input_dict)

        return example

    def _build_source_folder_cache(self) -> None:
        self.folder_to_indices = dict()
        self.index_to_folder_pos = []
        dataset_length = len(self)
        folder_items: Dict[str, List[Tuple[int, int]]] = dict()
        for idx in range(dataset_length):
            data_info = self.get_data_info(idx)
            folder = data_info.get('source_folder', 'unknown')
            frame_order = int(data_info.get('frame_order', idx))
            folder_items.setdefault(folder, []).append((frame_order, idx))
        folder_positions: Dict[int, Tuple[str, int]] = dict()
        for folder, items in folder_items.items():
            items.sort(key=lambda pair: pair[0])
            indices = [idx for _, idx in items]
            self.folder_to_indices[folder] = indices
            for pos, data_idx in enumerate(indices):
                folder_positions[data_idx] = (folder, pos)
        for idx in range(dataset_length):
            self.index_to_folder_pos.append(
                folder_positions.get(
                    idx,
                    (self.get_data_info(idx).get('source_folder', 'unknown'), 0)))
