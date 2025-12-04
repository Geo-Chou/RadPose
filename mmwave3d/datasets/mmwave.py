from mmengine.fileio import join_path, load, list_dir_or_file
from mmengine.logging import print_log
from mmengine.utils import is_abs
from mmengine.dataset import BaseDataset

import numpy as np
from typing import Callable, List, Optional, Set, Union, Dict
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
                 **kwargs)->None:
        self.num_frames = num_frames
        self.backend_args = backend_args
        self.load_eval_anns = load_eval_anns
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)
    
    def load_data_list(self) -> List[dict]:
        raw_data_list = list ()
        
        #assume the data is stored in the following structure:
        # /data_root/
        #    ├── train/
        #    │    ├── annotations/
        #    │    │    ├── sample1.json
        #    │    ├── radar_points/
        #    │    │    ├── sample1.npz
        radar_list = sorted(os.listdir(osp.join(self.data_root,  'val' if self.test_mode else 'train', 'radar_points')))
        for i, ann_name in enumerate(sorted(os.listdir(self.ann_file))):
            ann = load(join_path(self.ann_file, ann_name))
            raw_data_list.append(dict(radar_path = "radar_points/"+radar_list[i],
                                      instances = [{'keypoints_3d':obj['keypoints_sp']}
                                                   for obj in ann],
                                    ))
        if len(raw_data_list) == 0:
            raise ValueError(f'No valid annotation file found in '
                             f"{self.ann_file}")

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
        info['radar_points_path'] = osp.join(self.data_root, 'val' if self.test_mode else 'train', info['radar_path'])

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
        max_frame_offset = int(self.num_frames/2)
        max_data_index = len(self.data_list) -1
        for i in range(self.num_frames):
            curr_index = index - max_frame_offset + i
            if curr_index < 0:
                curr_index = 0
            elif curr_index > max_data_index:
                curr_index = max_data_index
            curr_info = self.get_data_info(curr_index)
            real_path_list.append(curr_info['radar_points_path'])

        input_dict['radar_points_paths'] = real_path_list
        input_dict['num_frames'] = self.num_frames


        example = self.pipeline(input_dict)

        return example
