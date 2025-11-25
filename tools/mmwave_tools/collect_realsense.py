import os
import glob
import shutil
from tqdm import tqdm

filter_path = '/workspace/ikmmpose/data/val/keypoints_3d'
filters = os.listdir(filter_path)
original_path = '/workspace/dataproc/data/1761275143824628506/realsense'
store_path = '/workspace/ikmmpose/realsense'

for f in tqdm(filters):
    founds = glob.glob(os.path.join(original_path,os.path.splitext(os.path.basename(f))[0]+'*'))
    for temp in founds:
        shutil.copy(temp,os.path.join(store_path,os.path.basename(temp)))