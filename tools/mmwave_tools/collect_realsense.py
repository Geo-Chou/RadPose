import os
import glob
import shutil
from tqdm import tqdm

filter_path = '/workspace/mmpose/data_batch/val/annotations'
filters = os.listdir(filter_path)
original_path = '/workspace/dataproc/data/1764559822514280600/realsense'
store_path = '/workspace/mmpose/realsense'

for f in tqdm(filters):
    founds = glob.glob(os.path.join(original_path,os.path.splitext(os.path.basename(f))[0]+'*'))
    for temp in founds:
        shutil.copy(temp,os.path.join(store_path,os.path.basename(temp)))