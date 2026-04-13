# RadPose: 3D Human Skeleton Estimation with an Integrated Sensing-and-Communication Device

> Geo Jie Zhou, Ruxin Lin, Shengding Liu, Bingheng Li
> 
> Department of Computer Science Engineering, Michigan State University

**This is the official implement of RadPose. (This work has been submit for publication, please check interest confliction)**

## Demo
<p align="center">
  <video src="./demo/demo_anonymous.mp4" width="720px" controls>
    您的浏览器不支持 video 标签。
  </video>
</p>

## Introduction
Human skeleton estimation under adverse lighting and occlusion conditions remains challenging for camera- and LiDAR-based systems. Although mmWave radars provides robustness to illumination and occlusion, they rely on dedicated hardware and licensed spectrum, limiting scalability.
In this paper, we present RadPose, a learning-based monostatic integrated sensing and communication (ISAC) framework that enables commodity mmWave communication devices to perform 3D human skeleton estimation. Unlike specialized radars, RadPose leverages monostatic full-duplex operation to extract channel impulse responses (CIRs) directly from standard OFDM communication signals, generating 3D radio point clouds without additional sensing hardware.
To address the sparsity and noise of RF observations, RadPose employs a cross-frame transformer-based skeleton inference network with semantic-guided spatial attention and target-driven temporal attention for robust spatio-temporal aggregation. It further adopts a joint-aware inverse kinematics refinement module based on graph convolution to enforce skeletal topology, bone-length consistency, and anatomical plausibility.
Extensive experiments demonstrate that RadPose achieves competitive accuracy compared to state-of-the-art mmWave radar-based approaches while maintaining robust performance in unseen and occluded scenarios.

## 🔧 Installation
This project is implemented based on [mmengine](https://mmengine.readthedocs.io/en/latest/). Before clone this repo, please follow official instructions of mmengine, mmcv, mmdet3d, mmpose to install prerequisited packages. 

**Note: version may has conflicts, should be resolved by checking offical documents to find best version for installation.**
Here, we provide the dependencies used for implementation on our server.
```
# Basic packages
conda create -n radpose python=3.8 -y
conda activate radpose
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# mmlab packages
pip install -U openmim
mim install mmengine
mim install 'mmcv==2.1.0'
mim install 'mmdet==3.2.0'
mim install "mmpose>=1.1.0"
git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
cd mmdetection3d
pip install -v -e .
cd ..

# radpose packages
# clone this repo
git clone https://github.com/Geo-Chou/RadPose.git
cd RadPose
pip install -v -e .

pip install spconv-cu118
```
If your current cuda is not 11.8, and you want to follow instructions above, please first change the environment parameters to **CUDA 11.8**.

##  Dataset Preparation
Please see [this blog](https://blog.geo-zhou.com/2025/11/01/251101_mmwave_Data_Collection/) for quick start.

## 🚀 Quick Start
### Training and Evaluation
```
# For Dist Training
cd RadPose
./tools/dist_train.sh configs/[target config file].py [number of gpus]

# For single-gpu training
cd RadPose
./tools/train.py configs/[target config file].py

# For evaluation
cd RadPose
./tools/test.py configs/[target config file].py work_dirs/[target checkpoint file].pth
```
To refine the skeleton with inverse kinematic, you may need train the backbone first, like this [config file](https://github.com/Geo-Chou/RadPose/blob/main/configs/mmwave_heatmap_v2_fpn_query_aug_physics_stem.py). Then, you should transfer the checkpoint into inverse kinematic version in order to refine.
```
python tools/remap_heatmap_ckpt_to_ik.py [source ckpt path] [target ckpt path]
```

### Visualization
To visualize the predicted pose, what you should do is add following metric into config file, and then run test code. When done, visualize the result from output directory.
```
    dict(
        type='MatchedPredictionDump',
        out_dir='work_dirs/predictions/',
        use_ik_output=True,
        include_gt=True,
        include_errors=False
    ),

python tools/visualize_skeleton_3d.py --input work_dirs/predictions/ --output work_dirs/vis_output/ --pred-only
```

## 🙏 Acknowledgement
Thanks a lot to these excellent projects:

[mmpose](https://github.com/open-mmlab/mmpose/tree/main)

[mmengine](https://github.com/open-mmlab/mmengine)