# Heatmap-Based mmWave Pose Estimation

This document summarizes the heatmap-based pose estimation configuration used by the project.

## Overview

The model predicts 2D range-azimuth heatmaps and elevation maps, then decodes 3D keypoints from the heatmap distributions. The training target is generated from 3D keypoints with Gaussian kernels in range and azimuth.

## Main Components

- Heatmap prediction for joint confidence in the range-azimuth plane.
- Elevation prediction at each heatmap location.
- Soft-argmax decoding for differentiable 3D keypoint estimation.
- Multi-person matching with brute-force or Hungarian assignment.
- MPJPE-based evaluation and optional visualization utilities.

## Typical Usage

```bash
python tools/train.py configs/[config].py
python tools/test.py configs/[config].py work_dirs/[checkpoint].pth
python tools/visualize_heatmap.py --help
```

## Notes

- Inputs and outputs use spherical coordinates where applicable.
- Azimuth targets handle angular wrap-around.
- Heatmap memory usage scales with batch size, number of persons, number of joints, and heatmap resolution.
- For reviewer release, paths should be set through command-line arguments or relative project paths.
