# 2.5D Heatmap-Based Pose Estimation

这是一个基于2.5D热力图的mmWave姿态估计系统，支持多人（最多2人）场景。

## 系统架构

### 1. **MmWaveHeatmapHead** ([heatmap_head.py](../mmwave3d/models/heads/heatmap_head.py))

预测头输出两个关键部分：

#### 输出张量
- **RA Heatmaps**: `(B, num_person, num_joints, range_bins, angle_bins)`
  - 2D热力图，表示关节在Range-Azimuth平面上的置信度/概率
  - 使用Sigmoid激活，值域[0, 1]
  
- **Elevation Maps**: `(B, num_person, num_joints, range_bins, angle_bins)`
  - 在每个RA位置存储elevation值
  - 使用Tanh激活归一化到[-1, 1]，训练更稳定

- **Keypoints**: `(B, num_person, num_joints, 3)`
  - 从热力图解码的3D关键点（球坐标：r, azimuth, elevation）
  - 用于后续evaluation计算MPJPE

#### 关键特性
- **Soft-argmax解码**：使用连续化的soft-argmax从热力图中提取关键点位置
  - Range: 加权平均 `r = Σ(r_u * H(u,v)) / Σ(H(u,v))`
  - Azimuth: 使用atan2处理角度wrap问题 `a = atan2(Σ(sin(a_v)*H), Σ(cos(a_v)*H))`
  - Elevation: 加权平均 `e = Σ(Z(u,v) * H(u,v)) / Σ(H(u,v))`

- **目标热力图生成**：自动从3D关键点生成训练目标
  - 使用2D高斯核：`H(u,v) = exp(-(Δr²)/(2σ_r²) - (Δa²)/(2σ_a²))`
  - 处理azimuth的周期性边界（wrap到[-π, π]）

### 2. **Loss Functions** ([heatmap_losses.py](../mmwave3d/models/losses/heatmap_losses.py))

#### Heatmap2DLoss
- **Focal Loss**：处理前景/背景不平衡
  ```
  L = -α(1-pred)^β * target * log(pred) - (1-α)pred^β * (1-target) * log(1-pred)
  ```
- **多人匹配**：
  - 自动为2人场景找最优预测-GT匹配
  - 基于成本矩阵（预测与GT热力图峰值的距离）
  - 使用brute-force匹配（适用于num_person≤2）
  - 对未匹配的预测施加惩罚（减少假阳性）

#### ElevationLoss
- **加权L1 Loss**：只在热力图置信度高的区域计算
  ```
  L_e = Σ w(u,v) * |pred_Z - target_Z|
  ```
  其中 `w(u,v) = H(u,v)` (热力图权重)
- **阈值过滤**：忽略热力图值<0.1的区域（避免背景干扰）

### 3. **Evaluation** ([heatmap_metrics.py](../mmwave3d/evaluation/heatmap_metrics.py))

#### HeatmapMPJPE
- **MPJPE计算**：从解码的关键点计算Mean Per Joint Position Error
- **多人匹配**：
  - 为每个样本找预测-GT的最优匹配
  - 使用Hungarian算法或brute-force（num_person≤2）
  - 按人平均计算MPJPE
- **输出指标**：
  - `MPJPE`: 整体平均误差
  - `MPJPE_joint_{j}`: 每个关节的误差统计
  - `num_persons`: 有效评估的人数

## 配置说明

参考配置文件：[mmwave_heatmap_2person.py](mmwave_heatmap_2person.py)

### 关键参数

```python
# 热力图分辨率
range_bins = 50      # Range维度的bin数
angle_bins = 180     # Azimuth维度的bin数（1度/bin）

# 高斯核参数
sigma_r = 0.15       # Range方向的高斯σ（米）
sigma_a = 0.087      # Azimuth方向的高斯σ（约5度）

# Head配置
head=dict(
    type='MmWaveHeatmapHead',
    num_person=2,    # 最多2人
    num_joints=17,   # 17个关节
    range_bins=range_bins,
    angle_bins=angle_bins,
    loss_heatmap=dict(
        type='Heatmap2DLoss',
        alpha=2.0,   # Focal loss参数
        beta=4.0,
        loss_weight=1.0
    ),
    loss_elevation=dict(
        type='ElevationLoss',
        threshold=0.1,
        loss_weight=0.5  # Elevation loss权重较低
    )
)

# 评估指标
val_evaluator = dict(
    type='HeatmapMPJPE',
    decode_from_heatmap=False  # 使用head解码的关键点
)
```

## 使用方法

### 训练
```bash
# 单卡训练
python tools/train.py configs/mmwave_heatmap_2person.py

# 多卡训练（4卡）
bash tools/dist_train.sh configs/mmwave_heatmap_2person.py 4
```

### 测试
```bash
# 单卡测试
python tools/test.py configs/mmwave_heatmap_2person.py work_dirs/mmwave_heatmap_2person/best_heatmap_MPJPE_epoch_*.pth

# 多卡测试
bash tools/dist_test.sh configs/mmwave_heatmap_2person.py work_dirs/mmwave_heatmap_2person/best_heatmap_MPJPE_epoch_*.pth 4
```

## 优势

### 相比直接回归3D坐标
1. **更密集的监督信号**：热力图提供2D空间上的密集梯度
2. **更好的收敛性**：Focal loss处理样本不平衡
3. **连续输出**：Soft-argmax保证可微分，梯度平滑
4. **多模态表示**：热力图峰值可以表示不确定性

### 多人场景
1. **自动匹配**：无需手动指定人的对应关系
2. **Hungarian算法**：支持任意多人的最优匹配（需要scipy）
3. **灵活扩展**：可轻松扩展到>2人，只需修改`num_person`参数
4. **假阳性惩罚**：未匹配的预测被抑制

## 可视化工具

### 使用方法

```bash
# 可视化单个样本的heatmap
python tools/visualize_heatmap.py \
    --config configs/mmwave_heatmap_2person.py \
    --sample-idx 0 \
    --out-dir visualizations/heatmap \
    --show  # 可选：直接显示而不保存

# 可视化特定关节
python tools/visualize_heatmap.py \
    --config configs/mmwave_heatmap_2person.py \
    --joint-idx 5 \
    --person-idx 0 \
    --out-dir visualizations/heatmap

# 可视化所有关节（网格显示）
python tools/visualize_heatmap.py \
    --config configs/mmwave_heatmap_2person.py \
    --out-dir visualizations/heatmap
```

### 可视化输出

工具会生成以下可视化：

1. **单关节热力图**：
   - 左图：RA热力图，显示置信度分布
   - 右图：Elevation地图，显示高度信息
   - 标注：预测点（青色星形）和真值点（绿色三角）

2. **所有关节网格图**：
   - 4x5网格显示17个关节的热力图
   - 每个子图显示一个关节的RA分布
   - 叠加预测和真值位置

3. **多人3D骨架**：
   - 三视图：顶视图(XY)、前视图(XZ)、侧视图(YZ)
   - 不同颜色表示不同的人
   - 显示完整的骨架连接

## 扩展到多人（>2人）

### 安装scipy（Hungarian算法依赖）

```bash
pip install scipy
```

### 修改配置

```python
# 在配置文件中修改
head=dict(
    type='MmWaveHeatmapHead',
    num_person=4,  # 改为4人或更多
    # ... 其他参数保持不变
)
```

### 算法说明

系统会自动选择最佳匹配算法：

- **scipy可用 + num_person>2**：使用Hungarian算法（最优匹配）
- **num_person=1**：直接选择最佳预测
- **num_person=2**：枚举所有排列（精确匹配）
- **scipy不可用 + num_person>2**：贪心匹配（次优但快速）

### 性能考虑

- **Hungarian算法**：O(n³)复杂度，适用于n≤10人
- **内存占用**：热力图内存 = B × P × J × H × W × 4字节
  - 示例：B=8, P=4, J=17, H=50, W=180 ≈ 980MB
- **建议**：
  - 4人以下：batch_size=8
  - 4-6人：batch_size=4
  - >6人：batch_size=2，考虑降低分辨率

## 调优建议

### 如果收敛慢
- 增加`sigma_r`和`sigma_a`（更宽的高斯核，更容易学习）
- 降低`loss_weight`中elevation的权重
- 使用`HeatmapMSELoss`替代`Heatmap2DLoss`（更简单）

### 如果精度不够
- 增加`range_bins`和`angle_bins`（更高分辨率）
- 减小`sigma_r`和`sigma_a`（更sharp的峰值）
- 调整focal loss的`alpha`和`beta`

### 如果多人匹配不准
- 确保安装scipy：`pip install scipy`
- Hungarian算法会自动用于>2人场景
- 检查成本矩阵的计算（可能需要更好的相似度度量）
- 增加未匹配预测的惩罚权重

## 可视化

可以使用提供的可视化工具：

```bash
# 基本使用
python tools/visualize_heatmap.py \
    --config configs/mmwave_heatmap_2person.py \
    --out-dir visualizations/heatmap

# 显示特定关节
python tools/visualize_heatmap.py \
    --config configs/mmwave_heatmap_2person.py \
    --joint-idx 5 \
    --show
```

热力图也可以用matplotlib直接可视化：
```python
import matplotlib.pyplot as plt

# 可视化某个关节的热力图
heatmap = pred_heatmaps[0, 0, 5]  # batch 0, person 0, joint 5
plt.imshow(heatmap.cpu().numpy(), cmap='hot')
plt.colorbar()
plt.xlabel('Azimuth bin')
plt.ylabel('Range bin')
plt.show()
```

## 注意事项

1. **坐标系**：输入/输出都使用球坐标(r, azimuth, elevation)
2. **归一化**：Elevation被归一化到[-1, 1]，需要在head中denormalize
3. **角度wrap**：Azimuth在生成目标热力图时需要处理周期性
4. **内存使用**：热力图占用较大内存（B×P×J×H×W），注意batch_size

## 扩展方向

- [x] 使用Hungarian算法支持任意多人匹配
- [x] 可视化工具（热力图、elevation、3D骨架）
- [ ] 热力图后处理（NMS等）
- [ ] 时序一致性约束
- [ ] 实时推理优化
- [ ] 不确定性估计（从热力图方差）
