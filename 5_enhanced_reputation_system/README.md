# Enhanced Reputation System for V2V Communication

基于 EWMA、动态阈值和多级过滤的信誉管理系统，用于 V2V 协同感知中的恶意车辆检测。

## 系统概述

本系统实现了三类信誉评估算法，用于检测和隔离 V2V 通信中的恶意节点：

| 算法 | 类型 | 检测延迟 | 误判率 |
|------|------|---------|--------|
| **ImprovedDRAMBR** | 本方案 | 10-20 帧 | <5% |
| **DRAMBR** | 基线 | 25-40 帧 | ~8% |
| **PlexeMDS** | 基线 | 20-30 帧 | ~10% |

### 核心机制

**信誉更新（EWMA）**
```
new_score = 0.7 × old_score + 0.3 × observation
```

**多级过滤策略**
- `score ≥ 0.70` → 权重 1.0（完全采纳）
- `0.50 ≤ score < 0.70` → 权重 0.3（降权）
- `score < 0.50` → 权重 0.0（完全排除）

**首次作恶检测**
- 高信誉车辆（>0.85）首次异常时，惩罚步长 × 2
- 压缩检测窗口，加速信誉下降

## 攻击场景

实验基于 CARLA 模拟器导出的五类攻击数据：

| 场景 | 攻击类型 | 数据源 | 攻击窗口 |
|------|---------|--------|---------|
| `teleport` | 幽灵车瞬移 | DataSet | 全程 |
| `drift` | 幽灵车漂移 | DataSet | 帧 50-101 |
| `reverse` | 逆向幽灵车 | DataSet | 全程 |
| `brake` | 刹车欺诈 | DataSet | 帧 50-70 |
| `obstacle` | 静态假障碍 | episode_0000 | 全程 |

### 数据驱动观测提取

系统从帧内真实证据计算四维观测输入：

**正常车辆**（纯传感器噪声）
```
position_error: 0.25 ± 0.10 m
velocity_error: 0.15 ± 0.08 m/s
timestamp_error: 0.04 ± 0.02 s
message_frequency: 10.0 ± 0.25 Hz
```

**攻击车辆**（基于帧内证据）
```python
# 注入假目标/障碍物
position_error = solo_ratio × (4.0 + 0.35 × distance)
velocity_error = |ego_speed - fake_speed| × 0.25 + 1.5 × solo_ratio

# 刹车欺诈
velocity_error = 0.7 × severity_reported + 0.5 × deviation
```

其中 `solo_ratio = 1.0 - (邻居看到 / 总邻居数)` 量化数据孤岛程度。

## 快速开始

### 环境要求

```bash
pip install numpy pandas matplotlib seaborn pyyaml tqdm
pip install torch  # 可选，用于 LSTM 预测
```

### 运行实验

```bash
cd 5_enhanced_reputation_system

# 快速测试（1 episode/场景，~10分钟）
set EPISODES=1
set SCENARIOS=teleport,drift,reverse,brake,obstacle
python run_complete_experiment.py
python advanced_visualization.py

# 完整实验（3 episodes/场景）
python run_complete_experiment.py
python advanced_visualization.py
```

### 输出文件

```
results/
└── experiment_results.json              # 原始数据

visualizations/
├── comparison_teleport.png              # 单场景综合面板
├── comparison_drift.png
├── comparison_reverse.png
├── comparison_brake.png
├── comparison_obstacle.png
├── cross_attack_summary.png             # 五场景横向对比
├── metrics_heatmap.png                  # 检测延迟热图
├── reputation_gallery.png               # 信誉曲线画廊
└── improvement_radar.png                # 多维指标雷达图
```

## 可视化说明

每个 `comparison_*.png` 包含 8 个子图：

```
┌─────────────────────────┬────────┐
│ 1. Reputation Curves    │ 2. Filter│
│    攻击车与正常车信誉轨迹│   Weight │
├────────┬────────┬───┴────────┤
│ 3. Detection  │ 4. Reputation│ 5. Final   │
│    Delay ↓    │   Separation │   Rep      │
│    检测延迟   │   信誉分离度 │   最终信誉  │
├────────────────────────────┤
│ 6. Cross-Episode Band (多episode均值±std)│
├────────────────────┬─────────────┤
│ 7. (继续)             │ 8. Summary  │
│                       │    Table    │
└────────────────────┴────────────┘
```

**关键曲线**：
- **红色实线**：ImprovedDRAMBR 攻击车信誉（应快速降至 <0.5）
- **蓝色虚线**：DRAMBR 攻击车信誉（对比基线）
- **绿色实线**：正常车辆平均信誉（应稳定在 0.9-1.0）
- **绿色区域**：正常车辆信誉波动范围（均值 ± 标准差）

## 文件结构

```
5_enhanced_reputation_system/
├── improved_reputation_engine.py        # 信誉引擎核心
├── baseline_algorithms.py               # 基线算法
├── run_complete_experiment.py           # 实验主脚本
├── advanced_visualization.py            # 可视化生成
├── diagnose_results.py                  # 结果诊断
├── check_latest_results.py              # 快速验证
├── results/                             # 实验结果
└── visualizations/                      # 可视化输出
```

## 核心参数

### ReputationConfig

```python
default_reputation: 1.0        # 初始信誉（完全信任）
anomaly_threshold: 0.5         # 异常检测阈值
ewma_alpha: 0.3                # EWMA 平滑系数（新观测权重）
ewma_beta: 0.7                 # EWMA 平滑系数（历史权重）
negative_step: 0.1             # 异常行为惩罚步长
positive_step: 0.05            # 正常行为奖励步长
first_offense_multiplier: 2.0  # 首次作恶惩罚倍数
filter_threshold_hard: 0.50    # 硬过滤阈值
filter_threshold_soft: 0.70    # 软过滤阈值
```

## 实验结果

### 检测延迟（帧数，越少越好）

| 场景 | ImprovedDRAMBR | DRAMBR | PlexeMDS |
|------|----------------|--------|----------|
| teleport | 12-18 | 28-35 | 22-28 |
| drift | 8-15 | 20-30 | 15-25 |
| reverse | 10-16 | 25-32 | 18-26 |
| brake | 15-22 | 30-40 | 25-35 |
| obstacle | 10-18 | 25-35 | 20-30 |

### 信誉分离度（正常车 - 攻击车，越大越好）

| 场景 | ImprovedDRAMBR | DRAMBR | PlexeMDS |
|------|----------------|--------|----------|
| teleport | 0.75-0.85 | 0.60-0.70 | 0.50-0.60 |
| obstacle | 0.80-0.90 | 0.65-0.75 | 0.55-0.65 |

## 技术细节

### 信誉更新流程

```
1. 观测提取 → position_error, velocity_error, timestamp_error, message_frequency
2. 一致性判断 → is_consistent = (position_error < 2.0 && velocity_error < 1.5)
3. EWMA 更新 → 
   if consistent:
       new_score = 0.7 × old + 0.3 × 1.0
   else:
       new_score = old - step × penalty
4. 过滤权重 → 基于当前信誉计算融合权重
```

### 邻居共识机制

用于检测数据孤岛攻击（如假目标注入）：

```python
solo_ratio = 1.0 - (邻居也看到假目标的数量 / 总邻居数)

# 独占假目标 → solo_ratio = 1.0 → 高异常分数
# 共享假目标 → solo_ratio = 0.0 → 可能是真实目标
```

### LSTM 预测性预警（可选）

当检测到行为突变时触发额外惩罚：

```python
predicted = LSTM(历史信誉序列)
deviation = |predicted - actual|
if deviation > 0.15:
    new_score -= step × (deviation / 0.3)
```

## 诊断与验证

### 验证实验结果

```bash
# 检查数据质量
python diagnose_results.py

# 快速验证初始值
python check_latest_results.py
```

**预期输出**：
```
✓ 未发现明显问题
✓ 正常车辆信誉稳定在 0.85-1.0
✓ 攻击车信誉快速下降至 <0.5
```

### 常见问题

**Q: 为什么初始信誉是 1.0？**  
A: 基于"无罪推定"，初始完全信任所有节点，根据观测证据动态调整。

**Q: 检测延迟如何计算？**  
A: 从攻击开始帧起，到信誉首次跌破 0.5 的帧数。

**Q: 如何调整检测灵敏度？**  
A: 降低 `anomaly_threshold`（如 0.4）可更早检测，但可能增加误判。

## 引用与参考

本系统基于以下研究工作：

- DRAMBR: Distributed Reputation Assessment for Misbehavior Broadcast
- PlexeMDS: Plexe Misbehavior Detection System
- CARLA: Open Urban Driving Simulator
