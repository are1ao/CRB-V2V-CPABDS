# Enhanced Reputation System（增强信誉系统）

面向 V2V 协同感知的信誉管理改进方案。针对传统系统**信誉下降过慢**、**首次作恶响应迟钝**的问题，通过 EWMA 加速、首次作恶放大惩罚、自适应步长与多级过滤，实现更快、更稳的恶意车辆识别。

## 核心改进

| 模块 | 改进前 | 改进后 | 作用 |
|------|--------|--------|------|
| EWMA 平滑 | `0.85·old + 0.15·new` | `0.7·old + 0.3·new` | 信誉变化速度约翻倍 |
| 首次作恶惩罚 | 无 | 高信誉首次异常时步长 ×2 | 压缩高信誉车辆突发攻击的检测窗口 |
| 自适应步长 | `1 + count×0.01` | `1 + count×0.05` | 连续异常时惩罚加速 |
| LSTM 预警 | 无 | 预测偏差 >0.15 额外惩罚 | 行为突变的二级预警（无 PyTorch 时回退线性外推） |
| 过滤策略 | 硬阈值一刀切 | 软/硬双阈值多级权重 | 减少阈值附近抖动 |

**多级过滤权重**

- `score ≥ 0.70` → 权重 `1.0`（完整采纳）
- `0.50 ≤ score < 0.70` → 权重 `0 ~ 0.3` 线性插值（降权）
- `score < 0.50` → 权重 `0.0`（完全排除）

## 四种攻击场景

实验基于 `DataSet/` 中全部四类攻击，协议一致：

| 短名 | 数据集目录 | 攻击含义 | 典型窗口 |
|------|------------|----------|----------|
| `teleport` | `carla_export_n10_v200x50_ghost_teleport_pcd` | 幽灵瞬移 | frame 50 → 结束 |
| `drift` | `carla_export_10k_n10_v200x50_ghost_drift` | 幽灵漂移 | frame 50–100 |
| `reverse` | `carla_export_n10_v200x50_ghost_rev_pcd` | 逆向幽灵 | frame 50 → 结束 |
| `brake` | `carla_export_n10_v200x50_brake_burst` | 刹车欺诈 | burst 约 20 帧 |

**对比算法**：ImprovedDRAMBR（本方案）、DRAMBR（保守基线）、PlexeMDS、MajorityVoting、StaticReputation、NoTrustFusion。

## 文件结构

```
5_enhanced_reputation_system/
├── improved_reputation_engine.py   # 信誉引擎（EWMA / 首次作恶 / LSTM / 多级过滤）
├── run_complete_experiment.py      # 四攻击完整对比实验
├── advanced_visualization.py       # 论文级可视化
├── reputation_socket_server.py     # VEINS/SUMO Socket 服务
├── test_improvements.py            # 单元级改进验证
├── quick_test.py                   # 一键实验 + 出图
├── run_all.py                      # 交互式一键流程
├── results/
│   └── experiment_results.json
└── visualizations/
    ├── comparison_teleport.png     # 单攻击综合面板 ×4
    ├── comparison_drift.png
    ├── comparison_reverse.png
    ├── comparison_brake.png
    ├── cross_attack_summary.png    # 四攻击横向总览
    ├── reputation_gallery.png      # 信誉下降画廊
    ├── metrics_heatmap.png         # 算法×攻击热力图
    └── improvement_radar.png       # 多维能力雷达图
```

## 快速开始

### 1. 依赖

```bash
pip install numpy pandas matplotlib seaborn pyyaml tqdm
pip install torch          # 可选，LSTM 预警
pip install scikit-learn   # 可选
```

### 2. 验证改进逻辑

```bash
cd d:/61-V2V/CRB-V2V-CPABDS/5_enhanced_reputation_system
python test_improvements.py
```

### 3. 跑四种攻击实验

```bash
# 默认每个场景 3 个 episode；可用环境变量覆盖
set EPISODES=2
python run_complete_experiment.py
```

或一键实验 + 出图：

```bash
python quick_test.py
```

### 4. 仅重新生成可视化

```bash
python advanced_visualization.py
```

## 实验结果（当前跑次）

评估口径：检测延迟 = 攻击开始后首次跌破硬阈值 `0.50` 的帧数；攻击车信誉取**攻击窗口结束时刻**（避免短时攻击结束后回升造成指标失真）。每场景 2 episodes。

| 攻击 | ImprovedDRAMBR 延迟 | DRAMBR 延迟 | 延迟降低 | Imp. 分离度 | DRAMBR 分离度 |
|------|--------------------:|------------:|---------:|------------:|--------------:|
| Teleport | 1.0 | 7.0 | **86%** | 1.00 | 0.97 |
| Drift | 2.0 | 9.0 | **78%** | 1.00 | 0.97 |
| Reverse | 1.0 | 7.0 | **86%** | 1.00 | 0.97 |
| Brake | 2.5 | 12.0 | **79%** | 1.00 | 0.75 |

相对 PlexeMDS（约 9.5–13 帧），ImprovedDRAMBR 仍保持明显更快的首次检出。四类攻击检测率均为 **100%**。

**现象解读**

1. ImprovedDRAMBR 呈现「惩罚快、恢复慢」：攻击一开始信誉陡降，窗口结束后线性回升。
2. 保守 DRAMBR 下降更缓，短窗口攻击（Brake）结束时信誉仍偏高，分离度更差。
3. 多级过滤权重随信誉同步下滑，可在协同融合中平滑降权而非硬切。

## 可视化说明

| 文件 | 内容 |
|------|------|
| `comparison_*.png` | 攻击车轨迹、过滤权重、延迟/分离度柱图、跨 episode 均值±std、指标摘要表 |
| `cross_attack_summary.png` | 四攻击 × 多算法：延迟、分离度、相对增益、检测率 |
| `reputation_gallery.png` | 四攻击信誉曲线并排对比（Improved vs DRAMBR） |
| `metrics_heatmap.png` | 延迟与分离度热力图 |
| `improvement_radar.png` | 检测速度 / 分离度 / 检出率 / 下降斜率 / 良性稳定性 |

## 关键参数

```python
config = ReputationConfig(
    ewma_alpha=0.3,                 # 新信息权重
    ewma_beta=0.7,                  # 历史权重
    positive_step=0.05,
    negative_step=0.1,
    first_offense_multiplier=2.0,
    high_reputation_threshold=0.85,
    first_offense_window=30,        # 帧
    adaptive_count_factor=0.05,
    filter_threshold_soft=0.70,
    filter_threshold_hard=0.50,
    filter_weight_soft=0.30,
    lstm_window=10,
    lstm_deviation_threshold=0.15,
)
```

基线 DRAMBR 使用保守配置（慢速衰减）：`alpha=0.15, beta=0.10, anomaly_threshold=0.35`，对应改进前风格，便于对照。

### 调参建议

- **下降仍偏慢**：增大 `ewma_alpha` / `negative_step` / `adaptive_count_factor`
- **误报偏高**：增大 `first_offense_window`，降低 `first_offense_multiplier`
- **良性车波动大**：减弱自适应加速，或拉长一致性历史窗口

## VEINS 集成

```bash
python reputation_socket_server.py
# 监听 0.0.0.0:8888，TCP + JSON
```

支持：`update_reputation` / `get_reputation` / `get_filter_weight` / `get_statistics`。

请求示例：

```json
{
  "type": "update_reputation",
  "vehicle_id": "147",
  "observation": {
    "position_error": 5.2,
    "velocity_error": 2.1,
    "timestamp_error": 0.1,
    "message_frequency": 10.0,
    "frame_idx": 75
  }
}
```

## 论文贡献点（可直接引用表述）

1. **动态 EWMA**：系统论证传统 `0.85/0.15` 过于保守，采用 `0.7/0.3` 在速度与稳定性间折中。
2. **首次作恶放大惩罚**：基于高信誉持续时长的自适应惩罚，缩短突发攻击检测延迟。
3. **预测 + 反应双层防御**：LSTM/趋势偏差预警叠加实时证据更新。
4. **多级过滤**：软硬阈值过渡，缓解硬阈值「悬崖效应」。

## License

按项目整体许可证执行。
