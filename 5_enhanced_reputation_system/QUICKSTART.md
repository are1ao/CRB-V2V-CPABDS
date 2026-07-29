# 快速开始指南

## 5分钟快速测试

### 1. 验证改进是否生效

```bash
cd d:\61-V2V\CRB-V2V-CPABDS\5_enhanced_reputation_system
python test_improvements.py
```

**预期输出**:
```
============================================================
增强信誉系统 - 改进效果验证测试
============================================================

============================================================
测试1: EWMA平滑系数 (0.7*old + 0.3*new)
============================================================
初始信誉: 0.5000
第1次异常后: 0.4150
第2次异常后: 0.3405
第3次异常后: 0.2783
第4次异常后: 0.2248
第5次异常后: 0.1823
✓ EWMA平滑系数测试通过

...

总计: 6/6 通过
🎉 所有测试通过！改进已正确实施。
```

### 2. 运行小规模实验

编辑 `run_complete_experiment.py`，设置较少的episodes:

```python
# 在main()函数中修改
results = runner.run_all_scenarios(scenarios, num_episodes=1)  # 改为1
```

然后运行:

```bash
python run_complete_experiment.py
```

### 3. 生成可视化

```bash
python advanced_visualization.py
```

查看 `visualizations/` 目录下的PNG图片。

---

## 完整实验流程

### Step 1: 数据准备

确认数据集路径正确:

```python
dataset_root = "d:/61-V2V/CRB-V2V-CPABDS/DataSet"
```

确保存在以下目录:
- `carla_export_n10_v200x50_ghost_teleport_pcd/`
- `carla_export_10k_n10_v200x50_ghost_drift/`
- `carla_export_n10_v200x50_brake_burst/` (可选)

### Step 2: 配置实验参数

在 `run_complete_experiment.py` 中:

```python
# 选择场景
scenarios = [
    "carla_export_n10_v200x50_ghost_teleport_pcd",
    "carla_export_10k_n10_v200x50_ghost_drift",
    # "carla_export_n10_v200x50_brake_burst",  # 可选
]

# 设置episode数量
num_episodes = 5  # 建议3-10个
```

在 `improved_reputation_engine.py` 中微调参数（可选）:

```python
config = ReputationConfig(
    ewma_alpha=0.3,              # 新信息权重
    ewma_beta=0.7,               # 历史权重
    first_offense_multiplier=2.0, # 首次作恶惩罚倍数
    adaptive_count_factor=0.05,  # 自适应加速系数
)
```

### Step 3: 运行完整实验

```bash
python run_complete_experiment.py
```

**运行时间**: 约5-15分钟（取决于episode数量和数据集大小）

**输出**:
- `results/experiment_results.json` - 完整实验数据
- 控制台日志 - 实时进度和统计

### Step 4: 生成可视化

```bash
python advanced_visualization.py
```

**输出**:
- `visualizations/comparison_teleport.png` - 瞬移攻击综合对比图
- `visualizations/comparison_drift.png` - 漂移攻击综合对比图
- 包含：信誉曲线、检测延迟、改进效果表格

### Step 5: 分析结果

打开生成的PNG图片，重点关注:

1. **信誉曲线**（左上）
   - ImprovedDRAMBR（实线红色）vs DRAMBR（虚线蓝色）
   - 攻击车信誉是否在50帧内跌破0.50

2. **检测延迟**（右上）
   - ImprovedDRAMBR应显著低于其他算法
   - 目标: <25帧

3. **改进效果表**（下方）
   - 检测延迟降低百分比
   - 信誉分离度提升
   - 攻击车最终信誉降低

---

## VEINS集成测试

### Step 1: 启动服务器

在一个终端中:

```bash
python reputation_socket_server.py
```

**预期输出**:
```
2026-07-28 13:30:00 - INFO - 初始化信誉服务器: 0.0.0.0:8888
2026-07-28 13:30:00 - INFO - ✓ 信誉服务器启动成功: 0.0.0.0:8888
2026-07-28 13:30:00 - INFO - 等待VEINS客户端连接...
```

### Step 2: 测试客户端

在另一个终端中，使用Python测试:

```python
import socket
import json

# 连接服务器
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1', 8888))

# 发送更新请求
request = {
    "type": "update_reputation",
    "vehicle_id": "147",
    "observation": {
        "position_error": 5.2,
        "velocity_error": 2.1,
        "timestamp_error": 0.1,
        "message_frequency": 10.0
    }
}

sock.sendall(json.dumps(request).encode('utf-8') + b'\n')

# 接收响应
response = sock.recv(4096).decode('utf-8')
print(json.loads(response))

sock.close()
```

**预期响应**:
```json
{
  "status": "success",
  "vehicle_id": "147",
  "old_score": 0.5,
  "new_score": 0.432,
  "filter_weight": 0.0,
  "warning_level": 1,
  "first_offense": false,
  "early_warning": false,
  "timestamp": 1722148200.123
}
```

---

## 故障排查

### 问题1: ModuleNotFoundError

**错误**: `ModuleNotFoundError: No module named 'improved_reputation_engine'`

**解决**:
```bash
# 确保在正确的目录
cd d:\61-V2V\CRB-V2V-CPABDS\5_enhanced_reputation_system

# 或设置PYTHONPATH
set PYTHONPATH=d:\61-V2V\CRB-V2V-CPABDS\5_enhanced_reputation_system
```

### 问题2: FileNotFoundError (数据集)

**错误**: `Episode not found: ...`

**解决**:
```python
# 在run_complete_experiment.py中检查路径
print(Path(dataset_root).exists())  # 应输出True

# 列出可用episodes
from pathlib import Path
episodes = list(Path(dataset_root).glob("*/episode_*"))
print(episodes)
```

### 问题3: 信誉下降仍然太慢

**解决**: 在 `improved_reputation_engine.py` 中增大参数:

```python
config = ReputationConfig(
    ewma_alpha=0.4,              # 从0.3增加到0.4
    negative_step=0.15,          # 从0.1增加到0.15
    first_offense_multiplier=2.5, # 从2.0增加到2.5
    adaptive_count_factor=0.08,  # 从0.05增加到0.08
)
```

### 问题4: 误报率过高

**解决**: 减小参数:

```python
config = ReputationConfig(
    ewma_alpha=0.25,             # 从0.3减小到0.25
    first_offense_window=50,     # 从30增加到50
    first_offense_multiplier=1.5, # 从2.0减小到1.5
)
```

### 问题5: 可视化字体乱码

**解决**: 安装中文字体或使用英文:

```python
# 在advanced_visualization.py中修改
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']  # 使用英文字体
```

---

## 性能优化建议

### 大规模实验

如果需要运行10+ episodes:

```python
# 启用多进程
from multiprocessing import Pool

def run_episode_wrapper(args):
    scenario, ep_idx = args
    return runner.run_single_episode(scenario, ep_idx)

with Pool(4) as pool:
    results = pool.map(run_episode_wrapper, episode_list)
```

### 内存优化

如果内存不足:

```python
# 在run_complete_experiment.py中
# 每处理一个episode后立即保存并清理
result = runner.run_single_episode(scenario, ep_idx)
with open(f"results/episode_{ep_idx}.json", 'w') as f:
    json.dump(result, f)
del result  # 释放内存
```

---

## 下一步

1. **调参优化**: 根据实验结果微调参数
2. **论文撰写**: 使用生成的图表和数据
3. **真实部署**: 集成到VEINS/SUMO仿真或真实车辆
4. **扩展功能**: 添加攻击类型自适应、联邦学习等

---

## 支持

如有问题，请检查:
1. `README.md` - 详细文档
2. `test_improvements.py` - 验证改进
3. 日志输出 - 调试信息

联系方式: [your.email@example.com]
