# -*- coding: utf-8 -*-
"""
快速测试脚本 - 使用1个episode快速生成可视化
适合快速验证系统功能
"""

import os
import sys
from pathlib import Path

print("\n" + "="*70)
print("  快速测试 - 生成可视化示例")
print("="*70)

# 检查当前目录
current = Path.cwd()
if not (current / "run_complete_experiment.py").exists():
    print("\n请在 5_enhanced_reputation_system 目录下运行此脚本")
    print("cd d:\\61-V2V\\CRB-V2V-CPABDS\\5_enhanced_reputation_system")
    sys.exit(1)

print("\n正在运行快速测试...")
print("- 场景: teleport / drift / reverse / brake / obstacle")
print("- Episodes: DataSet 每场景 1 个；obstacle 为独立 episode_0000")
print("- 观测: 数据驱动四维接口\n")

# 运行实验
print("[1/2] 运行实验...")
os.environ["EPISODES"] = "1"
os.environ["SCENARIOS"] = "teleport,drift,reverse,brake,obstacle"
os.system("python run_complete_experiment.py")

# 生成可视化
print("\n[2/2] 生成可视化...")
os.system("python advanced_visualization.py")

# 检查结果
print("\n" + "="*70)
print("  结果检查")
print("="*70)

results_dir = Path("results")
vis_dir = Path("visualizations")

if results_dir.exists():
    result_files = list(results_dir.glob("*.json"))
    print(f"\n[OK] 实验结果: {len(result_files)} 个文件")
    for f in result_files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
else:
    print("\n[WARNING] 未找到results目录")

if vis_dir.exists():
    vis_files = list(vis_dir.glob("*.png"))
    print(f"\n[OK] 可视化结果: {len(vis_files)} 个文件")
    for f in vis_files:
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
        print(f"    位置: {f.absolute()}")
else:
    print("\n[WARNING] 未找到visualizations目录")

print("\n" + "="*70)
print("  完成！")
print("="*70)
print("\n查看可视化结果:")
for name in [
    "comparison_teleport.png",
    "comparison_drift.png",
    "comparison_reverse.png",
    "comparison_brake.png",
    "cross_attack_summary.png",
    "metrics_heatmap.png",
    "reputation_gallery.png",
    "improvement_radar.png",
]:
    p = vis_dir / name if vis_dir.exists() else Path("visualizations") / name
    print(f"  {p if not vis_dir.exists() else p.absolute()}")
print()
