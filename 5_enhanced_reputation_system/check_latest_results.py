# -*- coding: utf-8 -*-
"""快速检查最新实验结果"""

import json

with open("d:/61-V2V/CRB-V2V-CPABDS/5_enhanced_reputation_system/results/experiment_results.json") as f:
    data = json.load(f)

obs = data["static_obstacle"][0]
adv = "147"
norm = "146"

print("\n" + "="*70)
print("  实验结果检查")
print("="*70)

print(f"\n[static_obstacle 场景]")
print(f"\n攻击车 {adv} - ImprovedDRAMBR:")
hist = obs["reputation_history"]["ImprovedDRAMBR"][adv]
print(f"  初始值 (帧0): {hist[0]:.3f}")
print(f"  帧10: {hist[10]:.3f}")
print(f"  帧50: {hist[50]:.3f}")

# 查找首次 <0.5 的帧
detection_frame = None
for i in range(200):
    if hist[i] < 0.5:
        detection_frame = i
        break

if detection_frame is not None:
    print(f"  首次 <0.5: 帧 {detection_frame} (值={hist[detection_frame]:.3f})")
    print(f"  Detection Delay: {detection_frame} 帧")
else:
    print(f"  首次 <0.5: 未检测到 ❌")
    print(f"  问题: 初始值可能仍然 <0.5 或没有下降到 <0.5")

print(f"\n正常车 {norm} - ImprovedDRAMBR:")
hist_norm = obs["reputation_history"]["ImprovedDRAMBR"][norm]
print(f"  初始值 (帧0): {hist_norm[0]:.3f}")
print(f"  帧50: {hist_norm[50]:.3f}")
print(f"  帧199: {hist_norm[-1]:.3f}")
print(f"  波动: {abs(hist_norm[-1] - hist_norm[0]):.3f}")

if hist_norm[0] >= 0.9:
    print(f"  状态: ✓ 初始值正常")
else:
    print(f"  状态: ❌ 初始值过低")