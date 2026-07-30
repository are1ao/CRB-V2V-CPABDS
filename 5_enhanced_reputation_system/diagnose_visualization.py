# -*- coding: utf-8 -*-
"""诊断为什么 Detection Delay 和 Reputation Separation 没有显示"""

import json
from pathlib import Path

result_file = Path("d:/61-V2V/CRB-V2V-CPABDS/5_enhanced_reputation_system/results/experiment_results.json")

with open(result_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\n" + "="*70)
print("  Detection Delay & Separation 诊断")
print("="*70)

# 检查 obstacle 场景
if "static_obstacle" in data:
    ep = data["static_obstacle"][0]
    adv = ep["adversary_ids"][0]
    frame_start, frame_end = ep["attack_window"]
    
    print(f"\n[static_obstacle 场景]")
    print(f"  攻击车: {adv}")
    print(f"  攻击窗口: [{frame_start}, {frame_end}]")
    print(f"  硬阈值 HARD_THR: 0.5")
    
    for algo in ["ImprovedDRAMBR", "DRAMBR", "PlexeMDS"]:
        if algo not in ep["reputation_history"]:
            continue
        
        hist = ep["reputation_history"][algo].get(adv, [])
        if not hist:
            continue
        
        # 查找首次跌破 0.5 的帧
        detection_frame = None
        for i in range(frame_start, min(len(hist), frame_end + 30)):
            if hist[i] < 0.5:
                detection_frame = i
                break
        
        delay = (detection_frame - frame_start) if detection_frame else None
        
        print(f"\n  [{algo}]")
        print(f"    初始值 (帧0): {hist[0]:.3f}")
        print(f"    攻击开始 (帧{frame_start}): {hist[frame_start]:.3f}")
        
        if detection_frame:
            print(f"    首次 <0.5: 帧 {detection_frame} (值={hist[detection_frame]:.3f})")
            print(f"    检测延迟: {delay} 帧 ✓")
        else:
            print(f"    首次 <0.5: 未检测到 ✗")
            print(f"    检测延迟: None (无数据)")
            
        # 检查正常车
        norm_ids = [v for v in ep["reputation_history"][algo].keys() if v != adv]
        if norm_ids:
            norm_curves = [ep["reputation_history"][algo][v] for v in norm_ids[:3]]
            norm_at_end = [c[min(frame_end-1, len(c)-1)] for c in norm_curves if c]
            adv_at_end = hist[min(frame_end-1, len(hist)-1)]
            separation = (sum(norm_at_end) / len(norm_at_end)) - adv_at_end if norm_at_end else 0
            
            print(f"    窗口结束 (帧{frame_end-1}):")
            print(f"      攻击车: {adv_at_end:.3f}")
            print(f"      正常车均值: {sum(norm_at_end)/len(norm_at_end):.3f}")
            print(f"      Separation: {separation:.3f}")

print("\n" + "="*70)
print("  问题诊断")
print("="*70)

print("""
[可能原因]

1. 初始值过低（修复前）
   - 攻击车初始 0.4-0.6（低于旧阈值 0.3 无关）
   - 但可能部分算法初始就 <0.5，导致无检测延迟

2. 阈值修改为 0.5 后
   - 检测更快（好事）
   - 但如果初始 <0.5，检测延迟=0 或 None

3. Separation 计算正常
   - 应该有数据，如果没显示可能是绘图逻辑问题

[解决方案]

修复后重新运行实验，确保:
  - 所有车辆初始 = 1.0 ✓
  - 攻击车从 1.0 开始下降
  - 检测延迟计算正确

然后 Detection Delay 和 Separation 应该正常显示。
""")

print("="*70)
print()
