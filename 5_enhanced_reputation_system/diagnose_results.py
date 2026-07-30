# -*- coding: utf-8 -*-
"""快速诊断实验结果数据质量"""

import json
from pathlib import Path

result_file = Path("d:/61-V2V/CRB-V2V-CPABDS/5_enhanced_reputation_system/results/experiment_results.json")

print("\n" + "="*70)
print("  实验结果数据质量诊断")
print("="*70)

with open(result_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"\n已加载场景: {list(data.keys())}")

# 检查每个场景
issues = []
for scenario, episodes in data.items():
    print(f"\n[{scenario}]")
    print(f"  Episodes: {len(episodes)}")
    
    for ep_idx, ep in enumerate(episodes):
        adv_ids = ep.get("adversary_ids", [])
        num_frames = ep.get("num_frames", 0)
        attack_window = ep.get("attack_window", [0, num_frames])
        
        print(f"  Episode {ep_idx}: {num_frames} 帧, 攻击车 {adv_ids}, 窗口 {attack_window}")
        
        # 检查各算法的信誉历史
        rep_hist = ep.get("reputation_history", {})
        
        for algo_name in ["ImprovedDRAMBR", "DRAMBR", "PlexeMDS"]:
            if algo_name not in rep_hist:
                continue
            
            algo_hist = rep_hist[algo_name]
            
            for adv in adv_ids:
                if adv not in algo_hist:
                    issues.append(f"{scenario} ep{ep_idx}: 攻击车 {adv} 缺少 {algo_name} 历史")
                    continue
                
                hist = algo_hist[adv]
                
                # 检查信誉下降
                initial = hist[0] if hist else 0.5
                frame_10 = hist[10] if len(hist) > 10 else initial
                frame_50 = hist[50] if len(hist) > 50 else initial
                final = hist[-1] if hist else initial
                
                # 问题 1: 信誉未下降
                if final >= initial * 0.95:
                    issues.append(
                        f"{scenario} ep{ep_idx} {algo_name}: 攻击车 {adv} 信誉未下降 "
                        f"({initial:.3f} → {final:.3f})"
                    )
                
                # 问题 2: 下降过慢（50 帧后仍高于 0.45）
                if len(hist) > 50 and frame_50 > 0.45 and algo_name == "ImprovedDRAMBR":
                    issues.append(
                        f"{scenario} ep{ep_idx} {algo_name}: 攻击车 {adv} 下降过慢 "
                        f"(帧50: {frame_50:.3f})"
                    )
                
                # 问题 3: 窗口外开始下降（数据泄露）
                window_start = attack_window[0]
                if window_start > 0 and len(hist) > window_start:
                    pre_attack = hist[max(0, window_start-1)]
                    if initial - pre_attack > 0.05:
                        issues.append(
                            f"{scenario} ep{ep_idx} {algo_name}: 攻击车 {adv} 窗口外已下降 "
                            f"(帧{window_start-1}: {pre_attack:.3f})"
                        )

print("\n" + "="*70)
print("  潜在问题检测")
print("="*70)

if not issues:
    print("\n✓ 未发现明显问题")
    print("\n数据质量检查要点:")
    print("  1. 攻击车信誉在攻击窗口内显著下降")
    print("  2. ImprovedDRAMBR 下降速度快于 DRAMBR")
    print("  3. 窗口外信誉保持稳定")
else:
    print(f"\n发现 {len(issues)} 个潜在问题:\n")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

# 详细检查 obstacle 场景
print("\n" + "="*70)
print("  static_obstacle 场景详细检查")
print("="*70)

if "static_obstacle" in data:
    obs_ep = data["static_obstacle"][0]
    adv = obs_ep["adversary_ids"][0]
    
    print(f"\n攻击车: {adv}")
    print(f"总帧数: {obs_ep['num_frames']}")
    print(f"攻击窗口: {obs_ep['attack_window']}")
    
    for algo in ["ImprovedDRAMBR", "DRAMBR", "PlexeMDS"]:
        if algo not in obs_ep["reputation_history"]:
            continue
        
        hist = obs_ep["reputation_history"][algo].get(adv, [])
        if not hist:
            continue
        
        print(f"\n[{algo}]")
        print(f"  帧0:   {hist[0]:.4f}")
        print(f"  帧10:  {hist[10]:.4f} (Δ = {hist[0]-hist[10]:.4f})")
        print(f"  帧50:  {hist[50]:.4f} (Δ = {hist[0]-hist[50]:.4f})")
        print(f"  帧100: {hist[100]:.4f} (Δ = {hist[0]-hist[100]:.4f})")
        print(f"  帧199: {hist[-1]:.4f} (Δ = {hist[0]-hist[-1]:.4f})")
        
        # 检测速度
        if hist[10] < hist[0] - 0.1:
            print(f"  ✓ 前10帧快速下降")
        else:
            print(f"  ✗ 前10帧下降缓慢")

# 检查正常车辆
print("\n" + "="*70)
print("  正常车辆信誉稳定性检查")
print("="*70)

normal_unstable = []
for scenario, episodes in data.items():
    for ep_idx, ep in enumerate(episodes):
        adv_ids = set(ep.get("adversary_ids", []))
        rep_hist = ep.get("reputation_history", {})
        
        for algo_name in ["ImprovedDRAMBR"]:
            if algo_name not in rep_hist:
                continue
            
            algo_hist = rep_hist[algo_name]
            
            for vehicle_id, hist in algo_hist.items():
                if vehicle_id in adv_ids:
                    continue
                
                if not hist:
                    continue
                
                initial = hist[0]
                final = hist[-1]
                variance = abs(final - initial)
                
                # 正常车辆信誉波动不应超过 0.15
                if variance > 0.15:
                    normal_unstable.append(
                        f"{scenario} ep{ep_idx}: 正常车 {vehicle_id} 波动过大 "
                        f"({initial:.3f} → {final:.3f}, Δ={variance:.3f})"
                    )

if not normal_unstable:
    print("\n✓ 正常车辆信誉稳定")
else:
    print(f"\n发现 {len(normal_unstable)} 个正常车辆异常波动:\n")
    for issue in normal_unstable[:5]:  # 只显示前5个
        print(f"  · {issue}")
    if len(normal_unstable) > 5:
        print(f"  ... (还有 {len(normal_unstable)-5} 个)")

print("\n" + "="*70)
print("  总结")
print("="*70)

if not issues and not normal_unstable:
    print("\n✓ 数据质量良好")
    print("  - 攻击车信誉正常下降")
    print("  - 正常车辆信誉稳定")
    print("  - ImprovedDRAMBR 性能符合预期")
    print("\n可视化结果应正确反映数据驱动观测提取的效果。")
else:
    print(f"\n⚠ 发现 {len(issues) + len(normal_unstable)} 个潜在问题")
    print("\n建议检查:")
    print("  1. ObservationExtractor 是否正确提取帧内证据")
    print("  2. 攻击窗口推断是否准确")
    print("  3. 是否所有场景都使用了数据驱动逻辑")

print()
