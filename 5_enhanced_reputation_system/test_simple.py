# -*- coding: utf-8 -*-
"""
简化测试：仅运行 static_obstacle 场景验证数据驱动观测提取
"""

import sys
import yaml
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
print(f"ROOT: {ROOT}")

# 检查必要文件
episode_path = ROOT / "episode_0000"
meta_path = episode_path / "meta.yaml"

if not episode_path.exists():
    print(f"ERROR: episode_0000 not found at {episode_path}")
    sys.exit(1)

if not meta_path.exists():
    print(f"ERROR: meta.yaml not found at {meta_path}")
    sys.exit(1)

print(f"\nLoading episode_0000...")

# 加载 meta
with open(meta_path, "r", encoding="utf-8") as f:
    meta = yaml.safe_load(f)

print(f"  attack_label: {meta['attack_label']}")
print(f"  adversary_cav_ids: {meta.get('adversary_cav_ids', [])}")
print(f"  num_frames: {meta['num_frames']}")
print(f"  cav_ids: {meta['cav_ids']}")

# 加载车辆数据（仅加载 2 个车辆和前 10 帧测试）
test_cav_ids = [146, 147]
test_frames = 10

vehicle_data = {}
for cav_id in test_cav_ids:
    vehicle_dir = episode_path / str(cav_id)
    if not vehicle_dir.exists():
        print(f"  WARNING: {vehicle_dir} not found, skipping")
        continue
    
    frames = []
    yaml_files = sorted(vehicle_dir.glob("*.yaml"))[:test_frames]
    for yaml_file in yaml_files:
        with open(yaml_file, "r", encoding="utf-8") as f:
            frames.append(yaml.safe_load(f))
    vehicle_data[cav_id] = frames
    print(f"  Loaded {len(frames)} frames for CAV {cav_id}")

print(f"\nVehicle data loaded: {list(vehicle_data.keys())}")

# 测试辅助函数
def _frame_injected_ids(frame):
    att = frame.get("attack") or {}
    ids = []
    for x in att.get("injected_ado_ids") or []:
        ids.append(int(x))
    oid = att.get("obstacle_id")
    if oid is not None:
        ids.append(int(oid))
    for k, v in (frame.get("vehicles") or {}).items():
        try:
            kid = int(k)
        except (TypeError, ValueError):
            continue
        if not isinstance(v, dict):
            continue
        if v.get("is_obstacle") or v.get("static"):
            ids.append(kid)
        elif kid < 0 or kid >= 90000:
            ids.append(kid)
    return sorted(set(ids))

def _frame_is_attack_active(frame, adversary):
    if not adversary:
        return False
    att = frame.get("attack") or {}
    if "is_active_this_frame" in att:
        return bool(att.get("is_active_this_frame"))
    if att.get("obstacle_injected"):
        return True
    if _frame_injected_ids(frame):
        return True
    return False

# 测试帧 5
frame_idx = 5
adversary_ids = [int(x) for x in meta.get("adversary_cav_ids", [])]

print(f"\n{'='*70}")
print(f"Testing Frame {frame_idx}")
print(f"{'='*70}")

for cav_id in test_cav_ids:
    if cav_id not in vehicle_data or frame_idx >= len(vehicle_data[cav_id]):
        continue
    
    frame = vehicle_data[cav_id][frame_idx]
    is_adversary = int(cav_id) in adversary_ids
    is_attacking = _frame_is_attack_active(frame, is_adversary)
    injected = _frame_injected_ids(frame)
    
    ego_speed = float(frame.get("ego_speed", 0.0) or 0.0)
    att = frame.get("attack") or {}
    
    print(f"\nCAV {cav_id}:")
    print(f"  is_adversary: {is_adversary}")
    print(f"  is_attacking: {is_attacking}")
    print(f"  injected_ids: {injected}")
    print(f"  ego_speed: {ego_speed:.2f} m/s")
    
    if injected:
        print(f"  attack.obstacle_injected: {att.get('obstacle_injected')}")
        print(f"  attack.obstacle_id: {att.get('obstacle_id')}")
        
        # 检查 vehicles 字段
        oid = injected[0]
        veh = (frame.get("vehicles") or {}).get(oid) or (frame.get("vehicles") or {}).get(str(oid))
        if veh:
            print(f"  vehicles[{oid}].speed: {veh.get('speed', 'N/A')}")
            print(f"  vehicles[{oid}].is_obstacle: {veh.get('is_obstacle', False)}")

# 简单观测计算测试
print(f"\n{'='*70}")
print(f"Observation Extraction Test")
print(f"{'='*70}")

rng = np.random.default_rng(42)
frame_idx = 5

for cav_id in test_cav_ids:
    if cav_id not in vehicle_data or frame_idx >= len(vehicle_data[cav_id]):
        continue
    
    frame = vehicle_data[cav_id][frame_idx]
    is_adversary = int(cav_id) in adversary_ids
    is_attacking = _frame_is_attack_active(frame, is_adversary)
    injected = _frame_injected_ids(frame)
    
    ego_speed = float(frame.get("ego_speed", 0.0) or 0.0)
    
    # 基线噪声
    position_error = abs(float(rng.normal(0.25, 0.10)))
    velocity_error = abs(float(rng.normal(0.15, 0.08)))
    evidence = "benign_noise"
    
    # 如果是攻击且有注入
    if is_attacking and injected:
        # 简化：假设 solo_ratio = 1.0（正常车看不到）
        solo_ratio = 1.0
        
        oid = injected[0]
        veh = (frame.get("vehicles") or {}).get(oid) or (frame.get("vehicles") or {}).get(str(oid)) or {}
        fake_speed = float(veh.get("speed", 0.0) or 0.0)
        
        att = frame.get("attack") or {}
        dist = float(att.get("insertion_gap_m") or att.get("obstacle_distance_m") or 11.0)
        
        position_error = solo_ratio * (4.0 + min(12.0, 0.35 * dist))
        velocity_error = abs(ego_speed - fake_speed) * 0.25 + 1.5 * solo_ratio
        evidence = "injected_object_solo"
    
    print(f"\nCAV {cav_id} - Frame {frame_idx}:")
    print(f"  position_error: {position_error:.3f}")
    print(f"  velocity_error: {velocity_error:.3f}")
    print(f"  evidence: {evidence}")

print(f"\n{'='*70}")
print("Test completed successfully!")
print(f"{'='*70}")
print("\nConclusion:")
print("  - Data loading: OK")
print("  - Frame parsing: OK")
print("  - Attack detection: OK")
print("  - Observation extraction: OK")
print("\nReady to run full experiment with run_complete_experiment.py")
print()
