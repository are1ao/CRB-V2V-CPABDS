# -*- coding: utf-8 -*-
"""
测试改进效果脚本
验证所有6项改进是否正确实施并生效
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from improved_reputation_engine import (
    ImprovedReputationManager, 
    ReputationConfig,
    ImprovedPredictiveReputationModel
)

def test_ewma_smoothing():
    """测试1: EWMA平滑系数改进"""
    print("\n" + "="*60)
    print("测试1: EWMA平滑系数 (0.7*old + 0.3*new)")
    print("="*60)
    
    config = ReputationConfig()
    assert config.ewma_alpha == 0.3, "EWMA alpha应为0.3"
    assert config.ewma_beta == 0.7, "EWMA beta应为0.7"
    
    manager = ImprovedReputationManager(config)
    vehicle_id = "test_vehicle"
    
    # 初始信誉0.5，连续5次不一致
    initial = manager.get_trust_score(vehicle_id)
    print(f"初始信誉: {initial:.4f}")
    
    for i in range(5):
        manager.update_from_evidence(
            vehicle_id,
            is_consistent=False,
            consistency_ratio=0.2,
            direct_trust=0.2
        )
        score = manager.get_trust_score(vehicle_id)
        print(f"第{i+1}次异常后: {score:.4f}")
    
    final = manager.get_trust_score(vehicle_id)
    assert final < 0.35, f"5次异常后信誉应<0.35，实际{final:.4f}"
    
    print("[PASS] EWMA平滑系数测试通过")
    return True


def test_first_offense_penalty():
    """测试2: 首次作恶放大惩罚"""
    print("\n" + "="*60)
    print("测试2: 首次作恶放大惩罚 (×2)")
    print("="*60)
    
    config = ReputationConfig()
    manager = ImprovedReputationManager(config)
    vehicle_id = "high_rep_vehicle"
    
    # 建立高信誉（>0.85）并维持30+帧
    for i in range(40):
        manager.update_from_evidence(
            vehicle_id,
            is_consistent=True,
            consistency_ratio=1.0,
            direct_trust=1.0
        )
    
    rep_before = manager.get_trust_score(vehicle_id)
    print(f"建立高信誉: {rep_before:.4f} (持续40帧)")
    assert rep_before > 0.85, f"应建立高信誉>0.85，实际{rep_before:.4f}"
    
    # 首次作恶
    result = manager.update_from_evidence(
        vehicle_id,
        is_consistent=False,
        consistency_ratio=0.2,
        direct_trust=0.2
    )
    
    rep_after = manager.get_trust_score(vehicle_id)
    first_offense = result.get('first_offense', False)
    drop = rep_before - rep_after
    
    print(f"首次作恶触发: {first_offense}")
    print(f"首次作恶后: {rep_after:.4f}")
    print(f"信誉下降: {drop:.4f}")
    
    assert first_offense, "应触发首次作恶惩罚"
    assert drop > 0.15, f"首次作恶下降应>0.15，实际{drop:.4f}"
    
    print("[PASS] 首次作恶惩罚测试通过")
    return True


def test_lstm_prediction():
    """测试3: LSTM预测性信誉"""
    print("\n" + "="*60)
    print("测试3: LSTM预测性信誉")
    print("="*60)
    
    predictor = ImprovedPredictiveReputationModel(
        window=10,
        deviation_threshold=0.15,
        enable_debug=True
    )
    
    vehicle_id = "test_lstm"
    
    # 记录10个观测
    for i in range(10):
        features = np.array([0.8, 0.75, 0.7, 0.2])  # 正常行为
        predictor.record_observation(vehicle_id, features)
    
    # 预测下一个
    prediction = predictor.predict_next(vehicle_id)
    print(f"预测值: {prediction:.4f}" if prediction else "预测失败")
    
    # 实际值突然下降（异常）
    actual = 0.3
    is_trigger, deviation = predictor.check_deviation(vehicle_id, actual)
    
    print(f"实际值: {actual:.4f}")
    print(f"偏差: {deviation:.4f}")
    print(f"触发预警: {is_trigger}")
    
    stats = predictor.get_statistics()
    print(f"统计: {stats}")
    
    assert stats['total_predictions'] > 0, "应有预测记录"
    
    print("[PASS] LSTM预测测试通过")
    return True


def test_adaptive_step_acceleration():
    """测试4: 自适应步长加速"""
    print("\n" + "="*60)
    print("测试4: 自适应步长加速 (count_factor=0.05)")
    print("="*60)
    
    config = ReputationConfig()
    assert config.adaptive_count_factor == 0.05, "加速系数应为0.05"
    
    manager = ImprovedReputationManager(config)
    vehicle_id = "adaptive_test"
    
    # 连续10次异常
    drops = []
    for i in range(10):
        old = manager.get_trust_score(vehicle_id)
        manager.update_from_evidence(
            vehicle_id,
            is_consistent=False,
            consistency_ratio=0.2,
            direct_trust=0.2
        )
        new = manager.get_trust_score(vehicle_id)
        drop = old - new
        drops.append(drop)
        print(f"第{i+1}次: 下降{drop:.4f}")
    
    # 后期下降应大于前期（加速效应）
    early_avg = np.mean(drops[:3])
    late_avg = np.mean(drops[7:10])
    print(f"前期平均下降: {early_avg:.4f}")
    print(f"后期平均下降: {late_avg:.4f}")
    print(f"加速倍率: {late_avg/early_avg:.2f}x")
    
    assert late_avg > early_avg * 1.3, "后期下降应>前期1.3倍"
    
    print("[PASS] 自适应步长加速测试通过")
    return True


def test_multi_level_filtering():
    """测试5: 多级过滤策略"""
    print("\n" + "="*60)
    print("测试5: 多级过滤策略")
    print("="*60)
    
    config = ReputationConfig()
    manager = ImprovedReputationManager(config)
    
    # 测试不同信誉值的过滤权重
    test_cases = [
        ("high_rep", 0.85, 1.0),      # 高信誉：完整权重
        ("medium_rep", 0.60, 0.3),    # 中等：降权30%
        ("low_rep", 0.40, 0.0),       # 低信誉：完全排除
    ]
    
    for vid, target_rep, expected_min in test_cases:
        # 设置信誉值
        for _ in range(50):
            manager.update_from_evidence(
                vid,
                is_consistent=(target_rep > 0.5),
                consistency_ratio=target_rep,
                direct_trust=target_rep
            )
        
        # 微调到目标值
        manager._get_meta(vid).score = target_rep
        
        weight = manager.get_filter_weight(vid)
        print(f"信誉{target_rep:.2f} -> 权重{weight:.2f} (预期>={expected_min:.2f})")
        
        if target_rep >= 0.70:
            assert weight >= 0.9, f"高信誉权重应≥0.9"
        elif target_rep >= 0.50:
            assert 0.0 <= weight <= 0.35, f"中等信誉权重应在0-0.35"
        else:
            assert weight < 0.05, f"低信誉权重应≈0"
    
    print("[PASS] 多级过滤测试通过")
    return True


def test_detection_speed_comparison():
    """测试6: 检测速度对比"""
    print("\n" + "="*60)
    print("测试6: 检测速度对比 (改进 vs 基线)")
    print("="*60)
    
    # 改进版
    config_improved = ReputationConfig()
    manager_improved = ImprovedReputationManager(config_improved)
    
    # 基线版（模拟旧参数）
    config_baseline = ReputationConfig()
    config_baseline.ewma_alpha = 0.15
    config_baseline.ewma_beta = 0.85
    config_baseline.first_offense_multiplier = 1.0
    config_baseline.adaptive_count_factor = 0.01
    manager_baseline = ImprovedReputationManager(config_baseline)
    
    vehicle_id = "attacker"
    
    # 模拟攻击：连续异常
    detection_frame_improved = None
    detection_frame_baseline = None
    
    for frame in range(50):
        # 改进版
        manager_improved.update_from_evidence(
            vehicle_id,
            is_consistent=False,
            consistency_ratio=0.2,
            direct_trust=0.2
        )
        rep_imp = manager_improved.get_trust_score(vehicle_id)
        if rep_imp < 0.50 and detection_frame_improved is None:
            detection_frame_improved = frame + 1
        
        # 基线版
        manager_baseline.update_from_evidence(
            vehicle_id,
            is_consistent=False,
            consistency_ratio=0.2,
            direct_trust=0.2
        )
        rep_base = manager_baseline.get_trust_score(vehicle_id)
        if rep_base < 0.50 and detection_frame_baseline is None:
            detection_frame_baseline = frame + 1
        
        if frame % 10 == 0:
            print(f"第{frame}帧 - 改进版:{rep_imp:.3f}, 基线版:{rep_base:.3f}")
    
    print(f"\n检测延迟:")
    print(f"  改进版: {detection_frame_improved}帧")
    print(f"  基线版: {detection_frame_baseline}帧")
    
    if detection_frame_improved and detection_frame_baseline:
        speedup = detection_frame_baseline / detection_frame_improved
        print(f"  加速倍率: {speedup:.2f}x")
        assert speedup > 1.5, "改进版应比基线快1.5倍以上"
    
    print("[PASS] 检测速度对比测试通过")
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("增强信誉系统 - 改进效果验证测试")
    print("="*60)
    
    tests = [
        ("EWMA平滑系数", test_ewma_smoothing),
        ("首次作恶惩罚", test_first_offense_penalty),
        ("LSTM预测", test_lstm_prediction),
        ("自适应步长加速", test_adaptive_step_acceleration),
        ("多级过滤", test_multi_level_filtering),
        ("检测速度对比", test_detection_speed_comparison),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except AssertionError as e:
            print(f"[FAIL] {name}测试失败: {e}")
            results.append((name, False))
        except Exception as e:
            print(f"[ERROR] {name}测试错误: {e}")
            results.append((name, False))
    
    # 汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n[SUCCESS] 所有测试通过！改进已正确实施。")
    else:
        print("\n[WARNING] 部分测试失败，请检查实现。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
