#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速验证脚本 - 验证改进的信誉系统核心功能
"""

import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from improved_reputation_engine import ImprovedReputationManager, ReputationConfig
    print("[OK] 成功导入改进的信誉引擎")
except ImportError as e:
    print(f"[ERROR] 导入失败: {e}")
    sys.exit(1)

# 测试核心功能
def quick_test():
    print("\n" + "="*60)
    print("快速功能验证")
    print("="*60)
    
    # 初始化
    config = ReputationConfig()
    manager = ImprovedReputationManager(config)
    
    # 验证配置
    print(f"\n1. EWMA配置:")
    print(f"   - alpha (新信息): {config.ewma_alpha}")
    print(f"   - beta (历史): {config.ewma_beta}")
    assert config.ewma_alpha == 0.3, "EWMA alpha应为0.3"
    assert config.ewma_beta == 0.7, "EWMA beta应为0.7"
    print("   [PASS] EWMA配置正确")
    
    print(f"\n2. 首次作恶惩罚配置:")
    print(f"   - 惩罚倍数: {config.first_offense_multiplier}")
    print(f"   - 高信誉阈值: {config.high_reputation_threshold}")
    assert config.first_offense_multiplier == 2.0, "惩罚倍数应为2.0"
    print("   [PASS] 首次作恶配置正确")
    
    print(f"\n3. 自适应步长配置:")
    print(f"   - 加速系数: {config.adaptive_count_factor}")
    assert config.adaptive_count_factor == 0.05, "加速系数应为0.05"
    print("   [PASS] 自适应步长配置正确")
    
    print(f"\n4. 多级过滤配置:")
    print(f"   - 软过滤阈值: {config.filter_threshold_soft}")
    print(f"   - 硬过滤阈值: {config.filter_threshold_hard}")
    assert config.filter_threshold_soft == 0.70, "软过滤应为0.70"
    assert config.filter_threshold_hard == 0.50, "硬过滤应为0.50"
    print("   [PASS] 多级过滤配置正确")
    
    # 测试信誉更新
    print(f"\n5. 信誉更新测试:")
    vehicle_id = "test_vehicle"
    
    # 连续异常
    for i in range(5):
        manager.update_from_evidence(
            vehicle_id,
            is_consistent=False,
            consistency_ratio=0.2,
            direct_trust=0.2
        )
    
    final_rep = manager.get_trust_score(vehicle_id)
    print(f"   - 5次异常后信誉: {final_rep:.4f}")
    assert final_rep < 0.35, f"5次异常后应<0.35，实际{final_rep:.4f}"
    print("   [PASS] 信誉下降速度正常")
    
    # 测试过滤权重
    print(f"\n6. 过滤权重测试:")
    weight = manager.get_filter_weight(vehicle_id)
    print(f"   - 当前信誉{final_rep:.4f}的过滤权重: {weight:.4f}")
    assert weight < 0.05, "低信誉权重应接近0"
    print("   [PASS] 过滤权重正确")
    
    print("\n" + "="*60)
    print("[SUCCESS] 所有核心功能验证通过！")
    print("="*60)
    print("\n下一步:")
    print("  1. 运行完整实验: python run_complete_experiment.py")
    print("  2. 生成可视化: python advanced_visualization.py")
    print("  3. 启动VEINS服务器: python reputation_socket_server.py")
    print()

if __name__ == "__main__":
    try:
        quick_test()
    except AssertionError as e:
        print(f"\n[FAIL] 验证失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
