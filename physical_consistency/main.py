# main_visualized.py
from data import DataGenerator
from imm_manager import IMMManager
from intermediate_fusion_manager import IntermediateFusionManager
from visualizer import VehicleMonitor
import numpy as np

# 初始化模块
data_gen = DataGenerator(num_vehicles=20)
imm_manager = IMMManager()
fusion_manager = IntermediateFusionManager()
monitor = VehicleMonitor(save_path="results")  # 可视化监控器

# 存储每步的分数和信誉
all_scores = {}
all_reputations = {}

# 模拟100个时间步
for t in range(100):
    print(f"\n=== Time {t} ===")
    msgs = data_gen.step(t)
    
    # 第一步：预计算所有车辆的初始投票（用于RSU邻居分数）
    temp_votes = {}
    scores_dict = {}
    
    for msg in msgs:
        vid = msg["vehicle_id"]
        z = np.array([msg["pos"][0], msg["pos"][1]])
        residual, mu = imm_manager.step(vid, z)
        # 临时计算融合分数，生成初始投票
        scores = fusion_manager.compute_all_scores(residual, vid, msg["vel"])
        scores_dict[vid] = scores
        temp_votes[vid] = fusion_manager.get_vote(scores["fused"])
    
    # 第二步：更新邻居投票，最终计算分数+信誉+投票
    for msg in msgs:
        vid = msg["vehicle_id"]
        z = np.array([msg["pos"][0], msg["pos"][1]])
        residual, mu = imm_manager.step(vid, z)
        
        # 中间层融合：计算所有分数
        scores = fusion_manager.compute_all_scores(residual, vid, msg["vel"])
        scores_dict[vid] = scores
        
        # 更新邻居投票（模拟RSU收集周边投票）
        fusion_manager.update_neighbor_votes(vid, temp_votes[vid])
        
        # 更新最终信誉
        reputation = fusion_manager.update_reputation(vid, scores["fused"])
        
        # 存储结果
        all_reputations[vid] = reputation
        
        # 打印结果
        print(f"{vid}: "
              f"phy={scores['physical']:.2f}, "
              f"traj={scores['trajectory']:.2f}, "
              f"rsu={scores['rsu']:.2f}, "
              f"fused={scores['fused']:.2f}, "
              f"rep={reputation:.2f}, "
              f"vote={fusion_manager.get_vote(scores['fused'])}")
    
    # 更新可视化数据
    monitor.update(t, msgs, scores_dict, all_reputations, data_gen.attack_vehicles)

# 运行完成后生成可视化报告
print("\n" + "="*60)
print("模拟完成，生成可视化报告...")
print("="*60)

# 1. 绘制分数演化图
monitor.plot_scores_evolution()

# 2. 绘制轨迹地图
monitor.plot_trajectory_map()

# 3. 绘制检测性能
monitor.plot_detection_performance()

# 4. 创建动画（可选，可能较慢）
print("\n生成动画中...")
monitor.create_animation(interval=200)

# 5. 生成综合报告
monitor.generate_report()

print("\n所有结果已保存到 'results' 文件夹！")