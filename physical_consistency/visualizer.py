# visualizer.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
import os

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class VehicleMonitor:
    def __init__(self, save_path="results"):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        # 历史数据存储
        self.history = {
            'time': [],
            'reputation': {},  # vid -> list
            'fused_score': {}, # vid -> list
            'physical_score': {},
            'trajectory_score': {},
            'rsu_score': {},
            'attack_vehicles': set()
        }
        
        # 实时轨迹数据
        self.trajectories = {}  # vid -> deque of (x, y)
        
    def update(self, t, msgs, scores_dict, reputations, attack_vehicles):
        """更新所有历史数据"""
        self.history['time'].append(t)
        self.history['attack_vehicles'] = attack_vehicles
        
        for msg in msgs:
            vid = msg['vehicle_id']
            scores = scores_dict[vid]
            rep = reputations[vid]
            
            # 初始化列表
            if vid not in self.history['reputation']:
                self.history['reputation'][vid] = []
                self.history['fused_score'][vid] = []
                self.history['physical_score'][vid] = []
                self.history['trajectory_score'][vid] = []
                self.history['rsu_score'][vid] = []
                self.trajectories[vid] = deque(maxlen=50)  # 保留最近50个轨迹点
            
            # 记录数据
            self.history['reputation'][vid].append(rep)
            self.history['fused_score'][vid].append(scores['fused'])
            self.history['physical_score'][vid].append(scores['physical'])
            self.history['trajectory_score'][vid].append(scores['trajectory'])
            self.history['rsu_score'][vid].append(scores['rsu'])
            
            # 记录轨迹（真实位置）
            self.trajectories[vid].append((msg['pos'][0], msg['pos'][1]))
    
    def plot_scores_evolution(self):
        """绘制各类分数随时间演化"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('车辆信誉评估分数演化', fontsize=16, fontweight='bold')
        
        time = self.history['time']
        
        # 1. 融合分数演化
        ax = axes[0, 0]
        for vid, scores in self.history['fused_score'].items():
            is_attack = vid in self.history['attack_vehicles']
            color = 'red' if is_attack else 'blue'
            alpha = 0.8 if is_attack else 0.5
            ax.plot(time[:len(scores)], scores, label=vid, color=color, alpha=alpha, linewidth=1.5)
        ax.axhline(y=0.6, color='green', linestyle='--', linewidth=2, label='阈值(0.6)')
        ax.set_xlabel('时间步')
        ax.set_ylabel('融合分数')
        ax.set_title('融合分数演化 (红色=攻击车辆)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. 信誉值演化
        ax = axes[0, 1]
        for vid, reps in self.history['reputation'].items():
            is_attack = vid in self.history['attack_vehicles']
            color = 'red' if is_attack else 'blue'
            alpha = 0.8 if is_attack else 0.5
            ax.plot(time[:len(reps)], reps, label=vid, color=color, alpha=alpha, linewidth=1.5)
        ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='初始信誉')
        ax.set_xlabel('时间步')
        ax.set_ylabel('信誉值')
        ax.set_title('信誉值演化 (红色=攻击车辆)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 3. 各类分数对比（取平均值）
        ax = axes[1, 0]
        attack_scores = {'physical': [], 'trajectory': [], 'rsu': [], 'fused': []}
        normal_scores = {'physical': [], 'trajectory': [], 'rsu': [], 'fused': []}
        
        for vid in self.history['fused_score'].keys():
            is_attack = vid in self.history['attack_vehicles']
            target = attack_scores if is_attack else normal_scores
            
            target['physical'].append(np.mean(self.history['physical_score'][vid]))
            target['trajectory'].append(np.mean(self.history['trajectory_score'][vid]))
            target['rsu'].append(np.mean(self.history['rsu_score'][vid]))
            target['fused'].append(np.mean(self.history['fused_score'][vid]))
        
        x = np.arange(4)
        width = 0.35
        
        attack_means = [np.mean(attack_scores[k]) if attack_scores[k] else 0 for k in ['physical', 'trajectory', 'rsu', 'fused']]
        normal_means = [np.mean(normal_scores[k]) if normal_scores[k] else 0 for k in ['physical', 'trajectory', 'rsu', 'fused']]
        
        ax.bar(x - width/2, normal_means, width, label='正常车辆', color='blue', alpha=0.7)
        ax.bar(x + width/2, attack_means, width, label='攻击车辆', color='red', alpha=0.7)
        ax.set_xlabel('分数类型')
        ax.set_ylabel('平均分数')
        ax.set_title('正常vs攻击车辆各类分数对比')
        ax.set_xticks(x)
        ax.set_xticklabels(['物理', '轨迹', 'RSU', '融合'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 信誉分布直方图
        ax = axes[1, 1]
        final_reps = [self.history['reputation'][vid][-1] for vid in self.history['reputation'].keys()]
        attack_reps = [self.history['reputation'][vid][-1] for vid in self.history['reputation'].keys() 
                      if vid in self.history['attack_vehicles']]
        normal_reps = [self.history['reputation'][vid][-1] for vid in self.history['reputation'].keys() 
                      if vid not in self.history['attack_vehicles']]
        
        ax.hist(normal_reps, bins=20, alpha=0.7, label='正常车辆', color='blue', density=True)
        ax.hist(attack_reps, bins=20, alpha=0.7, label='攻击车辆', color='red', density=True)
        ax.axvline(x=0.6, color='green', linestyle='--', linewidth=2, label='可信阈值')
        ax.set_xlabel('最终信誉值')
        ax.set_ylabel('概率密度')
        ax.set_title('最终信誉值分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/scores_evolution.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_trajectory_map(self):
        """绘制车辆轨迹地图（区分正常/攻击）"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制轨迹
        for vid, trajectory in self.trajectories.items():
            if len(trajectory) < 2:
                continue
            
            traj_array = np.array(trajectory)
            is_attack = vid in self.history['attack_vehicles']
            
            color = 'red' if is_attack else 'blue'
            label = vid if is_attack else None
            alpha = 0.8 if is_attack else 0.4
            
            ax.plot(traj_array[:, 0], traj_array[:, 1], 
                   color=color, alpha=alpha, linewidth=1.5, label=label)
            
            # 标记起点和终点
            ax.scatter(traj_array[0, 0], traj_array[0, 1], 
                      color=color, s=100, marker='o', edgecolors='black', zorder=5)
            ax.scatter(traj_array[-1, 0], traj_array[-1, 1], 
                      color=color, s=100, marker='s', edgecolors='black', zorder=5)
            
            # 添加车辆标签
            if len(traj_array) > 0:
                ax.text(traj_array[-1, 0], traj_array[-1, 1], f'  {vid}', 
                       fontsize=8, alpha=0.7)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.5, label='正常车辆轨迹'),
            Patch(facecolor='red', alpha=0.5, label='攻击车辆轨迹'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                      markersize=8, label='起点'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black', 
                      markersize=8, label='终点')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_xlabel('X 坐标 (m)')
        ax.set_ylabel('Y 坐标 (m)')
        ax.set_title('车辆运动轨迹 (红色=攻击车辆)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/trajectory_map.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_detection_performance(self):
        """绘制检测性能指标"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. ROC曲线模拟
        ax = axes[0]
        thresholds = np.linspace(0, 1, 50)
        tpr = []
        fpr = []
        
        final_reps = {}
        for vid in self.history['reputation'].keys():
            final_reps[vid] = self.history['reputation'][vid][-1]
        
        for thresh in thresholds:
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            
            for vid, rep in final_reps.items():
                predicted_attack = rep < thresh
                actual_attack = vid in self.history['attack_vehicles']
                
                if actual_attack and predicted_attack:
                    tp += 1
                elif actual_attack and not predicted_attack:
                    fn += 1
                elif not actual_attack and predicted_attack:
                    fp += 1
                elif not actual_attack and not predicted_attack:
                    tn += 1
            
            tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, label='ROC曲线')
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='随机猜测')
        ax.set_xlabel('假正率 (FPR)')
        ax.set_ylabel('真正率 (TPR)')
        ax.set_title('ROC曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 计算AUC
        auc = np.trapz(tpr, fpr)
        ax.text(0.6, 0.2, f'AUC = {auc:.3f}', fontsize=12, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # 2. 检测性能指标
        ax = axes[1]
        
        # 计算最终检测结果（使用阈值0.6）
        thresh = 0.6
        tp = fp = tn = fn = 0
        for vid, rep in final_reps.items():
            predicted_attack = rep < thresh
            actual_attack = vid in self.history['attack_vehicles']
            
            if actual_attack and predicted_attack:
                tp += 1
            elif actual_attack and not predicted_attack:
                fn += 1
            elif not actual_attack and predicted_attack:
                fp += 1
            elif not actual_attack and not predicted_attack:
                tn += 1
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = ['准确率', '精确率', '召回率', 'F1分数']
        values = [accuracy, precision, recall, f1]
        colors = ['green', 'blue', 'orange', 'red']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.set_ylim([0, 1])
        ax.set_ylabel('分数')
        ax.set_title('检测性能指标')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/detection_performance.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_animation(self, interval=100):
        """创建实时动画"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 初始化
        lines = {}
        reputation_lines = {}
        
        def init():
            ax1.set_xlim(0, 20)
            ax1.set_ylim(0, 20)
            ax1.set_xlabel('X 坐标 (m)')
            ax1.set_ylabel('Y 坐标 (m)')
            ax1.set_title('实时车辆轨迹')
            ax1.grid(True, alpha=0.3)
            
            ax2.set_xlim(0, len(self.history['time']) or 100)
            ax2.set_ylim(0, 1)
            ax2.set_xlabel('时间步')
            ax2.set_ylabel('信誉值')
            ax2.set_title('信誉值实时演化')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0.6, color='green', linestyle='--', alpha=0.5)
            
            return []
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            
            # 更新轨迹图
            ax1.set_xlim(0, 20)
            ax1.set_ylim(0, 20)
            ax1.set_xlabel('X 坐标 (m)')
            ax1.set_ylabel('Y 坐标 (m)')
            ax1.set_title('实时车辆轨迹')
            ax1.grid(True, alpha=0.3)
            
            for vid, trajectory in self.trajectories.items():
                if len(trajectory) < 2:
                    continue
                traj_array = np.array(trajectory)
                is_attack = vid in self.history['attack_vehicles']
                color = 'red' if is_attack else 'blue'
                ax1.plot(traj_array[:, 0], traj_array[:, 1], 
                        color=color, alpha=0.6, linewidth=1)
                ax1.scatter(traj_array[-1, 0], traj_array[-1, 1], 
                           color=color, s=50, edgecolors='black')
            
            # 更新信誉图
            ax2.set_xlim(0, max(len(self.history['time']), 100))
            ax2.set_ylim(0, 1)
            ax2.set_xlabel('时间步')
            ax2.set_ylabel('信誉值')
            ax2.set_title('信誉值实时演化')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='可信阈值')
            
            for vid, reps in self.history['reputation'].items():
                if len(reps) > 0:
                    is_attack = vid in self.history['attack_vehicles']
                    color = 'red' if is_attack else 'blue'
                    alpha = 0.8 if is_attack else 0.4
                    time_steps = range(len(reps))
                    ax2.plot(time_steps, reps, label=vid if is_attack else "", 
                            color=color, alpha=alpha, linewidth=1)
            
            ax2.legend(loc='upper right', fontsize=8)
            
            return []
        
        ani = animation.FuncAnimation(fig, update, frames=range(len(self.history['time'])), 
                                     init_func=init, interval=interval, repeat=False)
        
        ani.save(f'{self.save_path}/animation.gif', writer='pillow', fps=10)
        plt.show()
        
        return ani
    
    def generate_report(self):
        """生成综合报告"""
        print("\n" + "="*60)
        print("车辆信誉评估系统 - 综合报告")
        print("="*60)
        
        # 最终信誉统计
        final_reps = {}
        for vid in self.history['reputation'].keys():
            final_reps[vid] = self.history['reputation'][vid][-1]
        
        attack_vehicles = self.history['attack_vehicles']
        normal_vehicles = [vid for vid in final_reps.keys() if vid not in attack_vehicles]
        
        print(f"\n车辆总数: {len(final_reps)}")
        print(f"攻击车辆数: {len(attack_vehicles)}")
        print(f"正常车辆数: {len(normal_vehicles)}")
        
        print(f"\n平均最终信誉值:")
        print(f"  攻击车辆: {np.mean([final_reps[vid] for vid in attack_vehicles]):.3f}")
        print(f"  正常车辆: {np.mean([final_reps[vid] for vid in normal_vehicles]):.3f}")
        
        # 检测结果
        thresh = 0.6
        detected_attacks = [vid for vid, rep in final_reps.items() if rep < thresh]
        
        tp = len([vid for vid in detected_attacks if vid in attack_vehicles])
        fp = len([vid for vid in detected_attacks if vid not in attack_vehicles])
        fn = len([vid for vid in attack_vehicles if vid not in detected_attacks])
        
        print(f"\n检测结果 (阈值={thresh}):")
        print(f"  正确检测攻击车辆: {tp}/{len(attack_vehicles)} ({tp/len(attack_vehicles)*100:.1f}%)")
        print(f"  误报正常车辆: {fp}/{len(normal_vehicles)} ({fp/len(normal_vehicles)*100:.1f}%)")
        
        # 详细车辆报告
        print("\n详细车辆报告:")
        print("-" * 60)
        print(f"{'车辆ID':<10} {'类型':<8} {'最终信誉':<10} {'最终融合分':<12} {'判定':<8}")
        print("-" * 60)
        
        for vid in sorted(final_reps.keys()):
            vtype = "攻击" if vid in attack_vehicles else "正常"
            rep = final_reps[vid]
            fused = self.history['fused_score'][vid][-1]
            verdict = "可疑" if rep < thresh else "可信"
            print(f"{vid:<10} {vtype:<8} {rep:<10.3f} {fused:<12.3f} {verdict:<8}")
        
        print("="*60)
        
        # 保存报告
        with open(f'{self.save_path}/report.txt', 'w', encoding='utf-8') as f:
            f.write("车辆信誉评估系统报告\n")
            f.write("="*60 + "\n")
            f.write(f"总车辆数: {len(final_reps)}\n")
            f.write(f"攻击车辆: {len(attack_vehicles)}\n")
            f.write(f"检测率: {tp/len(attack_vehicles)*100:.1f}%\n")
            f.write(f"误报率: {fp/len(normal_vehicles)*100:.1f}%\n")
        
