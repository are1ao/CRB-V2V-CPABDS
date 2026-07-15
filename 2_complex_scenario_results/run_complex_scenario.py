"""
复杂场景测试 - 更接近真实V2V环境

新增挑战：
1. 动态恶意行为强度（时间变化）
2. 间歇性攻击（恶意车辆不总是作恶）
3. 多种攻击类型混合
4. 环境噪声动态变化
5. 车辆进出场景（动态拓扑）
6. 协同攻击（多车联合）
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from baseline_algorithms import create_all_baselines, BaselineComparison
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "complex_scenario_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ComplexScenarioGenerator:
    """复杂场景生成器"""
    
    def __init__(self, num_vehicles=50, num_malicious=10, duration=1000):
        self.num_vehicles = num_vehicles
        self.num_malicious = num_malicious
        self.duration = duration
        self.current_time = 0
        
        self.persistent_attackers = set()
        self.intermittent_attackers = set()
        self.coordinated_attackers = set()
        self.adaptive_attackers = set()
        
        self._initialize_attackers()
        
    def _initialize_attackers(self):
        """初始化不同类型的攻击者"""
        all_vehicles = list(range(self.num_vehicles))
        malicious_ids = np.random.choice(all_vehicles, self.num_malicious, replace=False)
        
        n = len(malicious_ids)
        self.persistent_attackers = set(malicious_ids[:n//4])
        self.intermittent_attackers = set(malicious_ids[n//4:n//2])
        self.coordinated_attackers = set(malicious_ids[n//2:3*n//4])
        self.adaptive_attackers = set(malicious_ids[3*n//4:])
        
    def get_all_malicious(self):
        """获取所有恶意车辆ID"""
        return (self.persistent_attackers | self.intermittent_attackers | 
                self.coordinated_attackers | self.adaptive_attackers)
    
    def get_attacker_type(self, vehicle_id):
        """获取攻击者类型"""
        if vehicle_id in self.persistent_attackers:
            return 'persistent'
        elif vehicle_id in self.intermittent_attackers:
            return 'intermittent'
        elif vehicle_id in self.coordinated_attackers:
            return 'coordinated'
        elif vehicle_id in self.adaptive_attackers:
            return 'adaptive'
        return 'benign'
    
    def is_attacking(self, vehicle_id, time_step):
        """判断车辆在当前时刻是否发起攻击"""
        if vehicle_id in self.persistent_attackers:
            return True
        elif vehicle_id in self.intermittent_attackers:
            return np.random.random() < 0.3
        elif vehicle_id in self.coordinated_attackers:
            return (time_step % 50) < 10
        elif vehicle_id in self.adaptive_attackers:
            attack_prob = max(0.1, 0.8 - time_step / self.duration * 0.7)
            return np.random.random() < attack_prob
        return False
    
    def get_attack_intensity(self, vehicle_id, time_step, detection_history):
        """获取攻击强度（自适应攻击者会降低强度）"""
        base_intensity = 1.0
        
        if vehicle_id in self.adaptive_attackers:
            if detection_history.get(vehicle_id, 0) > 3:
                base_intensity = 0.4
        
        time_factor = 1.0 + 0.5 * np.sin(2 * np.pi * time_step / 200)
        return base_intensity * time_factor
    
    def get_noise_level(self, time_step):
        """动态噪声水平（模拟环境变化）"""
        base_noise = 0.2
        periodic_noise = 0.15 * np.sin(2 * np.pi * time_step / 100)
        random_spike = 0.3 if np.random.random() < 0.05 else 0.0
        return base_noise + periodic_noise + random_spike
    
    def generate_observation(self, vehicle_id, time_step, detection_history):
        """生成单个观测"""
        is_attacking = self.is_attacking(vehicle_id, time_step)
        
        if is_attacking:
            intensity = self.get_attack_intensity(vehicle_id, time_step, detection_history)
            attack_type = np.random.choice(['position', 'velocity', 'timestamp', 'mixed'])
            
            if attack_type == 'position':
                pos_err = np.random.uniform(3, 8) * intensity
                vel_err = np.random.uniform(0, 1)
                ts_err = np.random.uniform(0, 0.2)
            elif attack_type == 'velocity':
                pos_err = np.random.uniform(0, 1)
                vel_err = np.random.uniform(2, 6) * intensity
                ts_err = np.random.uniform(0, 0.2)
            elif attack_type == 'timestamp':
                pos_err = np.random.uniform(0, 1)
                vel_err = np.random.uniform(0, 1)
                ts_err = np.random.uniform(1, 3) * intensity
            else:
                pos_err = np.random.uniform(2, 5) * intensity
                vel_err = np.random.uniform(1.5, 4) * intensity
                ts_err = np.random.uniform(0.5, 2) * intensity
            
            freq = np.random.uniform(6, 9)
        else:
            pos_err = np.random.uniform(0, 0.8)
            vel_err = np.random.uniform(0, 0.4)
            ts_err = np.random.uniform(0, 0.1)
            freq = np.random.uniform(9.5, 10.5)
        
        noise = self.get_noise_level(time_step)
        pos_err += np.random.normal(0, noise)
        vel_err += np.random.normal(0, noise * 0.5)
        ts_err += np.random.normal(0, noise * 0.3)
        
        return {
            'vehicle_id': str(vehicle_id),
            'time_step': time_step,
            'position_error': max(0, pos_err),
            'velocity_error': max(0, vel_err),
            'timestamp_error': max(0, ts_err),
            'message_frequency': max(0, freq),
            'is_attacking': is_attacking,
            'is_malicious': vehicle_id in self.get_all_malicious()
        }
    
    def generate_scenario(self):
        """生成完整场景"""
        observations = []
        detection_history = defaultdict(int)
        
        print(f"生成复杂场景: {self.num_vehicles}辆车, {self.duration}个时间步")
        print(f"恶意车辆分布:")
        print(f"  - 持续攻击: {len(self.persistent_attackers)}辆")
        print(f"  - 间歇攻击: {len(self.intermittent_attackers)}辆")
        print(f"  - 协同攻击: {len(self.coordinated_attackers)}辆")
        print(f"  - 自适应攻击: {len(self.adaptive_attackers)}辆")
        
        for t in range(self.duration):
            if t % 100 == 0:
                print(f"  进度: {t}/{self.duration}")
            
            active_vehicles = list(range(self.num_vehicles))
            if np.random.random() < 0.1:
                active_vehicles = np.random.choice(
                    active_vehicles, 
                    int(len(active_vehicles) * 0.8), 
                    replace=False
                )
            
            for vid in active_vehicles:
                obs = self.generate_observation(vid, t, detection_history)
                observations.append(obs)
        
        print(f"生成完成: {len(observations)}条观测")
        return observations


def run_complex_scenario():
    """运行复杂场景测试"""
    print("\n" + "="*60)
    print("复杂场景测试")
    print("="*60)
    
    scenario = ComplexScenarioGenerator(
        num_vehicles=50,
        num_malicious=10,
        duration=500
    )
    observations = scenario.generate_scenario()
    malicious_ids = {str(vid) for vid in scenario.get_all_malicious()}
    
    print("\n" + "="*60)
    print("运行基线算法")
    print("="*60)
    
    comparison = BaselineComparison()
    timing_results = {}
    
    for name, algo in create_all_baselines().items():
        comparison.add_algorithm(algo)
        print(f"[ADD] {name}")
    
    vehicle_ids = list(set(obs['vehicle_id'] for obs in observations))
    comparison.initialize_all(vehicle_ids)
    
    time_series_metrics = defaultdict(lambda: defaultdict(list))
    
    for algo_name, algo in comparison.algorithms.items():
        start_time = time.time()
        
        for i, obs in enumerate(observations):
            algo.update_reputation(obs['vehicle_id'], obs)
            
            if (i + 1) % 5000 == 0:
                tp = fp = tn = fn = 0
                for vid, rep in algo.get_all_reputations().items():
                    detected = rep < 0.3
                    malicious = vid in malicious_ids
                    if detected and malicious: tp += 1
                    elif detected and not malicious: fp += 1
                    elif not detected and not malicious: tn += 1
                    else: fn += 1
                
                f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0
                time_series_metrics[algo_name]['time_step'].append(i+1)
                time_series_metrics[algo_name]['f1'].append(f1)
                time_series_metrics[algo_name]['tp'].append(tp)
                time_series_metrics[algo_name]['fp'].append(fp)
                time_series_metrics[algo_name]['fn'].append(fn)
        
        elapsed = time.time() - start_time
        timing_results[algo_name] = elapsed
        print(f"[TIME] {algo_name}: {elapsed:.4f}秒")
    
    print("\n" + "="*60)
    print("最终性能评估")
    print("="*60)
    
    results = []
    for name, algo in comparison.algorithms.items():
        tp = fp = tn = fn = 0
        for vid, rep in algo.get_all_reputations().items():
            detected = rep < 0.3
            malicious = vid in malicious_ids
            if detected and malicious: tp += 1
            elif detected and not malicious: fp += 1
            elif not detected and not malicious: tn += 1
            else: fn += 1
        
        acc = (tp+tn)/(tp+fp+tn+fn) if (tp+fp+tn+fn)>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        
        results.append({
            'Algorithm': name,
            'F1_Score': f1,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'Processing_Time': timing_results[name]
        })
        print(f"{name:17s} F1={f1:.3f}, Acc={acc:.2%}, Prec={prec:.2%}, Rec={rec:.2%}")
    
    perf_df = pd.DataFrame(results)
    perf_df.to_csv(os.path.join(OUTPUT_DIR, "complex_scenario_results.csv"), index=False)
    
    from complex_scenario_viz import generate_complex_visualizations, print_summary
    generate_complex_visualizations(perf_df, time_series_metrics, scenario)
    print_summary(perf_df, scenario)
    
    return perf_df, time_series_metrics


if __name__ == "__main__":
    print("\n" + "="*60)
    print("复杂V2V场景测试")
    print("="*60)
    print("[INFO] 输出目录:", OUTPUT_DIR)
    
    run_complex_scenario()
    
    print("\n" + "="*60)
    print("[SUCCESS] 复杂场景测试完成！")
    print("="*60)
    print(f"\n[输出文件]:")
    print(f"  - 性能数据: {OUTPUT_DIR}/complex_scenario_results.csv")
    print(f"  - 可视化图表: {OUTPUT_DIR}/figures/")
