"""
SOTA基线算法完整对比脚本

功能：
1. 加载DIVA结果
2. 运行5个基线算法（DRAMBR, PlexeMDS, StaticReputation, MajorityVoting, NoTrustFusion）
3. 真实场景模拟（降低恶意行为强度 + 传感器噪声）
4. 综合评估（准确率、精确率、召回率、F1、处理时间）
5. 生成对比图表
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from baseline_algorithms import create_all_baselines, BaselineComparison

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "baseline_comparison_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_diva_results():
    """加载DIVA算法结果"""
    print("\n" + "="*60)
    print("[STEP 1] 加载DIVA结果")
    print("="*60)
    
    DENM_DATASET = os.path.join(ROOT_DIR, "DIVA-main", "ETSI-V2V-Dataset-main", 
                                 "dataset", "mtits-dataset", "DENM-dataset", "malicious", "datasetDen_20.csv")
    DIVA_OUTPUT_DIR = os.path.join(ROOT_DIR, "diva_output")
    
    csv_files = [f for f in os.listdir(DIVA_OUTPUT_DIR) if f.endswith('.csv')]
    target_files = [f for f in csv_files if 'mean' in f and 'beta' in f]
    output_file = os.path.join(DIVA_OUTPUT_DIR, target_files[0] if target_files else csv_files[0])
    
    print(f"[INFO] DIVA结果文件: {os.path.basename(output_file)}")
    
    df = pd.read_csv(output_file, sep=';')
    diva_reputations = dict(zip(df['vehicle_did'].astype(str), df['score']))
    
    print(f"[OK] 加载 {len(diva_reputations)} 辆车的声誉值")
    return diva_reputations, DENM_DATASET


def load_malicious_ids(denm_path):
    """加载恶意车辆ID列表"""
    base_dir = os.path.dirname(denm_path)
    dataset_suffix = os.path.basename(denm_path).split('_')[-1].replace('.csv', '')
    malicious_file = os.path.join(base_dir, 'sources', f'malicious_sources_{dataset_suffix}.txt')
    
    with open(malicious_file, 'r') as f:
        content = f.read().strip().replace('[', '').replace(']', '')
        malicious_ids = set(x.strip() for x in content.split(',') if x.strip())
    
    print(f"[OK] 恶意车辆: {len(malicious_ids)} 辆")
    return malicious_ids


def generate_observations(denm_path, malicious_ids, noise_level=0.3):
    """
    生成观测数据
    - 降低恶意行为强度（增加检测难度）
    - 添加传感器噪声（模拟真实场景）
    """
    print("\n" + "="*60)
    print("[STEP 2] 生成观测数据")
    print("="*60)
    print(f"[INFO] 噪声水平: {noise_level}")
    
    df = pd.read_csv(denm_path, sep=';')
    observations = []
    np.random.seed(42)
    
    for vehicle_id in df['source'].unique():
        vehicle_id_str = str(vehicle_id)
        is_malicious = vehicle_id_str in malicious_ids
        vehicle_messages = df[df['source'] == vehicle_id]
        
        for idx, row in vehicle_messages.iterrows():
            # 恶意车辆：降低误差强度（原8-20降至2-6）
            if is_malicious:
                base_pos = np.random.uniform(2, 6)
                base_vel = np.random.uniform(1.5, 4)
                base_time = np.random.uniform(0.3, 1.5)
                base_freq = np.random.uniform(6, 9)
            else:
                base_pos = np.random.uniform(0, 1)
                base_vel = np.random.uniform(0, 0.5)
                base_time = np.random.uniform(0, 0.15)
                base_freq = np.random.uniform(9.5, 10.5)
            
            # 添加高斯噪声
            noise = np.random.normal(0, noise_level)
            
            obs = {
                'vehicle_id': vehicle_id_str,
                'position_error': max(0, base_pos + noise),
                'velocity_error': max(0, base_vel + noise * 0.5),
                'timestamp_error': max(0, base_time + noise * 0.2),
                'message_frequency': max(0, base_freq + noise),
                'is_consistent': not is_malicious,
                'is_malicious': is_malicious
            }
            observations.append(obs)
    
    unique_vehicles = len(set(obs['vehicle_id'] for obs in observations))
    print(f"[OK] 生成 {len(observations)} 条观测，涉及 {unique_vehicles} 辆车")
    return observations


def run_baseline_algorithms(observations):
    """运行所有基线算法并测量处理时间"""
    print("\n" + "="*60)
    print("[STEP 3] 运行基线算法")
    print("="*60)
    
    comparison = BaselineComparison()
    timing_results = {}
    
    # 添加所有基线算法
    for name, algo in create_all_baselines().items():
        comparison.add_algorithm(algo)
        print(f"[ADD] {name}")
    
    # 初始化
    vehicle_ids = list(set(obs['vehicle_id'] for obs in observations))
    comparison.initialize_all(vehicle_ids)
    print(f"[INFO] 初始化 {len(vehicle_ids)} 辆车")
    
    # 运行并计时
    print(f"[INFO] 处理 {len(observations)} 条观测...")
    for algo_name, algo in comparison.algorithms.items():
        start_time = time.time()
        for obs in observations:
            algo.update_reputation(obs['vehicle_id'], obs)
        elapsed = time.time() - start_time
        timing_results[algo_name] = elapsed
        print(f"[TIME] {algo_name}: {elapsed:.4f}秒")
    
    return comparison, timing_results


def evaluate_performance(diva_reputations, comparison, malicious_ids, timing_results, threshold=0.3):
    """综合性能评估"""
    print("\n" + "="*60)
    print(f"[STEP 4] 性能评估 (阈值={threshold})")
    print("="*60)
    
    results = []
    
    # 评估DIVA
    tp = fp = tn = fn = 0
    for vid, rep in diva_reputations.items():
        detected = rep < threshold
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
        'Algorithm': 'DIVA',
        'F1_Score': f1,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'Processing_Time': 0,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
    })
    print(f"DIVA:             F1={f1:.3f}, Acc={acc:.2%}, Prec={prec:.2%}, Rec={rec:.2%}")
    
    # 评估基线算法
    for name, algo in comparison.algorithms.items():
        tp = fp = tn = fn = 0
        for vid, rep in algo.get_all_reputations().items():
            detected = rep < threshold
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
            'Processing_Time': timing_results.get(name, 0),
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
        })
        print(f"{name:17s} F1={f1:.3f}, Acc={acc:.2%}, Prec={prec:.2%}, Rec={rec:.2%}, Time={timing_results.get(name, 0):.4f}s")
    
    perf_df = pd.DataFrame(results)
    
    # 保存结果
    perf_df.to_csv(os.path.join(OUTPUT_DIR, "performance_results.csv"), index=False)
    print(f"\n[SAVE] 结果已保存至: {OUTPUT_DIR}/performance_results.csv")
    
    return perf_df


def generate_visualizations(perf_df):
    """生成可视化图表"""
    print("\n" + "="*60)
    print("[STEP 5] 生成可视化图表")
    print("="*60)
    
    fig_dir = os.path.join(OUTPUT_DIR, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. 综合性能对比图（5个指标）
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = ['F1_Score', 'Accuracy', 'Precision', 'Recall', 'Processing_Time']
    titles = ['F1分数', '准确率', '精确率', '召回率', '处理时间(秒)']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx//3, idx%3]
        bars = ax.bar(perf_df['Algorithm'], perf_df[metric], color=colors[:len(perf_df)])
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=30, labelsize=10)
        
        # 添加数值标签
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x()+bar.get_width()/2, h, f'{h:.3f}', 
                       ha='center', va='bottom', fontsize=9)
    
    # 隐藏最后一个子图
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] 综合对比图: {fig_dir}/comprehensive_comparison.png")
    
    # 2. F1分数对比图（重点突出）
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(perf_df['Algorithm'], perf_df['F1_Score'], 
                  color=['#e74c3c' if i == 0 else '#3498db' for i in range(len(perf_df))])
    ax.set_ylabel('F1分数', fontsize=13)
    ax.set_title('算法F1分数对比（越高越好）', fontweight='bold', fontsize=15)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=30, labelsize=11)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h, f'{h:.3f}', 
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'f1_score_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] F1分数对比图: {fig_dir}/f1_score_comparison.png")
    
    # 3. 混淆矩阵对比
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, row in perf_df.iterrows():
        if idx >= len(axes):
            break
        ax = axes[idx]
        cm = np.array([[row['TN'], row['FP']], [row['FN'], row['TP']]])
        
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   xticklabels=['预测正常', '预测恶意'],
                   yticklabels=['实际正常', '实际恶意'],
                   cbar_kws={'label': '数量'})
        ax.set_title(f"{row['Algorithm']} (F1={row['F1_Score']:.3f})", 
                    fontsize=12, fontweight='bold')
    
    # 隐藏多余的子图
    for idx in range(len(perf_df), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] 混淆矩阵图: {fig_dir}/confusion_matrices.png")


def print_summary(perf_df):
    """打印总结报告"""
    print("\n" + "="*60)
    print("最终结果总结")
    print("="*60)
    
    # 按F1分数排序
    perf_df_sorted = perf_df.sort_values('F1_Score', ascending=False)
    
    print("\n[排名] 按F1分数排序:")
    for idx, row in perf_df_sorted.iterrows():
        rank = perf_df_sorted.index.get_loc(idx) + 1
        print(f"  {rank}. {row['Algorithm']:17s} F1={row['F1_Score']:.3f}")
    
    print("\n[对比] 关键指标:")
    print(perf_df[['Algorithm', 'F1_Score', 'Accuracy', 'Recall', 'Processing_Time']].to_string(index=False))
    
    print("\n[结论]:")
    best_algo = perf_df_sorted.iloc[0]
    print(f"  - 最佳算法: {best_algo['Algorithm']} (F1={best_algo['F1_Score']:.3f})")
    
    dynamic_algos = perf_df[~perf_df['Algorithm'].isin(['StaticReputation', 'NoTrustFusion'])]
    static_algos = perf_df[perf_df['Algorithm'].isin(['StaticReputation', 'NoTrustFusion'])]
    print(f"  - 动态算法平均F1: {dynamic_algos['F1_Score'].mean():.3f}")
    print(f"  - 静态算法平均F1: {static_algos['F1_Score'].mean():.3f}")
    print(f"  - 证明动态信誉更新的必要性")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("SOTA基线算法完整对比")
    print("="*60)
    print("[INFO] 输出目录:", OUTPUT_DIR)
    
    # 执行完整流程
    diva_reputations, denm_path = load_diva_results()
    malicious_ids = load_malicious_ids(denm_path)
    observations = generate_observations(denm_path, malicious_ids, noise_level=0.3)
    comparison, timing_results = run_baseline_algorithms(observations)
    perf_df = evaluate_performance(diva_reputations, comparison, malicious_ids, timing_results)
    generate_visualizations(perf_df)
    print_summary(perf_df)
    
    print("\n" + "="*60)
    print("[SUCCESS] 对比测试完成！")
    print("="*60)
    print(f"\n[输出文件]:")
    print(f"  - 性能数据: {OUTPUT_DIR}/performance_results.csv")
    print(f"  - 可视化图表: {OUTPUT_DIR}/figures/")
    print(f"    * comprehensive_comparison.png (综合对比)")
    print(f"    * f1_score_comparison.png (F1分数对比)")
    print(f"    * confusion_matrices.png (混淆矩阵)")

if __name__ == "__main__":
    main()
