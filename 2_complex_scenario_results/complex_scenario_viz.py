"""
复杂场景可视化模块 - 增强版
包含所有基线算法的全面对比：F1、准确率、召回率、精确率、误报率等
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "complex_scenario_results")


def generate_all_metrics_comparison(perf_df):
    """生成所有性能指标的全面对比图"""
    fig_dir = os.path.join(OUTPUT_DIR, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    
    algos = perf_df['Algorithm'].tolist()
    colors_map = {
        'DRAMBR': '#3498db',
        'PlexeMDS': '#e74c3c',
        'MajorityVoting': '#2ecc71',
        'EnhancedDRAMBR': '#9b59b6',
        'StaticReputation': '#95a5a6',
        'NoTrustFusion': '#7f8c8d'
    }
    colors = [colors_map.get(a, '#34495e') for a in algos]
    
    ax = fig.add_subplot(gs[0, 0])
    f1_scores = perf_df['F1_Score'].tolist()
    bars = ax.bar(algos, f1_scores, color=colors)
    ax.set_ylabel('F1分数', fontsize=12)
    ax.set_title('F1分数对比', fontweight='bold', fontsize=13)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    for bar, f1 in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, f1, f'{f1:.3f}',
               ha='center', va='bottom', fontsize=9)
    
    ax = fig.add_subplot(gs[0, 1])
    accuracy = perf_df['Accuracy'].tolist()
    bars = ax.bar(algos, accuracy, color=colors)
    ax.set_ylabel('准确率', fontsize=12)
    ax.set_title('准确率对比', fontweight='bold', fontsize=13)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, acc, f'{acc:.3f}',
               ha='center', va='bottom', fontsize=9)
    
    ax = fig.add_subplot(gs[0, 2])
    recall = perf_df['Recall'].tolist()
    bars = ax.bar(algos, recall, color=colors)
    ax.set_ylabel('召回率', fontsize=12)
    ax.set_title('召回率对比', fontweight='bold', fontsize=13)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    for bar, rec in zip(bars, recall):
        ax.text(bar.get_x() + bar.get_width()/2, rec, f'{rec:.3f}',
               ha='center', va='bottom', fontsize=9)
    
    ax = fig.add_subplot(gs[1, 0])
    precision = perf_df['Precision'].tolist()
    bars = ax.bar(algos, precision, color=colors)
    ax.set_ylabel('精确率', fontsize=12)
    ax.set_title('精确率对比', fontweight='bold', fontsize=13)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    for bar, prec in zip(bars, precision):
        if prec > 0:
            ax.text(bar.get_x() + bar.get_width()/2, prec, f'{prec:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax = fig.add_subplot(gs[1, 1])
    times = perf_df['Processing_Time'].tolist()
    bars = ax.bar(algos, times, color=colors)
    ax.set_ylabel('处理时间(秒)', fontsize=12)
    ax.set_title('计算效率对比', fontweight='bold', fontsize=13)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.grid(axis='y', alpha=0.3)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, t, f'{t:.2f}s',
               ha='center', va='bottom', fontsize=9)
    
    ax = fig.add_subplot(gs[1, 2])
    metrics_data = perf_df[['Algorithm', 'F1_Score', 'Accuracy', 'Precision', 'Recall']].set_index('Algorithm')
    dynamic_algos = metrics_data[~metrics_data.index.isin(['StaticReputation', 'NoTrustFusion'])]
    x = np.arange(len(dynamic_algos))
    width = 0.2
    
    ax.bar(x - 1.5*width, dynamic_algos['F1_Score'], width, label='F1', color='#3498db')
    ax.bar(x - 0.5*width, dynamic_algos['Accuracy'], width, label='准确率', color='#2ecc71')
    ax.bar(x + 0.5*width, dynamic_algos['Precision'], width, label='精确率', color='#f39c12')
    ax.bar(x + 1.5*width, dynamic_algos['Recall'], width, label='召回率', color='#e74c3c')
    
    ax.set_ylabel('数值', fontsize=12)
    ax.set_title('动态算法综合指标对比', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(dynamic_algos.index, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    ax = fig.add_subplot(gs[2, :2])
    tp_vals = perf_df['TP'].tolist()
    fp_vals = perf_df['FP'].tolist()
    fn_vals = perf_df['FN'].tolist()
    
    x = np.arange(len(algos))
    width = 0.25
    
    ax.bar(x - width, tp_vals, width, label='True Positive', color='#2ecc71')
    ax.bar(x, fp_vals, width, label='False Positive', color='#e74c3c')
    ax.bar(x + width, fn_vals, width, label='False Negative', color='#f39c12')
    
    ax.set_ylabel('数量', fontsize=12)
    ax.set_title('检测结果详细对比 (TP/FP/FN)', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    ax = fig.add_subplot(gs[2, 2])
    dynamic_perf = perf_df[~perf_df['Algorithm'].isin(['StaticReputation', 'NoTrustFusion'])]
    sorted_perf = dynamic_perf.sort_values('F1_Score', ascending=True)
    
    bars = ax.barh(sorted_perf['Algorithm'], sorted_perf['F1_Score'], 
                   color=[colors_map.get(a, '#34495e') for a in sorted_perf['Algorithm']])
    ax.set_xlabel('F1分数', fontsize=12)
    ax.set_title('动态算法F1排名', fontweight='bold', fontsize=13)
    ax.set_xlim(0, 1.1)
    ax.grid(axis='x', alpha=0.3)
    for i, (bar, f1) in enumerate(zip(bars, sorted_perf['F1_Score'])):
        ax.text(f1 + 0.02, i, f'{f1:.3f}', va='center', fontsize=10)
    
    ax = fig.add_subplot(gs[3, :])
    metrics_matrix = perf_df[['Algorithm', 'F1_Score', 'Accuracy', 'Precision', 'Recall']].set_index('Algorithm')
    
    im = ax.imshow(metrics_matrix.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(metrics_matrix.index)))
    ax.set_yticks(np.arange(len(metrics_matrix.columns)))
    ax.set_xticklabels(metrics_matrix.index, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(metrics_matrix.columns, fontsize=11)
    
    for i in range(len(metrics_matrix.columns)):
        for j in range(len(metrics_matrix.index)):
            text = ax.text(j, i, f'{metrics_matrix.iloc[j, i]:.3f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    ax.set_title('性能指标热力图', fontweight='bold', fontsize=14, pad=20)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('数值', fontsize=11)
    
    plt.savefig(os.path.join(fig_dir, 'all_metrics_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] 全指标对比图: {fig_dir}/all_metrics_comparison.png")


def generate_performance_radar_chart(perf_df):
    """生成性能雷达图"""
    fig_dir = os.path.join(OUTPUT_DIR, 'figures')
    
    dynamic_algos = perf_df[~perf_df['Algorithm'].isin(['StaticReputation', 'NoTrustFusion'])]
    
    categories = ['F1分数', '准确率', '精确率', '召回率']
    num_vars = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors_map = {
        'DRAMBR': '#3498db',
        'PlexeMDS': '#e74c3c',
        'MajorityVoting': '#2ecc71',
        'EnhancedDRAMBR': '#9b59b6'
    }
    
    for idx, row in dynamic_algos.iterrows():
        values = [row['F1_Score'], row['Accuracy'], row['Precision'], row['Recall']]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=row['Algorithm'], color=colors_map.get(row['Algorithm'], '#34495e'))
        ax.fill(angles, values, alpha=0.15, color=colors_map.get(row['Algorithm'], '#34495e'))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)
    
    ax.set_title('算法性能雷达图', fontweight='bold', fontsize=15, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.savefig(os.path.join(fig_dir, 'performance_radar.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] 性能雷达图: {fig_dir}/performance_radar.png")


def generate_confusion_matrices_grid(perf_df):
    """生成所有算法的混淆矩阵网格"""
    fig_dir = os.path.join(OUTPUT_DIR, 'figures')
    
    dynamic_algos = perf_df[~perf_df['Algorithm'].isin(['StaticReputation', 'NoTrustFusion'])]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(dynamic_algos.iterrows()):
        if idx >= 4:
            break
        
        ax = axes[idx]
        cm = np.array([[row['TN'], row['FP']], [row['FN'], row['TP']]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['预测正常', '预测恶意'],
                   yticklabels=['实际正常', '实际恶意'],
                   cbar_kws={'label': '数量'}, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        
        ax.set_title(f"{row['Algorithm']}\nF1={row['F1_Score']:.3f}, Recall={row['Recall']:.2%}", 
                    fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'confusion_matrices_grid.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] 混淆矩阵网格: {fig_dir}/confusion_matrices_grid.png")


def generate_efficiency_analysis(perf_df):
    """生成效率分析图（性能vs时间）"""
    fig_dir = os.path.join(OUTPUT_DIR, 'figures')
    
    dynamic_algos = perf_df[~perf_df['Algorithm'].isin(['StaticReputation', 'NoTrustFusion'])]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors_map = {
        'DRAMBR': '#3498db',
        'PlexeMDS': '#e74c3c',
        'MajorityVoting': '#2ecc71',
        'EnhancedDRAMBR': '#9b59b6'
    }
    
    for _, row in dynamic_algos.iterrows():
        ax1.scatter(row['Processing_Time'], row['F1_Score'], 
                   s=300, alpha=0.6, color=colors_map.get(row['Algorithm'], '#34495e'),
                   label=row['Algorithm'])
        ax1.annotate(row['Algorithm'], 
                    (row['Processing_Time'], row['F1_Score']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax1.set_xlabel('处理时间(秒)', fontsize=12)
    ax1.set_ylabel('F1分数', fontsize=12)
    ax1.set_title('性能-效率权衡分析', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    efficiency_score = dynamic_algos['F1_Score'] / (dynamic_algos['Processing_Time'] + 0.01)
    sorted_eff = dynamic_algos.copy()
    sorted_eff['Efficiency'] = efficiency_score
    sorted_eff = sorted_eff.sort_values('Efficiency', ascending=True)
    
    bars = ax2.barh(sorted_eff['Algorithm'], sorted_eff['Efficiency'],
                   color=[colors_map.get(a, '#34495e') for a in sorted_eff['Algorithm']])
    ax2.set_xlabel('效率分数 (F1/时间)', fontsize=12)
    ax2.set_title('算法效率排名', fontweight='bold', fontsize=14)
    ax2.grid(axis='x', alpha=0.3)
    
    for i, (bar, eff) in enumerate(zip(bars, sorted_eff['Efficiency'])):
        ax2.text(eff + 0.1, i, f'{eff:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'efficiency_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] 效率分析图: {fig_dir}/efficiency_analysis.png")


def generate_complex_visualizations(perf_df, time_series_metrics, scenario):
    """生成复杂场景的所有可视化（增强版）"""
    fig_dir = os.path.join(OUTPUT_DIR, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    generate_all_metrics_comparison(perf_df)
    
    generate_performance_radar_chart(perf_df)
    
    generate_confusion_matrices_grid(perf_df)
    
    generate_efficiency_analysis(perf_df)
    
    # 原有的时间序列和基础对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax = axes[0, 0]
    for algo_name, metrics in time_series_metrics.items():
        if algo_name not in ['StaticReputation', 'NoTrustFusion']:
            ax.plot(metrics['time_step'], metrics['f1'], 
                   marker='o', label=algo_name, linewidth=2, markersize=4)
    ax.set_xlabel('时间步', fontsize=12)
    ax.set_ylabel('F1分数', fontsize=12)
    ax.set_title('F1分数随时间变化', fontweight='bold', fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    ax = axes[0, 1]
    if 'DRAMBR' in time_series_metrics:
        metrics = time_series_metrics['DRAMBR']
        ax.plot(metrics['time_step'], metrics['tp'], 
               marker='o', label='True Positive', color='green', linewidth=2, markersize=4)
        ax.plot(metrics['time_step'], metrics['fp'], 
               marker='s', label='False Positive', color='red', linewidth=2, markersize=4)
        ax.plot(metrics['time_step'], metrics['fn'], 
               marker='^', label='False Negative', color='orange', linewidth=2, markersize=4)
    ax.set_xlabel('时间步', fontsize=12)
    ax.set_ylabel('数量', fontsize=12)
    ax.set_title('DRAMBR检测结果随时间变化', fontweight='bold', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    algos = perf_df['Algorithm'].tolist()
    f1_scores = perf_df['F1_Score'].tolist()
    colors = ['#e74c3c' if f1 < 0.5 else '#f39c12' if f1 < 0.8 else '#2ecc71' 
              for f1 in f1_scores]
    bars = ax.barh(algos, f1_scores, color=colors)
    ax.set_xlabel('F1分数', fontsize=12)
    ax.set_title('最终F1分数对比', fontweight='bold', fontsize=14)
    ax.set_xlim(0, 1.1)
    for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
        ax.text(f1 + 0.02, i, f'{f1:.3f}', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    
    ax = axes[1, 1]
    times = perf_df['Processing_Time'].tolist()
    bars = ax.bar(algos, times, color='#3498db')
    ax.set_ylabel('处理时间(秒)', fontsize=12)
    ax.set_title('算法处理时间对比', fontweight='bold', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, t, f'{t:.2f}s',
               ha='center', va='bottom', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'complex_scenario_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] 复杂场景分析图: {fig_dir}/complex_scenario_analysis.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    attack_types = ['持续攻击', '间歇攻击', '协同攻击', '自适应攻击']
    counts = [
        len(scenario.persistent_attackers),
        len(scenario.intermittent_attackers),
        len(scenario.coordinated_attackers),
        len(scenario.adaptive_attackers)
    ]
    colors_pie = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db']
    wedges, texts, autotexts = ax.pie(counts, labels=attack_types, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    ax.set_title('恶意车辆类型分布', fontweight='bold', fontsize=14)
    
    ax = axes[1]
    difficulty_scores = perf_df[perf_df['Algorithm'].isin(
        ['DRAMBR', 'PlexeMDS', 'MajorityVoting', 'EnhancedDRAMBR']
    )][['Algorithm', 'Recall']].sort_values('Recall')
    
    bars = ax.barh(difficulty_scores['Algorithm'], difficulty_scores['Recall'],
                   color='#2ecc71')
    ax.set_xlabel('召回率', fontsize=12)
    ax.set_title('不同算法对复杂攻击的检测能力', fontweight='bold', fontsize=14)
    ax.set_xlim(0, 1.1)
    for i, (idx, row) in enumerate(difficulty_scores.iterrows()):
        ax.text(row['Recall'] + 0.02, i, f"{row['Recall']:.2%}",
               va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'attack_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] 攻击分析图: {fig_dir}/attack_analysis.png")


def print_summary(perf_df, scenario):
    """打印总结"""
    print("\n" + "="*60)
    print("复杂场景测试总结")
    print("="*60)
    
    print("\n[场景配置]:")
    print(f"  - 车辆总数: {scenario.num_vehicles}")
    print(f"  - 恶意车辆: {scenario.num_malicious}")
    print(f"  - 时间步数: {scenario.duration}")
    print(f"  - 攻击类型: 4种（持续/间歇/协同/自适应）")
    
    print("\n[性能排名] (按F1分数):")
    sorted_df = perf_df.sort_values('F1_Score', ascending=False)
    for idx, row in sorted_df.iterrows():
        rank = sorted_df.index.get_loc(idx) + 1
        print(f"  {rank}. {row['Algorithm']:17s} F1={row['F1_Score']:.3f}, "
              f"Recall={row['Recall']:.2%}, Time={row['Processing_Time']:.2f}s")
    
    print("\n[关键发现]:")
    best = sorted_df.iloc[0]
    print(f"  - 最佳算法: {best['Algorithm']} (F1={best['F1_Score']:.3f})")
    
    dynamic_algos = perf_df[~perf_df['Algorithm'].isin(['StaticReputation', 'NoTrustFusion'])]
    print(f"  - 动态算法平均F1: {dynamic_algos['F1_Score'].mean():.3f}")
    print(f"  - 动态算法平均召回率: {dynamic_algos['Recall'].mean():.2%}")
    
    if best['Recall'] < 1.0:
        print(f"  - 复杂场景挑战: 最佳算法仍有 {(1-best['Recall'])*100:.1f}% 漏检率")
        print(f"  - 说明间歇性和自适应攻击增加了检测难度")
    else:
        print(f"  - 所有动态算法都成功应对了复杂攻击场景")
