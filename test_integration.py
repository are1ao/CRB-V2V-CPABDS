"""
DIVA + OpenCOOD 最小连调脚本（基于实际 DIVA 实现）
功能：
1. 调用 DIVA 的 v2v.py 生成声誉分数文件
2. 读取声誉分数，模拟 OpenCOOD 融合过程
3. 将声誉分数保存为 JSON 文件，供 OpenCOOD 直接加载
4. 对声誉结果进行简单统计分析，验证恶意检测效果
"""

import os
import sys
import subprocess
import pandas as pd
import torch
import json
import numpy as np

# ==================== 配置区 ====================

# 获取脚本所在目录（即仓库根目录 CRB-V2V-CPABDS/）
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 构造各个路径
CAM_DATASET = os.path.join(ROOT_DIR,"DIVA-main", "ETSI-V2V-Dataset-main", "dataset", "mtits-dataset", "CAM-dataset", "datasetCam.csv")
DENM_DATASET = os.path.join(ROOT_DIR,"DIVA-main", "ETSI-V2V-Dataset-main", "dataset", "mtits-dataset", "DENM-dataset", "malicious", "datasetDen_20.csv")
INIT_REPUTATION = os.path.join(ROOT_DIR, "DIVA-main", "dataset", "initial_reputations.csv")
COVERAGE_FILE = os.path.join(ROOT_DIR, "DIVA-main", "dataset", "coverage.json")
OUTPUT_DIR = os.path.join(ROOT_DIR, "diva_output")
V2V_SCRIPT = os.path.join(ROOT_DIR, "DIVA-main", "reputation_algorithm", "v2v.py")

# 将 C4-main 加入 sys.path，以便导入 opencood
C4_DIR = os.path.join(ROOT_DIR, "C4-main")
sys.path.insert(0, C4_DIR)


# ==================== 第一步：运行 DIVA ====================
def run_diva_generate_reputations():
    required_files = [CAM_DATASET, DENM_DATASET, INIT_REPUTATION, COVERAGE_FILE]
    for f in required_files:
        if not os.path.exists(f):
            print(f"❌ 缺少文件: {f}")
            return None
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cmd = [
        sys.executable, V2V_SCRIPT,
        "-dc", CAM_DATASET,
        "-dd", DENM_DATASET,
        "-r", INIT_REPUTATION,
        "-c", COVERAGE_FILE,
        "-o", OUTPUT_DIR,
        "-a", "0.5", "-b", "0.5",
        "-wc", "600", "-wd", "20",
        "--thresholds_type", "mean",
        "--startTime", "2017-06-26 12:00:00"
    ]
    
    print("🔄 正在运行 DIVA 算法，请稍候...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ DIVA 运行成功！")
    except subprocess.CalledProcessError as e:
        print(f"❌ DIVA 运行失败，错误码: {e.returncode}")
        print(e.stderr)
        return None
    
    # 改进的文件查找逻辑：直接找 OUTPUT_DIR 下最新的 CSV 文件
    csv_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')]
    if not csv_files:
        print("❌ 未找到任何输出 CSV 文件")
        return None
    # 优先选择包含 'mean' 和 'beta' 的文件
    target_files = [f for f in csv_files if 'mean' in f and 'beta' in f]
    if target_files:
        output_file = os.path.join(OUTPUT_DIR, target_files[0])
    else:
        output_file = os.path.join(OUTPUT_DIR, csv_files[0])
    print(f"📁 声誉文件已生成: {output_file}")
    return output_file

def load_reputation_scores(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    score_dict = dict(zip(df['vehicle_did'].astype(str), df['score']))
    print(f"✅ 读取到 {len(score_dict)} 辆车的声誉分数")
    return score_dict, df

# ==================== 第二步：验证 DIVA 结果 ====================
def analyze_reputation(df, denm_dataset_path):
    """分析声誉分布，检查是否区分了正常/恶意车辆"""
    print("\n📊 声誉分数统计：")
    print(df['score'].describe())

    # 根据实际路径构造恶意车辆 ID 列表文件路径
    # 原始 DENM 路径: .../malicious/datasetDen_20.csv
    # 恶意列表路径: .../malicious/sources/malicious_sources_20.txt
    base_dir = os.path.dirname(denm_dataset_path)  # .../malicious
    dataset_suffix = os.path.basename(denm_dataset_path).split('_')[-1].replace('.csv', '')  # 20
    malicious_file = os.path.join(base_dir, 'sources', f'malicious_sources_{dataset_suffix}.txt')

    if os.path.exists(malicious_file):
        with open(malicious_file, 'r') as f:
            # 读取文件内容，格式可能是多行数字，也可能是一行逗号分隔
            content = f.read().strip()
            if ',' in content:
                malicious_ids = [x.strip() for x in content.split(',')]
            else:
                malicious_ids = [line.strip() for line in content.split('\n') if line.strip()]
        
        df['is_malicious'] = df['vehicle_did'].astype(str).isin(malicious_ids)
        normal = df[~df['is_malicious']]['score']
        malicious = df[df['is_malicious']]['score']
        
        print(f"\n🔍 正常车辆数量: {len(normal)}, 平均声誉: {normal.mean():.3f} (±{normal.std():.3f})")
        print(f"🔍 恶意车辆数量: {len(malicious)}, 平均声誉: {malicious.mean():.3f} (±{malicious.std():.3f})")
        
        threshold = 0.3
        tp = (malicious < threshold).sum()
        fn = (malicious >= threshold).sum()
        tn = (normal >= threshold).sum()
        fp = (normal < threshold).sum()
        accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
        print(f"\n🎯 阈值 {threshold} 下检测准确率: {accuracy:.2%}")
    else:
        print(f"⚠️ 未找到恶意车辆 ID 列表: {malicious_file}，跳过分类统计。")

    # ===== 可视化 =====
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = os.path.join(OUTPUT_DIR, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 确保分类列存在
    if 'is_malicious' in df.columns:
        # 为绘图创建可读标签
        df['车辆类别'] = df['is_malicious'].map({False: '正常车辆', True: '恶意车辆'})

        # 1. 直方图
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='score', hue='车辆类别', bins=20,
                     palette={'正常车辆': 'green', '恶意车辆': 'red'}, alpha=0.6)
        plt.xlabel('声誉分数')
        plt.ylabel('车辆数量')
        plt.title('车辆声誉分数分布直方图')
        plt.xlim(-0.05, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(fig_dir, 'reputation_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 直方图已保存至: {fig_dir}/reputation_histogram.png")

        # 2. 箱线图
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='车辆类别', y='score', palette={'正常车辆': 'green', '恶意车辆': 'red'})
        plt.xlabel('车辆类别')
        plt.ylabel('声誉分数')
        plt.title('正常与恶意车辆声誉分数对比')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(fig_dir, 'reputation_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 箱线图已保存至: {fig_dir}/reputation_boxplot.png")

        # 3. CDF 图
        plt.figure(figsize=(10, 6))
        for label, color, name in [(False, 'green', '正常车辆'), (True, 'red', '恶意车辆')]:
            subset = df[df['is_malicious'] == label]['score']
            if len(subset) > 0:
                sorted_scores = np.sort(subset)
                cdf = np.arange(1, len(sorted_scores)+1) / len(sorted_scores)
                plt.plot(sorted_scores, cdf, drawstyle='steps-post',
                         color=color, linewidth=2, label=name)
        plt.xlabel('声誉分数')
        plt.ylabel('累积比例')
        plt.title('车辆声誉分数累积分布函数 (CDF)')
        plt.xlim(-0.05, 1.05)
        plt.ylim(0, 1.05)
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(fig_dir, 'reputation_cdf.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📉 CDF 图已保存至: {fig_dir}/reputation_cdf.png")
    else:
        # 无恶意标签时的简化直方图
        plt.figure(figsize=(10, 6))
        sns.histplot(df['score'], bins=20, color='blue', alpha=0.6)
        plt.xlabel('声誉分数')
        plt.ylabel('车辆数量')
        plt.title('车辆声誉分数分布直方图')
        plt.xlim(-0.05, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(fig_dir, 'reputation_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 直方图已保存至: {fig_dir}/reputation_histogram.png")

# ==================== 第三步：模拟融合 ====================
def simulate_opencood_with_reputation(scores):
    ego_id = 'vehicle_1'
    cav_ids = ['vehicle_2', 'vehicle_3']
    ego_score = scores.get(ego_id, 0.5)
    cav_scores = [scores.get(cid, 0.5) for cid in cav_ids]
    
    print(f"\n🚗 自车 {ego_id} 声誉: {ego_score:.3f}")
    print(f"🚙 协作车声誉: {dict(zip(cav_ids, cav_scores))}")
    
    ego_feat = torch.randn(256, 64, 64)
    cav_feats = [torch.randn(256, 64, 64) for _ in cav_ids]
    weights = [ego_score] + cav_scores
    fused = sum(w * f for w, f in zip(weights, [ego_feat] + cav_feats))
    print("✅ 声誉加权融合模拟成功，特征图形状:", fused.shape)

# ==================== 主流程 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("DIVA + OpenCOOD 最小连调（恶意数据集验证）")
    print("=" * 60)
    
    rep_file = run_diva_generate_reputations()
    if rep_file is None:
        sys.exit(1)
    
    reputation_scores, df = load_reputation_scores(rep_file)
    
    # 分析结果
    analyze_reputation(df, DENM_DATASET)
    
    # 保存 JSON
    json_path = "reputation_map.json"   # 放在当前目录，便于后续使用
    with open(json_path, 'w') as f:
        json.dump(reputation_scores, f, indent=2)
    print(f"\n✅ 声誉映射 JSON 已保存至: {json_path}")
    
    simulate_opencood_with_reputation(reputation_scores)
    print("\n🎉 连调完成！")