# -*- coding: utf-8 -*-
"""
一键运行脚本 - 快速测试改进的信誉系统
"""

import sys
import subprocess
from pathlib import Path

def print_banner(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_command(cmd, description):
    print(f">>> {description}")
    print(f">>> 命令: {cmd}\n")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"\n⚠️ 警告: {description} 失败 (返回码: {result.returncode})")
        return False
    return True

def main():
    print_banner("🚀 增强信誉系统 - 一键测试")
    
    # 检查工作目录
    current_dir = Path.cwd()
    print(f"当前目录: {current_dir}")
    
    if not (current_dir / "improved_reputation_engine.py").exists():
        print("\n⚠️ 错误: 请在 5_enhanced_reputation_system 目录下运行此脚本")
        print("   cd d:\\61-V2V\\CRB-V2V-CPABDS\\5_enhanced_reputation_system")
        sys.exit(1)
    
    # 步骤1: 运行测试
    print_banner("步骤 1/3: 验证改进效果")
    if not run_command("python test_improvements.py", "运行改进验证测试"):
        print("\n⚠️ 测试失败，请检查实现。继续下一步? (y/n)")
        if input().lower() != 'y':
            sys.exit(1)
    
    # 步骤2: 询问是否运行完整实验
    print_banner("步骤 2/3: 运行完整实验")
    print("是否运行完整实验? (这将需要5-15分钟)")
    print("  y - 运行完整实验 (3-5个episodes)")
    print("  n - 跳过")
    choice = input("请选择 (y/n): ").lower()
    
    if choice == 'y':
        run_command("python run_complete_experiment.py", "运行完整对比实验")
        
        # 步骤3: 生成可视化
        print_banner("步骤 3/3: 生成可视化")
        
        results_file = Path("results/experiment_results.json")
        if results_file.exists():
            print("✓ 找到实验结果文件")
            run_command("python advanced_visualization.py", "生成可视化图表")
            
            vis_dir = Path("visualizations")
            if vis_dir.exists():
                vis_files = list(vis_dir.glob("*.png"))
                if vis_files:
                    print(f"\n✓ 生成了 {len(vis_files)} 个可视化文件:")
                    for f in vis_files:
                        print(f"  - {f.name}")
        else:
            print("⚠️ 未找到实验结果文件，跳过可视化生成")
    else:
        print("跳过完整实验")
    
    # 完成
    print_banner("✅ 测试完成")
    print("下一步:")
    print("  1. 查看 README.md 了解详细文档")
    print("  2. 查看 QUICKSTART.md 了解快速开始指南")
    print("  3. 运行 'python reputation_socket_server.py' 启动VEINS服务器")
    print("  4. 查看 visualizations/ 目录下的可视化结果\n")

if __name__ == "__main__":
    main()
