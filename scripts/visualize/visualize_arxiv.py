"""
ogbn-arxiv 实验结果可视化
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

LOG_DIR = 'logs'
OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_latest_results():
    """加载最新的实验结果"""
    import glob
    files = glob.glob(os.path.join(LOG_DIR, 'arxiv_results_*.csv'))
    if not files:
        raise FileNotFoundError("No result files found in logs/")
    latest = max(files, key=os.path.getctime)
    print(f"Loading results from: {latest}")
    return pd.read_csv(latest)


def plot_sparsity_robustness(df):
    """绘制稀疏鲁棒性曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['model'].unique()
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    markers = ['o', 's', '^', 'D']
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        drop_rates = model_data['drop_rate'].values * 100
        accuracies = model_data['accuracy'].values * 100
        stds = model_data['std'].values * 100
        
        ax.plot(drop_rates, accuracies, marker=markers[i], 
                color=colors[i], linewidth=2, markersize=8, label=model)
        ax.fill_between(drop_rates, accuracies - stds, accuracies + stds,
                       color=colors[i], alpha=0.15)
    
    ax.set_xlabel('Edge Drop Rate (%)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Sparsity Robustness on ogbn-arxiv (with Real Paper Titles)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_xticks([0, 25, 50, 75, 90])
    ax.set_ylim([45, 60])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'arxiv_sparsity_robustness.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_model_comparison(df):
    """绘制模型对比条形图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = df['model'].unique()
    drop_rates = df['drop_rate'].unique()
    
    x = np.arange(len(drop_rates))
    width = 0.2
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model].sort_values('drop_rate')
        accuracies = model_data['accuracy'].values * 100
        stds = model_data['std'].values * 100
        
        bars = ax.bar(x + i * width, accuracies, width, 
                     label=model, color=colors[i], yerr=stds, capsize=3)
    
    ax.set_xlabel('Edge Drop Rate', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Model Comparison at Different Sparsity Levels', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'{int(r*100)}%' for r in drop_rates])
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim([45, 60])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'arxiv_model_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_llm_improvement(df):
    """绘制 LLM 嵌入带来的提升"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    drop_rates = df['drop_rate'].unique()
    
    gcn_raw = df[df['model'] == 'GCN (Raw)'].sort_values('drop_rate')['accuracy'].values * 100
    gcn_llm = df[df['model'] == 'GCN (LLM)'].sort_values('drop_rate')['accuracy'].values * 100
    improvement = gcn_llm - gcn_raw
    
    colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvement]
    bars = ax.bar(range(len(drop_rates)), improvement, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Edge Drop Rate', fontsize=12)
    ax.set_ylabel('Accuracy Improvement (%)', fontsize=12)
    ax.set_title('LLM Embedding Improvement over Raw Features', fontsize=14)
    ax.set_xticks(range(len(drop_rates)))
    ax.set_xticklabels([f'{int(r*100)}%' for r in drop_rates])
    
    # 添加数值标签
    for i, (bar, imp) in enumerate(zip(bars, improvement)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
               f'+{imp:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'arxiv_llm_improvement.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_summary_table(df):
    """生成结果汇总表格图"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # 创建 pivot 表格
    pivot = df.pivot(index='model', columns='drop_rate', values='accuracy') * 100
    pivot = pivot.round(2)
    pivot.columns = [f'{int(c*100)}%' for c in pivot.columns]
    
    # 创建表格
    table = ax.table(
        cellText=pivot.values,
        rowLabels=pivot.index,
        colLabels=pivot.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # 高亮最佳值
    for i in range(len(pivot.columns)):
        col_vals = pivot.iloc[:, i].values
        best_idx = np.argmax(col_vals)
        table[(best_idx + 1, i)].set_facecolor('#d5f5e3')
    
    ax.set_title('Test Accuracy (%) at Different Sparsity Levels', fontsize=14, pad=20)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'arxiv_results_table.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == '__main__':
    print("="*50)
    print("Generating visualizations for ogbn-arxiv results")
    print("="*50)
    
    df = load_latest_results()
    print(f"\nLoaded {len(df)} result entries")
    
    plot_sparsity_robustness(df)
    plot_model_comparison(df)
    plot_llm_improvement(df)
    plot_summary_table(df)
    
    print(f"\nAll figures saved to {OUTPUT_DIR}/")
