"""
综合三个数据集的实验结果可视化
生成三个数据集上各模型在完整图上的性能对比图
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _pick_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


# 加载三个数据集的结果
arxiv_df = pd.read_csv('logs/arxiv_results.csv')
wikics_df = pd.read_csv('logs/wikics_results.csv')
news_path = _pick_first_existing([
    'logs/20news_results_sentence-bert.csv',
    'logs/20news_results_sentence_bert.csv',
    'logs/20news_results.csv',
])
news_df = pd.read_csv(news_path)
print(f"Loaded 20News results from: {news_path}")

def _select_drop(df, drop):
    return df[df['drop_rate'] == drop].copy()

datasets = ['ogbn-arxiv', 'WikiCS', '20 Newsgroups']
models = ['GCN (Raw)', 'GCN (LLM)', 'Ours (LLM-GNN)']
colors = ['#3498db', '#9b59b6', '#e74c3c']

def _extract_acc_std(df, model):
    model_data = df[df['model'] == model]
    if len(model_data) == 0:
        return 0.0, 0.0
    acc = float(model_data['accuracy'].values[0]) * 100
    std = float(model_data['std'].values[0]) * 100 if 'std' in model_data.columns else 0.0
    return acc, std

def _build_matrix(arxiv_sub, wikics_sub, news_sub):
    acc_matrix = []
    std_matrix = []
    for df in [arxiv_sub, wikics_sub, news_sub]:
        row_acc = []
        row_std = []
        for m in models:
            acc, std = _extract_acc_std(df, m)
            row_acc.append(acc)
            row_std.append(std)
        acc_matrix.append(row_acc)
        std_matrix.append(row_std)
    return np.array(acc_matrix), np.array(std_matrix)


drop_settings = [0.0, 0.9]
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

all_acc = []
for ax, drop in zip(axes, drop_settings):
    arxiv_sub = _select_drop(arxiv_df, drop)
    wikics_sub = _select_drop(wikics_df, drop)
    news_sub = _select_drop(news_df, drop)
    acc_matrix, std_matrix = _build_matrix(arxiv_sub, wikics_sub, news_sub)
    all_acc.append(acc_matrix)

    x = np.arange(len(datasets))
    width = 0.22
    for i, model in enumerate(models):
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            acc_matrix[:, i],
            width,
            label=model,
            color=colors[i],
            edgecolor='black',
            linewidth=0.5,
            yerr=std_matrix[:, i] if np.any(std_matrix[:, i]) else None,
            capsize=3
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.6,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_title(f"Drop Rate = {int(drop*100)}%", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')


axes[0].set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Dataset', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Dataset', fontsize=14, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=11, framealpha=0.9)
fig.suptitle('Model Comparison on Three Datasets (0% vs 90% Edge Drop)',
             fontsize=16, fontweight='bold', y=1.02)

all_acc = np.concatenate(all_acc, axis=None)
plt.ylim([max(0, float(np.min(all_acc) - 8)), min(100, float(np.max(all_acc) + 8))])
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, 'three_datasets_comparison.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()

print("\n" + "="*70)
print("SUMMARY: Test Accuracy (%) on Full/Sparse Graphs (0% vs 90% Drop)")
print("="*70)

