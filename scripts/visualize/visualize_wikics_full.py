"""
WikiCS完整图实验结果可视化
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

# 加载完整图结果
df = pd.read_csv('logs/wikics_results.csv')
print(f"Loaded {len(df)} results")
print(df)

# 1. 稀疏鲁棒性曲线
fig, ax = plt.subplots(figsize=(10, 6))
models = ['MLP (LLM)', 'GCN (Raw)', 'GCN (LLM)', 'Ours (LLM-GNN)']
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
markers = ['o', 's', '^', 'D']

for i, model in enumerate(models):
    model_data = df[df['model'] == model].sort_values('drop_rate')
    drop_rates = model_data['drop_rate'].values * 100
    accuracies = model_data['accuracy'].values * 100
    ax.plot(drop_rates, accuracies, marker=markers[i], color=colors[i], 
            linewidth=2.5, markersize=10, label=model)

ax.set_xlabel('Edge Drop Rate (%)', fontsize=14)
ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title('Sparsity Robustness on WikiCS', fontsize=16)
ax.legend(loc='lower left', fontsize=12)
ax.set_xticks([0, 25, 50, 75, 90])
ax.set_ylim([65, 82])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wikics_full_sparsity_robustness.png'), dpi=150)
print("Saved: wikics_full_sparsity_robustness.png")
plt.close()

# 2. 模型对比条形图
fig, ax = plt.subplots(figsize=(12, 6))
drop_rates = df['drop_rate'].unique()
x = np.arange(len(drop_rates))
width = 0.2

for i, model in enumerate(models):
    model_data = df[df['model'] == model].sort_values('drop_rate')
    accuracies = model_data['accuracy'].values * 100
    ax.bar(x + i * width, accuracies, width, label=model, color=colors[i])

ax.set_xlabel('Edge Drop Rate', fontsize=14)
ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title('Model Comparison on WikiCS', fontsize=16)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f'{int(r*100)}%' for r in drop_rates])
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim([65, 82])
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wikics_full_model_comparison.png'), dpi=150)
print("Saved: wikics_full_model_comparison.png")
plt.close()

# 3. LLM 嵌入提升图
fig, ax = plt.subplots(figsize=(8, 6))
gcn_raw = df[df['model'] == 'GCN (Raw)'].sort_values('drop_rate')['accuracy'].values * 100
gcn_llm = df[df['model'] == 'GCN (LLM)'].sort_values('drop_rate')['accuracy'].values * 100
improvement = gcn_llm - gcn_raw

bars = ax.bar(range(len(drop_rates)), improvement, color='#27ae60', edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Edge Drop Rate', fontsize=14)
ax.set_ylabel('Accuracy Improvement (%)', fontsize=14)
ax.set_title('LLM Embedding Improvement over Raw Features (WikiCS)', fontsize=16)
ax.set_xticks(range(len(drop_rates)))
ax.set_xticklabels([f'{int(r*100)}%' for r in drop_rates])

for bar, imp in zip(bars, improvement):
    y_pos = bar.get_height() + 0.2 if bar.get_height() > 0 else bar.get_height() - 0.5
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
           f'{imp:+.1f}%', ha='center', va='bottom' if bar.get_height() > 0 else 'top', 
           fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wikics_full_llm_improvement.png'), dpi=150)
print("Saved: wikics_full_llm_improvement.png")
plt.close()

# 4. Ours vs Baselines 提升图
fig, ax = plt.subplots(figsize=(10, 6))
ours = df[df['model'] == 'Ours (LLM-GNN)'].sort_values('drop_rate')['accuracy'].values * 100
mlp = df[df['model'] == 'MLP (LLM)'].sort_values('drop_rate')['accuracy'].values * 100

imp_vs_gcn = ours - gcn_llm
imp_vs_mlp = ours - mlp

x = np.arange(len(drop_rates))
width = 0.35

ax.bar(x - width/2, imp_vs_gcn, width, label='vs GCN (LLM)', color='#3498db')
ax.bar(x + width/2, imp_vs_mlp, width, label='vs MLP (LLM)', color='#9b59b6')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Edge Drop Rate', fontsize=14)
ax.set_ylabel('Accuracy Improvement (%)', fontsize=14)
ax.set_title('LLM-GNN Improvement over Baselines (WikiCS)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels([f'{int(r*100)}%' for r in drop_rates])
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wikics_full_ours_improvement.png'), dpi=150)
print("Saved: wikics_full_ours_improvement.png")
plt.close()

# 5. 结果汇总表格
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

pivot = df.pivot(index='model', columns='drop_rate', values='accuracy') * 100
pivot = pivot.round(2)
pivot.columns = [f'{int(c*100)}%' for c in pivot.columns]

pivot = pivot.reindex(['MLP (LLM)', 'GCN (Raw)', 'GCN (LLM)', 'Ours (LLM-GNN)'])

table = ax.table(
    cellText=pivot.values,
    rowLabels=pivot.index,
    colLabels=pivot.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.3, 2.0)

for i in range(len(pivot.columns)):
    col_vals = pivot.iloc[:, i].values
    best_idx = np.argmax(col_vals)
    table[(best_idx + 1, i)].set_facecolor('#d5f5e3')

ax.set_title('Test Accuracy (%) on WikiCS', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wikics_full_results_table.png'), dpi=150)
print("Saved: wikics_full_results_table.png")
plt.close()

print(f"\nAll figures saved to {OUTPUT_DIR}/")
print("\nResults Summary:")
print(pivot.to_string())
