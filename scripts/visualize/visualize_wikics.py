"""WikiCS 实验结果可视化"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv('logs/wikics_results.csv')
print("WikiCS Results:")
print(df)

# 稀疏鲁棒性曲线
fig, ax = plt.subplots(figsize=(10, 6))
models = ['MLP (LLM)', 'GCN (Raw)', 'GCN (LLM)', 'Ours (LLM-GNN)']
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
markers = ['o', 's', '^', 'D']

for i, model in enumerate(models):
    model_data = df[df['model'] == model].sort_values('drop_rate')
    drop_rates = model_data['drop_rate'].values * 100
    accuracies = model_data['accuracy'].values * 100
    stds = model_data['std'].values * 100
    ax.errorbar(drop_rates, accuracies, yerr=stds, marker=markers[i], 
               color=colors[i], linewidth=2.5, markersize=10, label=model, capsize=4)

ax.set_xlabel('Edge Drop Rate (%)', fontsize=14)
ax.set_ylabel('Test Accuracy (%)', fontsize=14)
ax.set_title('Sparsity Robustness on WikiCS (11.7k nodes)', fontsize=16)
ax.legend(loc='lower left', fontsize=12)
ax.set_xticks([0, 25, 50, 75, 90])
ax.set_ylim([65, 85])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'wikics_sparsity_robustness.png'), dpi=150)
print("Saved: wikics_sparsity_robustness.png")
plt.close()

# 结果汇总表
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
plt.savefig(os.path.join(OUTPUT_DIR, 'wikics_results_table.png'), dpi=150)
print("Saved: wikics_results_table.png")
plt.close()

print("\nWikiCS visualization complete!")
