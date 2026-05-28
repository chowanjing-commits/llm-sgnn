"""消融实验可视化"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载数据
k_df = pd.read_csv('logs/sensitivity_k.csv')
beta_df = pd.read_csv('logs/sensitivity_beta.csv')
ablation_df = pd.read_csv('logs/ablation_study.csv')

# 1. K 值敏感性曲线
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(k_df['k'], k_df['accuracy']*100, yerr=k_df['std']*100, 
           marker='o', capsize=5, linewidth=2, markersize=8, color='#e74c3c')
ax.set_xlabel('Number of Neighbors (k)', fontsize=13)
ax.set_ylabel('Test Accuracy (%)', fontsize=13)
ax.set_title('K-Neighbors Sensitivity Analysis', fontsize=14)
ax.set_xticks(k_df['k'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sensitivity_k.png'), dpi=150)
print("Saved: sensitivity_k.png")
plt.close()

# 2. Beta 值敏感性曲线
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(beta_df['beta'], beta_df['accuracy']*100, yerr=beta_df['std']*100,
           marker='s', capsize=5, linewidth=2, markersize=8, color='#3498db')
ax.set_xlabel('Repair Budget (beta)', fontsize=13)
ax.set_ylabel('Test Accuracy (%)', fontsize=13)
ax.set_title('Beta Sensitivity Analysis', fontsize=14)
ax.set_xticks(beta_df['beta'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sensitivity_beta.png'), dpi=150)
print("Saved: sensitivity_beta.png")
plt.close()

# 3. 消融实验条形图
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#27ae60', '#3498db', '#e74c3c', '#95a5a6']
bars = ax.barh(range(len(ablation_df)), ablation_df['accuracy']*100, 
              xerr=ablation_df['std']*100, capsize=5, color=colors)
ax.set_yticks(range(len(ablation_df)))
ax.set_yticklabels(ablation_df['ablation'], fontsize=11)
ax.set_xlabel('Test Accuracy (%)', fontsize=13)
ax.set_title('Ablation Study: Component Contributions', fontsize=14)
ax.set_xlim([45, 60])

for i, (bar, acc) in enumerate(zip(bars, ablation_df['accuracy']*100)):
    ax.text(acc + 0.5, bar.get_y() + bar.get_height()/2, 
           f'{acc:.1f}%', va='center', fontsize=11, fontweight='bold')

ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_study.png'), dpi=150)
print("Saved: ablation_study.png")
plt.close()

# 4. 组合图
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# K sensitivity
axes[0].errorbar(k_df['k'], k_df['accuracy']*100, yerr=k_df['std']*100,
                marker='o', capsize=4, linewidth=2, markersize=7, color='#e74c3c')
axes[0].set_xlabel('k', fontsize=12)
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].set_title('(a) K Sensitivity', fontsize=13)
axes[0].set_xticks(k_df['k'])
axes[0].grid(True, alpha=0.3)

# Beta sensitivity
axes[1].errorbar(beta_df['beta'], beta_df['accuracy']*100, yerr=beta_df['std']*100,
                marker='s', capsize=4, linewidth=2, markersize=7, color='#3498db')
axes[1].set_xlabel('beta', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].set_title('(b) Beta Sensitivity', fontsize=13)
axes[1].set_xticks(beta_df['beta'])
axes[1].grid(True, alpha=0.3)

# Ablation
short_names = ['Full', 'w/o Struct', 'w/o LLM', 'w/o Both']
bars = axes[2].bar(range(4), ablation_df['accuracy']*100, 
                  yerr=ablation_df['std']*100, capsize=4, color=colors)
axes[2].set_xticks(range(4))
axes[2].set_xticklabels(short_names, fontsize=10)
axes[2].set_ylabel('Accuracy (%)', fontsize=12)
axes[2].set_title('(c) Ablation Study', fontsize=13)
axes[2].set_ylim([45, 62])
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_combined.png'), dpi=150)
print("Saved: ablation_combined.png")
plt.close()

print("\nAll ablation figures saved!")
