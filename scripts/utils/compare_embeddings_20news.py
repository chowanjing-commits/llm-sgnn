"""
比较不同 LLM 嵌入方法在 20 Newsgroups 数据集上的性能
对比：Sentence-BERT vs HuggingFace Qwen vs Ollama Qwen
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import config
import numpy as np

def load_results(llm_type=None, result_file=None):
    """加载实验结果"""
    if result_file is None:
        if llm_type == 'sentence-bert':
            candidates = [
                'logs/20news_results_sentence-bert.csv',
                'logs/20news_results_sentence_bert.csv',
                'logs/20news_results.csv',
            ]
        elif llm_type == 'hf-qwen':
            candidates = [
                'logs/20news_results_hf-qwen-len256-pca384.csv',
                'logs/20news_results_hf_qwen_len256_pca384.csv',
                'logs/20news_results_hf_qwen.csv',
            ]
        elif llm_type == 'hf-qwen-no-pca':
            candidates = [
                'logs/20news_results_hf-qwen-len256.csv',
                'logs/20news_results_hf_qwen_len256.csv',
                'logs/20news_results_hf_qwen_len256_nopca.csv',
            ]
        elif llm_type == 'ollama-qwen':
            candidates = [
                'logs/20news_results_ollama_qwen.csv',
            ]
        else:
            raise ValueError(f"Unknown llm_type: {llm_type}")

        result_file = next((p for p in candidates if os.path.exists(p)), candidates[0])
    
    if not os.path.exists(result_file):
        print(f"Warning: {result_file} not found")
        return None
    
    return pd.read_csv(result_file)


def plot_single_result_file(result_file, title=None, output_path=None):
    df = load_results(result_file=result_file)
    if df is None or len(df) == 0:
        return

    df = df.copy()
    df['drop_rate_pct'] = df['drop_rate'] * 100
    df['accuracy_pct'] = df['accuracy'] * 100

    if title is None:
        llm_type = df['llm_type'].iloc[0] if 'llm_type' in df.columns else os.path.basename(result_file)
        title = f"20 Newsgroups Results ({llm_type})"

    if output_path is None:
        base = os.path.splitext(os.path.basename(result_file))[0]
        safe_name = base
        if safe_name.startswith('20news_results_'):
            safe_name = safe_name[len('20news_results_'):]
        output_path = os.path.join('figures', f"20news_{safe_name}_sparsity_robustness.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sns.set(style='whitegrid')
    plt.figure(figsize=(12, 6))

    model_order = ['MLP (LLM)', 'GCN (Raw)', 'GCN (LLM)', 'Ours (LLM-GNN)']
    for model_name in model_order:
        subset = df[df['model'] == model_name].sort_values('drop_rate')
        if len(subset) == 0:
            continue
        plt.plot(
            subset['drop_rate_pct'],
            subset['accuracy_pct'],
            marker='o',
            linewidth=2,
            label=model_name,
            markersize=7
        )

        if 'std' in subset.columns:
            std_pct = subset['std'] * 100
            plt.fill_between(
                subset['drop_rate_pct'],
                subset['accuracy_pct'] - std_pct,
                subset['accuracy_pct'] + std_pct,
                alpha=0.12
            )

    plt.xlabel('Edge Drop Rate (%)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()


def plot_model_comparison_from_file(result_file, title=None, output_path=None):
    df = load_results(result_file=result_file)
    if df is None or len(df) == 0:
        return

    df = df.copy()
    models = ['MLP (LLM)', 'GCN (Raw)', 'GCN (LLM)', 'Ours (LLM-GNN)']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

    drop_rates = sorted(df['drop_rate'].unique())
    x = np.arange(len(drop_rates))
    width = 0.2

    if title is None:
        llm_type = df['llm_type'].iloc[0] if 'llm_type' in df.columns else os.path.basename(result_file)
        title = f"Model Comparison on 20Newsgroups ({llm_type})"

    if output_path is None:
        base = os.path.splitext(os.path.basename(result_file))[0]
        safe_name = base
        if safe_name.startswith('20news_results_'):
            safe_name = safe_name[len('20news_results_'):]
        output_path = os.path.join('figures', f"20news_{safe_name}_model_comparison.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        model_data = df[df['model'] == model].sort_values('drop_rate')
        if len(model_data) == 0:
            continue
        accuracies = model_data['accuracy'].values * 100
        stds = model_data['std'].values * 100 if 'std' in model_data.columns else None

        ax.bar(
            x + i * width,
            accuracies,
            width,
            label=model,
            color=colors[i],
            yerr=stds,
            capsize=3 if stds is not None else 0
        )

    ax.set_xlabel('Edge Drop Rate', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'{int(r*100)}%' for r in drop_rates])
    ax.legend(loc='upper right', fontsize=11)

    # 动态 y 轴范围（保持与 figures/ 中风格一致，同时避免截断）
    y_min = float((df['accuracy'].min() * 100) - 5)
    y_max = float((df['accuracy'].max() * 100) + 5)
    ax.set_ylim([max(0, y_min), min(100, y_max)])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()


def compare_all_methods():
    """比较所有方法的性能"""
    print("="*80)
    print("20 Newsgroups: LLM Embedding Methods Comparison")
    print("="*80)
    
    methods = ['sentence-bert', 'hf-qwen', 'hf-qwen-no-pca', 'ollama-qwen']
    all_results = {}
    
    for method in methods:
        df = load_results(method)
        if df is not None:
            all_results[method] = df
            print(f"\n{method.upper()} Results:")
            pivot = df.pivot(index='model', columns='drop_rate', values='accuracy') * 100
            print(pivot.round(2).to_string())
    
    if len(all_results) < 2:
        print("\nNeed at least 2 methods to compare. Please run experiments first.")
        return
    
    # 合并所有结果
    combined_df = pd.concat(all_results.values(), ignore_index=True)
    
    # 为 LLM-GNN 模型绘制对比图
    llm_gnn_data = combined_df[combined_df['model'] == 'Ours (LLM-GNN)']
    
    if len(llm_gnn_data) > 0:
        print("\n" + "="*80)
        print("LLM-GNN Performance Comparison Across Different Embeddings")
        print("="*80)
        
        pivot_comparison = llm_gnn_data.pivot(index='llm_type', columns='drop_rate', values='accuracy') * 100
        print(pivot_comparison.round(2).to_string())
        
        # 绘制对比图
        plt.figure(figsize=(12, 6))
        
        for llm_type in llm_gnn_data['llm_type'].unique():
            subset = llm_gnn_data[llm_gnn_data['llm_type'] == llm_type]
            subset = subset.sort_values('drop_rate')
            plt.plot(subset['drop_rate'] * 100, subset['accuracy'] * 100, 
                    marker='o', linewidth=2, label=llm_type, markersize=8)
        
        plt.xlabel('Edge Drop Rate (%)', fontsize=12)
        plt.ylabel('Test Accuracy (%)', fontsize=12)
        plt.title('LLM-GNN: Embedding Method Comparison on 20 Newsgroups', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = 'figures/20news_embedding_comparison.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to: {output_path}")
        plt.close()
    
    # 对比所有模型在不同嵌入下的表现
    print("\n" + "="*80)
    print("All Models Performance Summary")
    print("="*80)
    
    for model_name in combined_df['model'].unique():
        print(f"\n{model_name}:")
        model_data = combined_df[combined_df['model'] == model_name]
        pivot = model_data.pivot(index='llm_type', columns='drop_rate', values='accuracy') * 100
        print(pivot.round(2).to_string())


def compare_two_methods(method1='sentence-bert', method2='hf-qwen'):
    """详细对比两种方法"""
    print("="*80)
    print(f"Detailed Comparison: {method1.upper()} vs {method2.upper()}")
    print("="*80)
    
    df1 = load_results(method1)
    df2 = load_results(method2)
    
    if df1 is None or df2 is None:
        print("One or both result files not found. Please run experiments first.")
        return
    
    # 对比 LLM-GNN 模型
    llm_gnn_1 = df1[df1['model'] == 'Ours (LLM-GNN)'].sort_values('drop_rate')
    llm_gnn_2 = df2[df2['model'] == 'Ours (LLM-GNN)'].sort_values('drop_rate')
    
    print(f"\nLLM-GNN Performance:")
    print(f"\n{method1.upper()}:")
    for _, row in llm_gnn_1.iterrows():
        print(f"  Drop Rate {row['drop_rate']*100:.0f}%: {row['accuracy']*100:.2f}% ± {row['std']*100:.2f}%")
    
    print(f"\n{method2.upper()}:")
    for _, row in llm_gnn_2.iterrows():
        print(f"  Drop Rate {row['drop_rate']*100:.0f}%: {row['accuracy']*100:.2f}% ± {row['std']*100:.2f}%")
    
    # 计算平均性能差异
    if len(llm_gnn_1) == len(llm_gnn_2):
        avg_diff = (llm_gnn_2['accuracy'].mean() - llm_gnn_1['accuracy'].mean()) * 100
        print(f"\nAverage Performance Difference: {avg_diff:+.2f}%")
        if avg_diff > 0:
            print(f"{method2.upper()} is better on average")
        elif avg_diff < 0:
            print(f"{method1.upper()} is better on average")
        else:
            print("Both methods perform similarly on average")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare LLM embedding methods on 20 Newsgroups')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all available methods')
    parser.add_argument('--plot-file', type=str, default=None,
                       help='Plot a single results CSV file (e.g., logs/20news_results_hf-qwen-len256-pca384.csv)')
    parser.add_argument('--plot-kind', type=str, default='curve',
                       choices=['curve', 'model-comparison'],
                       help='Plot type when using --plot-file')
    parser.add_argument('--plot-title', type=str, default=None,
                       help='Optional plot title when using --plot-file')
    parser.add_argument('--plot-output', type=str, default=None,
                       help='Optional output path when using --plot-file')
    parser.add_argument('--method1', type=str, default='sentence-bert',
                       choices=['sentence-bert', 'hf-qwen', 'hf-qwen-no-pca', 'ollama-qwen'],
                       help='First method to compare')
    parser.add_argument('--method2', type=str, default='hf-qwen',
                       choices=['sentence-bert', 'hf-qwen', 'hf-qwen-no-pca', 'ollama-qwen'],
                       help='Second method to compare')
    
    args = parser.parse_args()
    
    if args.plot_file is not None:
        if args.plot_kind == 'model-comparison':
            plot_model_comparison_from_file(args.plot_file, title=args.plot_title, output_path=args.plot_output)
        else:
            plot_single_result_file(args.plot_file, title=args.plot_title, output_path=args.plot_output)
    elif args.compare_all:
        compare_all_methods()
    else:
        compare_two_methods(args.method1, args.method2)
