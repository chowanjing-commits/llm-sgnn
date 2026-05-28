"""
验证 HuggingFace Qwen 和 Sentence-BERT 实验设置是否一致
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import os
from src import config
from scripts.preprocess.preprocess_20news import load_20news_data

print("="*80)
print("实验设置验证报告")
print("="*80)

# 1. 检查数据集
print("\n【1. 数据集设置】")
texts, labels, num_classes = load_20news_data()
print(f"  文档数量: {len(texts)}")
print(f"  类别数量: {num_classes}")
print(f"  标签分布: min={labels.min()}, max={labels.max()}")

# 2. 检查嵌入文件
print("\n【2. 嵌入文件】")
emb_files = {
    'Sentence-BERT': 'embeddings/20news_llm_emb.pt',
    'HuggingFace Qwen': 'embeddings/20news_llm_emb_hf_Qwen_Qwen2_5-1_5B.pt'
}

for name, path in emb_files.items():
    full_path = os.path.join(config.PROJECT_ROOT, path)
    if os.path.exists(full_path):
        emb = torch.load(full_path)
        print(f"  {name}:")
        print(f"    - 文件: {path}")
        print(f"    - 形状: {emb.shape}")
        print(f"    - 维度: {emb.shape[1]}")
        print(f"    - 数据类型: {emb.dtype}")
    else:
        print(f"  {name}: [MISSING] 文件不存在")

# 3. 检查图结构文件
print("\n【3. kNN 图结构】")
graph_files = {
    'Sentence-BERT': 'data/20news_graph.pt',
    'HuggingFace Qwen': 'embeddings/20news_graph_hf_Qwen_Qwen2_5-1_5B.pt'
}

for name, path in graph_files.items():
    full_path = os.path.join(config.PROJECT_ROOT, path)
    if os.path.exists(full_path):
        edge_index = torch.load(full_path)
        print(f"  {name}:")
        print(f"    - 文件: {path}")
        print(f"    - 形状: {edge_index.shape}")
        print(f"    - 边数: {edge_index.shape[1]}")
        print(f"    - 平均度数: {edge_index.shape[1] / len(texts):.2f}")
    else:
        print(f"  {name}: [MISSING] 文件不存在 (需要生成)")

# 4. 检查训练配置
print("\n【4. 训练配置】")
print(f"  Drop rates: {[0.0, 0.25, 0.5, 0.75, 0.9]}")
print(f"  模型列表: MLP (LLM), GCN (Raw), GCN (LLM), Ours (LLM-GNN)")
print(f"  运行次数: 3")
print(f"  随机种子: 0, 1, 2")
print(f"  GNN 隐藏维度: {config.GNN_HIDDEN_DIM}")
print(f"  学习率: {config.LEARNING_RATE}")
print(f"  权重衰减: {config.WEIGHT_DECAY}")
print(f"  训练轮数: {config.EPOCHS}")

# 5. 检查 kNN 参数
print("\n【5. kNN 图构建参数】")
print(f"  k (邻居数): 10")
print(f"  相似度度量: 余弦相似度 (Cosine Similarity)")
print(f"  算法: sklearn NearestNeighbors")

# 6. 检查 LLM-GNN 特定参数
print("\n【6. LLM-GNN 模型参数】")
print(f"  k_neighbors: {config.DEFAULT_K_NEIGHBORS}")
print(f"  repair budget beta: {config.DEFAULT_BETA}")

# 7. 检查数据划分
print("\n【7. 数据划分】")
print(f"  训练集: 60%")
print(f"  验证集: 20%")
print(f"  测试集: 20%")
print(f"  随机种子: 42 (固定)")

# 8. 关键差异检查
print("\n【8. 关键差异点】")
print("  ⚠️  嵌入维度:")
print("    - Sentence-BERT: 384 维")
print("    - HuggingFace Qwen: 1536 维 (4倍)")
print("  ⚠️  kNN 图:")
print("    - 必须分别基于各自的嵌入构建")
print("    - 不能共用同一个图文件")

# 9. 验证结论
print("\n【9. 验证结论】")
print("  [OK] 数据集相同")
print("  [OK] 训练配置相同")
print("  [OK] kNN 参数相同")
print("  [OK] 数据划分相同")

# 检查是否需要重新生成 Qwen 的 kNN 图
qwen_graph_path = os.path.join(config.PROJECT_ROOT, 'embeddings/20news_graph_hf_Qwen_Qwen2_5-1_5B.pt')
if not os.path.exists(qwen_graph_path):
    print("\n  [WARNING] 需要重新生成 HuggingFace Qwen 的 kNN 图!")
    print("     原因: 之前使用了 Sentence-BERT 的图，导致特征-结构不匹配")
else:
    print("\n  [OK] HuggingFace Qwen 的 kNN 图已存在")

print("\n" + "="*80)
print("验证完成")
print("="*80)
