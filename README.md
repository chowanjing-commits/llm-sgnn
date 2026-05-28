# LLM-Augmented GNN for Sparse Graphs
# 面向稀疏图的 LLM 嵌入增强 GNN 模型

## 项目简介

本项目研究如何利用大语言模型 (LLM) 的语义理解能力，增强图神经网络 (GNN) 在稀疏图场景下的性能。我们使用 **ogbn-arxiv**、**20Newsgroups** 和 **WikiCS** 等包含真实文本属性的数据集，验证了 LLM 语义特征与结构学习相结合的有效性。

**核心思路**：
1. **特征增强**：利用 LLM (如 BERT, RoBERTa 等) 提取节点文本的深度语义嵌入，替代传统的 Bag-of-Words (BoW) 或 TF-IDF 特征。
2. **结构增强**：基于语义相似度构建 kNN 图或进行动态结构学习，弥补稀疏拓扑（如边缺失）带来的信息损失。

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric (PyG)
- OGB (Open Graph Benchmark)
- CUDA 11.8+ (推荐)
- 显存: 16GB (推荐 RTX 5060Ti 16GB 或更高，用于 LLM 推理)

## 安装

```bash
pip install -r requirements.txt
```

## 项目结构

```
LLM-SGNN/
├── src/                       # 核心代码模块
│   ├── config.py              # 全局参数配置
│   └── models.py              # 模型定义 (MLP, GCN, GAT, LLM_GNN)
├── scripts/                   # 脚本目录
│   ├── preprocess/            # 数据预处理脚本
│   │   ├── preprocess_arxiv.py
│   │   ├── preprocess_20news.py
│   │   └── preprocess_wikics.py
│   ├── train/                 # 训练脚本
│   │   ├── train_arxiv.py
│   │   ├── train_20news.py
│   │   └── train_wikics.py
│   ├── visualize/             # 可视化脚本
│   │   ├── visualize_arxiv.py
│   │   ├── visualize_20news.py
│   │   └── visualize_wikics.py
│   └── utils/                 # 工具脚本
│       ├── run_ablation.py
│       └── compare_embeddings_20news.py
├── docs/                      # 当前实验文档
│   └── arxiv_fixed_k10_node_drop.md
├── archive/                   # 归档的旧文档、图表与实验结果
│   └── legacy_experiment_report/
├── data/                      # 数据存放目录
├── logs/                      # 实验日志与结果 CSV
├── figures/                   # 生成的图表
└── checkpoints/               # 模型检查点
```

## 快速开始

本项目为每个数据集提供了独立的训练脚本，会自动处理数据下载、LLM 特征提取（或加载预计算特征）以及完整的对比实验。

### 1. 运行 ogbn-arxiv 实验

ogbn-arxiv 是一个大规模引文网络数据集，包含约 17 万篇论文。

```bash
python scripts/train/train_arxiv.py
```
该脚本将：
- 下载/加载 ogbn-arxiv 数据集
- 提取或加载论文标题/摘要的 LLM 嵌入
- 在不同稀疏度（Drop Edge Rate: 0%, 25%, 50%, 75%, 90%）下运行对比实验
- 结果保存在 `logs/arxiv_results_*.csv`

### 2. 运行 20 Newsgroups 实验

20 Newsgroups 是经典的文本分类数据集，构建为文档图。

```bash
python scripts/train/train_20news.py
```
该脚本将：
- 下载并预处理 20 Newsgroups 文本
- 基于 LLM 语义嵌入构建 k-NN 图结构（20 Newsgroups 原始无边）
- 运行完整对比实验
- 结果保存在 `logs/20news_results.csv`

### 3. 运行 WikiCS 实验

WikiCS 是基于 Wikipedia 的计算机科学领域网页引用网络。

```bash
python scripts/train/train_wikics.py
```
该脚本将：
- 自动处理 WikiCS 数据集
- 运行完整对比实验
- 结果保存在 `logs/wikics_results.csv`

## 实验说明

所有实验均对比以下模型变体：

| 模型 | 说明 |
|------|------|
| `MLP (LLM)` | Baseline：仅使用 LLM 语义特征，不利用图结构 |
| `GCN (Raw)` | 使用原始特征 (如 BoW/TF-IDF) + 稀疏图结构 |
| `GCN (LLM)` | 使用 LLM 语义特征 + 稀疏图结构 |
| `Ours (LLM-GNN)` | **本文方法**：LLM 语义特征 + 动态结构增强/kNN 图 |

实验会自动测试不同的 **Drop Edge Rates** (0% - 90%) 来模拟稀疏图场景，验证模型在拓扑信息缺失时的鲁棒性。

对于 20 Newsgroups 数据集，由于原始没有图结构，我们将基于 TF-IDF 构建的 kNN 图视为原始结构 $E_{raw}$，并在其上随机丢边（`drop_rate`）以模拟稀疏性；同时基于 LLM 嵌入构建语义 kNN 图作为 $E_{sem}$，本文方法在训练时使用并集融合 $E_{aug} = \text{drop}(E_{raw}) \cup E_{sem}$。

## 实验结果

运行脚本后，结果将以 CSV 格式保存在 `logs/` 目录下，并会在控制台打印如下格式的汇总表：

```
SUMMARY: Test Accuracy (%) at Different Sparsity Levels
============================================================
drop_rate       0.0   0.25   0.50   0.75   0.90
model
GCN (LLM)      71.52  70.12  68.45  65.30  60.15
GCN (Raw)      65.20  63.10  60.50  55.20  48.10
MLP (LLM)      68.10  68.10  68.10  68.10  68.10
Ours (LLM-GNN) 73.80  72.90  71.50  69.80  67.20
```
*(注：以上数据仅为示例，实际运行结果取决于具体实验设置和随机种子)*

## 文档与归档说明

- 当前实验说明位于 `docs/`
- 旧版长报告及其配套图表、结果已归档到 `archive/legacy_experiment_report/`

## 引用

如果您使用了本项目的代码，请引用：

```
@misc{llm-sgnn,
  title={LLM-Augmented GNN for Sparse Graphs},
  year={2024}
}
```
