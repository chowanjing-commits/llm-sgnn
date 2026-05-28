"""
Configuration file for LLM-Augmented GNN for Sparse Graphs
存放所有超参数
"""
import os

# ========================
# Path Configuration
# ========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, 'embeddings')  # 生成的句向量/特征文件
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

# Create directories if not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ========================
# Dataset Configuration
# ========================
DATASETS = ['ogbn-arxiv', '20Newsgroups', 'WikiCS']  # 项目使用的数据集
DATASETS_WITH_TEXT = ['ogbn-arxiv', '20Newsgroups', 'WikiCS']  # 带真实文本的数据集
DEFAULT_DATASET = 'ogbn-arxiv'

# ogbn-arxiv 配置
USE_ARXIV = True  # 设为 True 使用带真实文本的 arxiv 数据集

# ========================
# LLM Encoder Configuration
# ========================
# 推荐使用轻量级 Embedding 模型
LLM_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'  # 只有 80MB，下载快
# 备选: 'BAAI/bge-base-en-v1.5' (438MB), 'BAAI/bge-m3', 'intfloat/e5-large-v2'
LLM_BATCH_SIZE = 32  # 批处理大小
LLM_MAX_LENGTH = 512  # 最大文本长度

# ========================
# Sparsity Simulation
# ========================
DROP_EDGE_RATES = [0.0, 0.25, 0.50, 0.75, 0.90]  # 边删除比例
DEFAULT_DROP_RATE = 0.50

# ========================
# Structure Learning (kNN)
# ========================
K_NEIGHBORS_LIST = [2, 3, 5, 8, 10]  # kNN 邻居数候选
DEFAULT_K_NEIGHBORS = 5
BETA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]  # 图融合权重 β
DEFAULT_BETA = 0.5  # A_final = β * A_sparse + (1-β) * A_sem

# ========================
# GNN Model Configuration
# ========================
GNN_HIDDEN_DIM = 256  # 隐藏层维度
GNN_NUM_LAYERS = 2  # GNN 层数
GNN_DROPOUT = 0.5  # Dropout 比例
GNN_HEADS = 4  # GAT attention heads

# ========================
# Training Configuration
# ========================
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200
PATIENCE = 20  # Early stopping patience
NUM_RUNS = 10  # 多次运行取平均

# ========================
# Device Configuration
# ========================
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# Random Seed
# ========================
SEED = 42

def set_seed(seed=SEED):
    """设置随机种子以保证可复现性"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
