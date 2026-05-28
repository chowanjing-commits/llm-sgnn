"""
ogbn-arxiv 数据集训练脚本
使用真实论文标题的 LLM 嵌入进行实验
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from src import config
from src.models import MLP, GCN, GAT, LLM_GNN
from scripts.preprocess.preprocess_arxiv import preprocess_arxiv, sample_arxiv_subgraph

# 修复 PyTorch 2.6 兼容性
import torch.serialization
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
torch.serialization.add_safe_globals([
    DataEdgeAttr, DataTensorAttr, Data,
    GlobalStorage, NodeStorage, EdgeStorage
])


def random_edge_drop(edge_index, drop_rate):
    """随机删除边"""
    if drop_rate <= 0:
        return edge_index
    
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges) > drop_rate
    return edge_index[:, keep_mask]


def train_epoch(model, data, optimizer, use_llm_features=False):
    """训练一个 epoch"""
    model.train()
    optimizer.zero_grad()
    
    x = data.x_llm if use_llm_features and hasattr(data, 'x_llm') else data.x
    
    if hasattr(model, 'structure_learner'):
        out = model(x, data.edge_index, original_edge_index=data.original_edge_index)
    else:
        out = model(x, data.edge_index)
    
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, data, use_llm_features=False):
    """评估模型"""
    model.eval()
    
    x = data.x_llm if use_llm_features and hasattr(data, 'x_llm') else data.x
    
    if hasattr(model, 'structure_learner'):
        out = model(x, data.edge_index, original_edge_index=data.original_edge_index)
    else:
        out = model(x, data.edge_index)
    
    pred = out.argmax(dim=1)
    
    results = {}
    for split, mask in [('train', data.train_mask), ('val', data.val_mask), ('test', data.test_mask)]:
        correct = (pred[mask] == data.y[mask]).sum().item()
        total = mask.sum().item()
        results[split] = correct / total if total > 0 else 0
    
    return results


def run_experiment(data, model_type, drop_rate=0.0, use_llm=True, num_epochs=200, lr=0.01):
    """运行单次实验"""
    device = config.DEVICE
    
    # 准备数据
    if drop_rate > 0:
        sparse_edge_index = random_edge_drop(data.edge_index, drop_rate)
    else:
        sparse_edge_index = data.edge_index
    
    # 确定输入维度
    if use_llm and hasattr(data, 'x_llm'):
        in_dim = data.x_llm.size(1)
    else:
        in_dim = data.x.size(1)
    
    num_classes = data.y.max().item() + 1
    
    # 创建模型
    if model_type == 'MLP':
        model = MLP(in_dim, config.GNN_HIDDEN_DIM, num_classes)
    elif model_type == 'GCN':
        model = GCN(in_dim, config.GNN_HIDDEN_DIM, num_classes)
    elif model_type == 'GAT':
        model = GAT(in_dim, config.GNN_HIDDEN_DIM, num_classes)
    elif model_type == 'LLM_GNN':
        model = LLM_GNN(in_dim, config.GNN_HIDDEN_DIM, num_classes, 
                       k_neighbors=config.DEFAULT_K_NEIGHBORS, beta=config.DEFAULT_BETA)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # 准备数据
    exp_data = Data(
        x=data.x.to(device),
        x_llm=data.x_llm.to(device) if hasattr(data, 'x_llm') else data.x.to(device),
        edge_index=sparse_edge_index.to(device),
        original_edge_index=data.edge_index.to(device),
        y=data.y.to(device),
        train_mask=data.train_mask.to(device),
        val_mask=data.val_mask.to(device),
        test_mask=data.test_mask.to(device)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    
    best_val_acc = 0
    best_test_acc = 0
    
    for epoch in range(num_epochs):
        loss = train_epoch(model, exp_data, optimizer, use_llm_features=use_llm)
        
        if (epoch + 1) % 10 == 0:
            results = evaluate(model, exp_data, use_llm_features=use_llm)
            if results['val'] > best_val_acc:
                best_val_acc = results['val']
                best_test_acc = results['test']
    
    return best_test_acc


def run_full_experiments(data, use_subgraph=True, subgraph_size=5000):
    """运行完整实验"""
    
    if use_subgraph:
        print(f"\nSampling subgraph with {subgraph_size} nodes for faster experiments...")
        data = sample_arxiv_subgraph(data, num_nodes=subgraph_size)
    
    results = []
    drop_rates = [0.0, 0.25, 0.50, 0.75, 0.90]
    
    models_config = [
        ('MLP (LLM)', 'MLP', True),
        ('GCN (Raw)', 'GCN', False),
        ('GCN (LLM)', 'GCN', True),
        ('Ours (LLM-GNN)', 'LLM_GNN', True),
    ]
    
    for drop_rate in drop_rates:
        print(f"\n{'='*50}")
        print(f"Drop Rate: {drop_rate*100:.0f}%")
        print('='*50)
        
        for model_name, model_type, use_llm in models_config:
            print(f"  Training {model_name}...", end=' ')
            
            # 多次运行取平均
            accs = []
            for seed in range(3):
                config.set_seed(seed)
                acc = run_experiment(data, model_type, drop_rate, use_llm, 
                                   num_epochs=100, lr=0.01)
                accs.append(acc)
            
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f"Acc: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
            
            results.append({
                'dataset': 'ogbn-arxiv',
                'drop_rate': drop_rate,
                'model': model_name,
                'accuracy': mean_acc,
                'std': std_acc
            })
    
    # 保存结果
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = os.path.join(config.LOG_DIR, f'arxiv_results_{timestamp}.csv')
    df.to_csv(result_path, index=False)
    print(f"\nResults saved to {result_path}")
    
    # 打印汇总
    print("\n" + "="*60)
    print("SUMMARY: Test Accuracy (%) at Different Sparsity Levels")
    print("="*60)
    
    pivot = df.pivot(index='model', columns='drop_rate', values='accuracy')
    pivot = pivot * 100  # 转换为百分比
    print(pivot.round(2).to_string())
    
    return df


if __name__ == '__main__':
    config.set_seed()
    
    print("="*60)
    print("LLM-GNN Experiment on ogbn-arxiv (with real paper titles)")
    print("="*60)
    
    # 加载预处理数据
    data, dataset = preprocess_arxiv()
    
    # 运行实验（使用子图加速）
    results = run_full_experiments(data, use_subgraph=True, subgraph_size=10000)
