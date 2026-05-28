"""
消融实验：k 值和 beta 值敏感性分析
在 ogbn-arxiv 子图上进行（加快速度）
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
from tqdm import tqdm
import gc

from src import config
from src.models import MLP, GCN, LLM_GNN
from scripts.preprocess.preprocess_arxiv import preprocess_arxiv, sample_arxiv_subgraph

import torch.serialization
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, Data, GlobalStorage, NodeStorage, EdgeStorage])


def random_edge_drop(edge_index, drop_rate):
    if drop_rate <= 0:
        return edge_index
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges) > drop_rate
    return edge_index[:, keep_mask]


def train_and_eval(data, k_neighbors, beta, drop_rate=0.5, num_epochs=200):
    """训练并评估 LLM-GNN 模型"""
    device = config.DEVICE
    
    sparse_edge = random_edge_drop(data.edge_index, drop_rate) if drop_rate > 0 else data.edge_index
    in_dim = data.x_llm.size(1)
    num_classes = data.y.max().item() + 1
    
    model = LLM_GNN(in_dim, config.GNN_HIDDEN_DIM, num_classes,
                   k_neighbors=k_neighbors, beta=beta)
    model = model.to(device)
    
    exp_data = Data(
        x=data.x.to(device),
        x_llm=data.x_llm.to(device),
        edge_index=sparse_edge.to(device),
        original_edge_index=data.edge_index.to(device),
        y=data.y.to(device),
        train_mask=data.train_mask.to(device),
        val_mask=data.val_mask.to(device),
        test_mask=data.test_mask.to(device)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=config.WEIGHT_DECAY)
    
    best_val_acc = 0
    best_test_acc = 0
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(exp_data.x_llm, exp_data.edge_index, original_edge_index=exp_data.original_edge_index)
        loss = F.cross_entropy(out[exp_data.train_mask], exp_data.y[exp_data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(exp_data.x_llm, exp_data.edge_index, original_edge_index=exp_data.original_edge_index)
                pred = out.argmax(dim=1)
                val_acc = (pred[exp_data.val_mask] == exp_data.y[exp_data.val_mask]).float().mean().item()
                test_acc = (pred[exp_data.test_mask] == exp_data.y[exp_data.test_mask]).float().mean().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
    
    del model, exp_data
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_test_acc


def run_k_sensitivity(data, drop_rate=0.5):
    """k 值敏感性分析"""
    print("\n" + "="*60)
    print("K-Neighbors Sensitivity Analysis")
    print("="*60)
    
    k_values = [2, 3, 5, 8, 10, 15, 20]
    beta = 0.5  # 固定 beta
    
    results = []
    for k in k_values:
        print(f"  k={k}...", end=" ", flush=True)
        accs = []
        for seed in range(3):
            config.set_seed(seed)
            acc = train_and_eval(data, k_neighbors=k, beta=beta, drop_rate=drop_rate)
            accs.append(acc)
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"Acc: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
        results.append({'k': k, 'beta': beta, 'accuracy': mean_acc, 'std': std_acc})
    
    return pd.DataFrame(results)


def run_beta_sensitivity(data, drop_rate=0.5):
    """beta 值敏感性分析"""
    print("\n" + "="*60)
    print("Beta Sensitivity Analysis")
    print("="*60)
    
    beta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    k = 5  # 固定 k
    
    results = []
    for beta in beta_values:
        print(f"  beta={beta}...", end=" ", flush=True)
        accs = []
        for seed in range(3):
            config.set_seed(seed)
            acc = train_and_eval(data, k_neighbors=k, beta=beta, drop_rate=drop_rate)
            accs.append(acc)
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"Acc: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
        results.append({'k': k, 'beta': beta, 'accuracy': mean_acc, 'std': std_acc})
    
    return pd.DataFrame(results)


def run_ablation_study(data, drop_rate=0.5):
    """消融实验：验证各组件贡献"""
    print("\n" + "="*60)
    print("Ablation Study")
    print("="*60)
    
    device = config.DEVICE
    in_dim_llm = data.x_llm.size(1)
    in_dim_raw = data.x.size(1)
    num_classes = data.y.max().item() + 1
    
    sparse_edge = random_edge_drop(data.edge_index, drop_rate)
    
    exp_data = Data(
        x=data.x.to(device),
        x_llm=data.x_llm.to(device),
        edge_index=sparse_edge.to(device),
        original_edge_index=data.edge_index.to(device),
        y=data.y.to(device),
        train_mask=data.train_mask.to(device),
        val_mask=data.val_mask.to(device),
        test_mask=data.test_mask.to(device)
    )
    
    ablations = [
        ("Full Model (LLM + Structure)", True, True),
        ("w/o Structure Learning", True, False),
        ("w/o LLM Features", False, True),
        ("w/o Both (GCN Raw)", False, False),
    ]
    
    results = []
    for name, use_llm, use_structure in ablations:
        print(f"  {name}...", end=" ", flush=True)
        
        accs = []
        for seed in range(3):
            config.set_seed(seed)
            
            if use_structure:
                model = LLM_GNN(in_dim_llm if use_llm else in_dim_raw, 
                              config.GNN_HIDDEN_DIM, num_classes,
                              k_neighbors=5, beta=0.5)
            else:
                model = GCN(in_dim_llm if use_llm else in_dim_raw,
                           config.GNN_HIDDEN_DIM, num_classes)
            
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=config.WEIGHT_DECAY)
            
            x = exp_data.x_llm if use_llm else exp_data.x
            best_test = 0
            
            for epoch in range(200):
                model.train()
                optimizer.zero_grad()
                if use_structure:
                    out = model(x, exp_data.edge_index, original_edge_index=exp_data.original_edge_index)
                else:
                    out = model(x, exp_data.edge_index)
                loss = F.cross_entropy(out[exp_data.train_mask], exp_data.y[exp_data.train_mask])
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        if use_structure:
                            out = model(x, exp_data.edge_index, original_edge_index=exp_data.original_edge_index)
                        else:
                            out = model(x, exp_data.edge_index)
                        pred = out.argmax(dim=1)
                        test_acc = (pred[exp_data.test_mask] == exp_data.y[exp_data.test_mask]).float().mean().item()
                        best_test = max(best_test, test_acc)
            
            accs.append(best_test)
            del model
            torch.cuda.empty_cache()
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"Acc: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
        results.append({'ablation': name, 'accuracy': mean_acc, 'std': std_acc})
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    print("="*60)
    print("Ablation & Sensitivity Analysis on ogbn-arxiv")
    print("="*60)
    
    config.set_seed()
    
    # 加载数据并采样子图（加快实验速度）
    print("\nLoading data...")
    data, _ = preprocess_arxiv()
    data = sample_arxiv_subgraph(data, num_nodes=20000, seed=42)
    print(f"Using subgraph: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    
    drop_rate = 0.5  # 中等稀疏度
    
    # 1. K 值敏感性
    k_results = run_k_sensitivity(data, drop_rate)
    k_results.to_csv('logs/sensitivity_k.csv', index=False)
    
    # 2. Beta 值敏感性
    beta_results = run_beta_sensitivity(data, drop_rate)
    beta_results.to_csv('logs/sensitivity_beta.csv', index=False)
    
    # 3. 消融实验
    ablation_results = run_ablation_study(data, drop_rate)
    ablation_results.to_csv('logs/ablation_study.csv', index=False)
    
    print("\n" + "="*60)
    print("All results saved to logs/")
    print("="*60)
    
    print("\nK Sensitivity:")
    print(k_results.to_string(index=False))
    
    print("\nBeta Sensitivity:")
    print(beta_results.to_string(index=False))
    
    print("\nAblation Study:")
    print(ablation_results.to_string(index=False))
