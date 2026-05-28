"""
继续运行 ogbn-arxiv 完整图实验
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
import gc

from src import config
from src.models import MLP, GCN, GAT, LLM_GNN
from scripts.preprocess.preprocess_arxiv import preprocess_arxiv

# PyTorch 2.6 兼容性
import torch.serialization
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
torch.serialization.add_safe_globals([
    DataEdgeAttr, DataTensorAttr, Data,
    GlobalStorage, NodeStorage, EdgeStorage
])


def random_edge_drop(edge_index, drop_rate):
    if drop_rate <= 0:
        return edge_index
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges) > drop_rate
    return edge_index[:, keep_mask]


def train_epoch(model, data, optimizer, use_llm_features=False):
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


def run_experiment(data, model_type, drop_rate, use_llm, num_epochs=500, lr=0.01, patience=100):
    device = config.DEVICE
    
    if drop_rate > 0:
        sparse_edge_index = random_edge_drop(data.edge_index, drop_rate)
    else:
        sparse_edge_index = data.edge_index
    
    in_dim = data.x_llm.size(1) if use_llm and hasattr(data, 'x_llm') else data.x.size(1)
    num_classes = data.y.max().item() + 1
    
    if model_type == 'MLP':
        model = MLP(in_dim, config.GNN_HIDDEN_DIM, num_classes)
    elif model_type == 'GCN':
        model = GCN(in_dim, config.GNN_HIDDEN_DIM, num_classes)
    elif model_type == 'LLM_GNN':
        model = LLM_GNN(in_dim, config.GNN_HIDDEN_DIM, num_classes,
                       k_neighbors=config.DEFAULT_K_NEIGHBORS, beta=config.DEFAULT_BETA)
    
    model = model.to(device)
    
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
    no_improve = 0
    
    for epoch in tqdm(range(num_epochs), desc=model_type, leave=False):
        loss = train_epoch(model, exp_data, optimizer, use_llm_features=use_llm)
        
        if (epoch + 1) % 10 == 0:
            results = evaluate(model, exp_data, use_llm_features=use_llm)
            if results['val'] > best_val_acc:
                best_val_acc = results['val']
                best_test_acc = results['test']
                no_improve = 0
            else:
                no_improve += 10
            if no_improve >= patience:
                break
    
    del model, exp_data
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_test_acc


def main():
    print("=" * 70)
    print("Continuing LLM-GNN Experiment on FULL ogbn-arxiv")
    print("=" * 70)
    
    config.set_seed()
    
    print("\nLoading data...")
    data, dataset = preprocess_arxiv()
    print(f"Full graph: {data.num_nodes:,} nodes, {data.edge_index.size(1):,} edges")
    
    existing = [
        {'dataset': 'ogbn-arxiv-full', 'drop_rate': 0.0, 'model': 'MLP (LLM)', 'accuracy': 0.619, 'std': 0.0},
        {'dataset': 'ogbn-arxiv-full', 'drop_rate': 0.0, 'model': 'GCN (Raw)', 'accuracy': 0.540, 'std': 0.0},
        {'dataset': 'ogbn-arxiv-full', 'drop_rate': 0.0, 'model': 'GCN (LLM)', 'accuracy': 0.610, 'std': 0.0},
    ]
    completed = {(r['drop_rate'], r['model']) for r in existing}
    results = existing.copy()
    
    drop_rates = [0.0, 0.25, 0.50, 0.75, 0.90]
    models_config = [
        ('MLP (LLM)', 'MLP', True),
        ('GCN (Raw)', 'GCN', False),
        ('GCN (LLM)', 'GCN', True),
        ('Ours (LLM-GNN)', 'LLM_GNN', True),
    ]
    
    for drop_rate in drop_rates:
        print(f"\n{'=' * 60}")
        print(f"Drop Rate: {drop_rate * 100:.0f}%")
        print('=' * 60)
        
        for model_name, model_type, use_llm in models_config:
            key = (drop_rate, model_name)
            
            if key in completed:
                print(f"  {model_name}: Already done, skipping...")
                continue
            
            print(f"\n  Training {model_name}...")
            
            config.set_seed(42)
            acc = run_experiment(data, model_type, drop_rate, use_llm)
            
            print(f"  Test Accuracy: {acc * 100:.2f}%")
            
            results.append({
                'dataset': 'ogbn-arxiv-full',
                'drop_rate': drop_rate,
                'model': model_name,
                'accuracy': acc,
                'std': 0.0
            })
            
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(config.LOG_DIR, 'arxiv_full_continued.csv'), index=False)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    df = pd.DataFrame(results)
    pivot = df.pivot(index='model', columns='drop_rate', values='accuracy') * 100
    print(pivot.round(2).to_string())
    
    df.to_csv(os.path.join(config.LOG_DIR, 'arxiv_full_final.csv'), index=False)
    print(f"\nResults saved to logs/arxiv_full_final.csv")


if __name__ == '__main__':
    main()
