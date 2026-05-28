"""
WikiCS 数据集实验
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
from scripts.preprocess.preprocess_wikics import preprocess_wikics

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


def sample_dropped_nodes(data, node_drop_rate):
    num_nodes = data.num_nodes
    dropped = torch.zeros(num_nodes, dtype=torch.bool)
    if node_drop_rate <= 0:
        return dropped

    candidate_nodes = torch.nonzero(data.train_mask, as_tuple=False).view(-1)

    if candidate_nodes.numel() == 0:
        return dropped

    num_drop = int(candidate_nodes.numel() * node_drop_rate)
    if node_drop_rate > 0 and num_drop == 0:
        num_drop = 1
    num_drop = min(num_drop, candidate_nodes.numel())
    if num_drop == 0:
        return dropped

    perm = torch.randperm(candidate_nodes.numel())[:num_drop]
    dropped[candidate_nodes[perm]] = True
    return dropped


def drop_incident_edges(edge_index, dropped_node_mask):
    if dropped_node_mask is None or not dropped_node_mask.any():
        return edge_index
    keep_mask = (~dropped_node_mask[edge_index[0]]) & (~dropped_node_mask[edge_index[1]])
    return edge_index[:, keep_mask]


def build_corrupted_graph(data, edge_drop_rate, node_drop_rate=0.0):
    dropped_node_mask = sample_dropped_nodes(data, node_drop_rate=node_drop_rate)
    isolated_edge_index = drop_incident_edges(data.edge_index, dropped_node_mask)
    sparse_edge_index = (
        random_edge_drop(isolated_edge_index, edge_drop_rate)
        if edge_drop_rate > 0
        else isolated_edge_index
    )
    return sparse_edge_index, dropped_node_mask


def train_and_eval(data, model_type, use_llm, drop_rate, num_epochs=300, node_drop_rate=0.0):
    device = config.DEVICE

    sparse_edge, dropped_node_mask = build_corrupted_graph(
        data,
        edge_drop_rate=drop_rate,
        node_drop_rate=node_drop_rate,
    )
    in_dim = data.x_llm.size(1) if use_llm else data.x.size(1)
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
        x_llm=data.x_llm.to(device),
        edge_index=sparse_edge.to(device),
        original_edge_index=data.edge_index.to(device),
        y=data.y.to(device),
        train_mask=data.train_mask.to(device),
        val_mask=data.val_mask.to(device),
        test_mask=data.test_mask.to(device),
        dropped_node_mask=dropped_node_mask.to(device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=config.WEIGHT_DECAY)

    best_val, best_test = 0, 0
    x = exp_data.x_llm if use_llm else exp_data.x

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        if model_type == 'LLM_GNN':
            out = model(x, exp_data.edge_index, original_edge_index=exp_data.original_edge_index)
        else:
            out = model(x, exp_data.edge_index)
        loss = F.cross_entropy(out[exp_data.train_mask], exp_data.y[exp_data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                if model_type == 'LLM_GNN':
                    out = model(x, exp_data.edge_index, original_edge_index=exp_data.original_edge_index)
                else:
                    out = model(x, exp_data.edge_index)
                pred = out.argmax(dim=1)
                if exp_data.val_mask.any():
                    val_acc = (pred[exp_data.val_mask] == exp_data.y[exp_data.val_mask]).float().mean().item()
                else:
                    val_acc = 0.0
                if exp_data.test_mask.any():
                    test_acc = (pred[exp_data.test_mask] == exp_data.y[exp_data.test_mask]).float().mean().item()
                else:
                    test_acc = 0.0
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc

    del model, exp_data
    torch.cuda.empty_cache()
    gc.collect()

    return best_test


def run_experiments(data, node_drop_rate=0.0):
    print("\n" + "="*60)
    print("WikiCS Experiments")
    print("="*60)

    drop_rates = [0.0, 0.25, 0.5, 0.75, 0.9]
    models_config = [
        ('MLP (LLM)', 'MLP', True),
        ('GCN (Raw)', 'GCN', False),
        ('GCN (LLM)', 'GCN', True),
        ('Ours (LLM-GNN)', 'LLM_GNN', True),
    ]

    results = []

    for drop_rate in drop_rates:
        print(f"\nDrop Rate: {drop_rate*100:.0f}%")

        for name, mtype, use_llm in models_config:
            print(f"  {name}...", end=" ", flush=True)

            accs = []
            for seed in range(3):
                config.set_seed(seed)
                acc = train_and_eval(
                    data,
                    mtype,
                    use_llm,
                    drop_rate,
                    node_drop_rate=node_drop_rate,
                )
                accs.append(acc)

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f"Acc: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")

            results.append({
                'dataset': 'WikiCS',
                'drop_rate': drop_rate,
                'node_drop_rate': node_drop_rate,
                'model': name,
                'accuracy': mean_acc,
                'std': std_acc
            })

    df = pd.DataFrame(results)
    if node_drop_rate > 0:
        result_path = f'logs/wikics_results_dropnode{int(node_drop_rate * 100)}.csv'
    else:
        result_path = 'logs/wikics_results.csv'
    df.to_csv(result_path, index=False)
    print(f"\nResults saved to {result_path}")

    print("\nSummary:")
    pivot = df.pivot(index='model', columns='drop_rate', values='accuracy') * 100
    print(pivot.round(2).to_string())

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='LLM-GNN Experiments on WikiCS')
    parser.add_argument('--node-drop-rate', type=float, default=0.0,
                       help='Fraction of nodes to isolate by removing all incident edges')
    args = parser.parse_args()

    print("="*60)
    print("LLM-GNN Experiments on WikiCS")
    print("="*60)

    config.set_seed()

    print("\nLoading WikiCS dataset...")
    data, _ = preprocess_wikics()
    print(f"Data: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    print(f"Node drop rate: {args.node_drop_rate:.2f}")

    results = run_experiments(
        data,
        node_drop_rate=args.node_drop_rate,
    )
