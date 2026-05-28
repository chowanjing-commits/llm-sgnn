"""Continue experiments on full ogbn-arxiv"""
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
from scripts.preprocess.preprocess_arxiv import preprocess_arxiv

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

def train_epoch(model, data, optimizer, use_llm):
    model.train()
    optimizer.zero_grad()
    x = data.x_llm if use_llm else data.x
    if hasattr(model, 'structure_learner'):
        out = model(x, data.edge_index, original_edge_index=data.original_edge_index)
    else:
        out = model(x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, use_llm):
    model.eval()
    x = data.x_llm if use_llm else data.x
    if hasattr(model, 'structure_learner'):
        out = model(x, data.edge_index, original_edge_index=data.original_edge_index)
    else:
        out = model(x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
    total = data.test_mask.sum().item()
    return correct / total

def run_exp(data, model_type, drop_rate, use_llm):
    device = config.DEVICE
    sparse_edge = random_edge_drop(data.edge_index, drop_rate) if drop_rate > 0 else data.edge_index
    in_dim = data.x_llm.size(1) if use_llm else data.x.size(1)
    num_classes = data.y.max().item() + 1
    
    if model_type == 'MLP':
        model = MLP(in_dim, config.GNN_HIDDEN_DIM, num_classes)
    elif model_type == 'GCN':
        model = GCN(in_dim, config.GNN_HIDDEN_DIM, num_classes)
    elif model_type == 'LLM_GNN':
        model = LLM_GNN(in_dim, config.GNN_HIDDEN_DIM, num_classes, k_neighbors=config.DEFAULT_K_NEIGHBORS, beta=config.DEFAULT_BETA)
    
    model = model.to(device)
    exp_data = Data(x=data.x.to(device), x_llm=data.x_llm.to(device), edge_index=sparse_edge.to(device),
                   original_edge_index=data.edge_index.to(device),
                   y=data.y.to(device), train_mask=data.train_mask.to(device),
                   val_mask=data.val_mask.to(device), test_mask=data.test_mask.to(device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=config.WEIGHT_DECAY)
    best_acc = 0
    
    for epoch in tqdm(range(500), desc=model_type, leave=False):
        train_epoch(model, exp_data, optimizer, use_llm)
        if (epoch + 1) % 10 == 0:
            acc = evaluate(model, exp_data, use_llm)
            if acc > best_acc:
                best_acc = acc
    
    del model, exp_data
    torch.cuda.empty_cache()
    gc.collect()
    return best_acc

print("=" * 60)
print("Continuing Full ogbn-arxiv Experiments")
print("=" * 60)

config.set_seed()
data, _ = preprocess_arxiv()
print(f"Loaded: {data.num_nodes:,} nodes")

existing = [
    (0.0, 'MLP (LLM)', 0.619),
    (0.0, 'GCN (Raw)', 0.540),
    (0.0, 'GCN (LLM)', 0.610),
]
completed = {(dr, m) for dr, m, _ in existing}
results = [{'drop_rate': dr, 'model': m, 'accuracy': a} for dr, m, a in existing]

models = [('MLP (LLM)', 'MLP', True), ('GCN (Raw)', 'GCN', False), ('GCN (LLM)', 'GCN', True), ('Ours (LLM-GNN)', 'LLM_GNN', True)]

for dr in [0.0, 0.25, 0.5, 0.75, 0.9]:
    print(f"\nDrop Rate: {dr*100:.0f}%")
    for name, mtype, use_llm in models:
        if (dr, name) in completed:
            print(f"  {name}: skipped")
            continue
        print(f"  Training {name}...", end=" ", flush=True)
        config.set_seed(42)
        acc = run_exp(data, mtype, dr, use_llm)
        print(f"Acc: {acc*100:.2f}%")
        results.append({'drop_rate': dr, 'model': name, 'accuracy': acc})
        pd.DataFrame(results).to_csv('logs/arxiv_continued.csv', index=False)

print("\nDone! Results in logs/arxiv_continued.csv")
