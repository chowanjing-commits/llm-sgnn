"""
Run equal-label-budget controls for training-node structural recovery.

The controls keep dropped nodes, sampled recovered nodes, and the successful
recovered training labels fixed across modes. Only edges and feature spaces vary.
"""
import argparse
import csv
import gc
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.serialization
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import EdgeStorage, GlobalStorage, NodeStorage
from torch_geometric.utils import add_self_loops

from scripts.preprocess.preprocess_arxiv import preprocess_arxiv, sample_arxiv_subgraph
from scripts.preprocess.preprocess_citation import preprocess_citation
from scripts.utils.run_single_dataset_pilot import build_corrupted_graph, sample_recovered_nodes
from src import config
from src.models import GAT, GCN, GraphSAGE

torch.serialization.add_safe_globals(
    [DataEdgeAttr, DataTensorAttr, Data, GlobalStorage, NodeStorage, EdgeStorage]
)


MODE_ORDER = [
    "sparse_llm",
    "semantic_edges_no_recovered_labels",
    "recovered_labels_no_edges",
    "recovered_labels_random_edges",
    "recovered_labels_mlp_edges",
    "recovered_labels_original_graph_mlp_edges",
    "ours_semantic_edges",
    "oracle_original_edges",
]

MODE_DISPLAY = {
    "sparse_llm": "Sparse LLM",
    "semantic_edges_no_recovered_labels": "Semantic Edges, No Recovered Labels",
    "recovered_labels_no_edges": "Recovered Labels, No Edges",
    "recovered_labels_random_edges": "Recovered Labels, Random Edges",
    "recovered_labels_mlp_edges": "Recovered Labels, Hadamard-MLP Edges",
    "recovered_labels_original_graph_mlp_edges": "Recovered Labels, Original-Graph MLP Edges",
    "ours_semantic_edges": "Ours: Recovered Labels + LLM Semantic Edges",
    "oracle_original_edges": "Recovered Labels, Oracle Original Edges",
}

RAW_FIELDNAMES = [
    "dataset",
    "backbone",
    "mode",
    "node_drop_rate",
    "seed",
    "accuracy",
    "dropped_train_nodes",
    "observed_train_nodes",
    "sampled_recovered_train_nodes",
    "successful_recovered_train_nodes",
    "train_nodes",
    "sparse_edges",
    "added_edges",
    "edge_hit_rate",
    "node_hit_rate",
    "avg_added_degree_per_successful",
    "repair_time_sec",
    "train_time_sec",
    "total_time_sec",
    "peak_gpu_mem_mb",
]


def parse_csv_floats(value):
    return [float(item) for item in value.split(",") if item.strip()]


def parse_csv_strings(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def append_raw_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def undirected_edge_set(edge_index):
    if edge_index.numel() == 0:
        return set()
    rows = edge_index[0].cpu().tolist()
    cols = edge_index[1].cpu().tolist()
    return {tuple(sorted((int(src), int(dst)))) for src, dst in zip(rows, cols) if src != dst}


def directed_edge_set(edge_index):
    if edge_index.numel() == 0:
        return set()
    rows = edge_index[0].cpu().tolist()
    cols = edge_index[1].cpu().tolist()
    return {(int(src), int(dst)) for src, dst in zip(rows, cols) if src != dst}


def adaptive_edge_budget(sparse_edge_index, observed_node_count, max_edges_per_node, k_neighbors):
    avg_observed_degree = (
        float(sparse_edge_index.size(1)) / max(int(observed_node_count), 1)
        if observed_node_count > 0
        else 0.0
    )
    return max(1, min(int(max_edges_per_node), int(k_neighbors), int(np.ceil(avg_observed_degree))))


def load_dataset(dataset, arxiv_subgraph_size):
    if dataset == "arxiv":
        data, _ = preprocess_arxiv()
        if arxiv_subgraph_size > 0:
            data = sample_arxiv_subgraph(data, num_nodes=arxiv_subgraph_size, seed=42)
            dataset_name = f"ogbn-arxiv-subgraph-{arxiv_subgraph_size}"
        else:
            dataset_name = "ogbn-arxiv"
    else:
        data, _ = preprocess_citation(dataset)
        dataset_name = "Cora" if dataset == "cora" else "PubMed"

    if hasattr(data, "text_available_mask"):
        data.repair_target_mask = data.text_available_mask
        data.semantic_candidate_mask = data.text_available_mask
    return data, dataset_name


def build_semantic_repair_edges(
    x_llm,
    sparse_edge_index,
    recovered_node_mask,
    candidate_node_mask,
    k_neighbors,
    max_edges_per_node,
    adaptive_threshold_alpha,
    observed_node_count,
):
    successful_recovered = torch.zeros_like(recovered_node_mask)
    if not recovered_node_mask.any() or max_edges_per_node <= 0:
        return sparse_edge_index.cpu(), successful_recovered, []

    recovered_nodes = torch.nonzero(recovered_node_mask, as_tuple=False).view(-1).cpu()
    candidate_nodes = torch.nonzero(candidate_node_mask, as_tuple=False).view(-1).cpu()
    if recovered_nodes.numel() == 0 or candidate_nodes.numel() == 0:
        return sparse_edge_index.cpu(), successful_recovered, []

    edge_budget = adaptive_edge_budget(
        sparse_edge_index=sparse_edge_index,
        observed_node_count=observed_node_count,
        max_edges_per_node=max_edges_per_node,
        k_neighbors=k_neighbors,
    )
    x = F.normalize(x_llm.detach().cpu(), p=2, dim=1)
    candidate_x = x[candidate_nodes].t().contiguous()
    existing_edges = directed_edge_set(sparse_edge_index)
    added_edges = []
    batch_size = 256
    topk = min(int(k_neighbors), candidate_nodes.numel())
    if topk <= 0:
        return sparse_edge_index.cpu(), successful_recovered, []

    for start in range(0, recovered_nodes.numel(), batch_size):
        batch_nodes = recovered_nodes[start : start + batch_size]
        sims = x[batch_nodes] @ candidate_x
        top_values, top_idx = torch.topk(sims, k=topk, dim=1, largest=True)

        for row_idx, node in enumerate(batch_nodes.tolist()):
            candidate_sims = top_values[row_idx]
            candidate_idx = top_idx[row_idx]
            threshold = candidate_sims.mean()
            if candidate_sims.numel() > 1:
                threshold = threshold + adaptive_threshold_alpha * candidate_sims.std(unbiased=False)
            valid_mask = candidate_sims >= threshold
            if not valid_mask.any():
                continue

            valid_idx = candidate_idx[valid_mask]
            valid_sims = candidate_sims[valid_mask]
            order = torch.argsort(valid_sims, descending=True)
            repaired = 0
            for candidate_pos in valid_idx[order].tolist():
                candidate = int(candidate_nodes[candidate_pos].item())
                if candidate == node:
                    continue
                src, dst = int(node), candidate
                if (src, dst) in existing_edges or (dst, src) in existing_edges:
                    continue
                added_edges.append((src, dst))
                added_edges.append((dst, src))
                existing_edges.add((src, dst))
                existing_edges.add((dst, src))
                repaired += 1
                if repaired >= edge_budget:
                    break
            if repaired > 0:
                successful_recovered[node] = True

        del sims, top_values, top_idx

    if not added_edges:
        return sparse_edge_index.cpu(), successful_recovered, []

    added_edge_index = torch.tensor(added_edges, dtype=torch.long).t().contiguous()
    repaired_edge_index = torch.cat([sparse_edge_index.cpu(), added_edge_index], dim=1)
    return repaired_edge_index, successful_recovered, added_edges


def build_random_matched_edges(sparse_edge_index, source_node_mask, candidate_node_mask, target_undirected_edges, seed):
    source_nodes = torch.nonzero(source_node_mask, as_tuple=False).view(-1).cpu().tolist()
    candidate_nodes = torch.nonzero(candidate_node_mask, as_tuple=False).view(-1).cpu().tolist()
    if not source_nodes or not candidate_nodes or target_undirected_edges <= 0:
        return sparse_edge_index.cpu(), []

    rng = np.random.default_rng(seed)
    existing_edges = directed_edge_set(sparse_edge_index)
    candidate_array = np.asarray(candidate_nodes, dtype=np.int64)
    source_order = list(source_nodes)
    rng.shuffle(source_order)
    added_edges = []
    attempts = 0
    max_attempts = max(target_undirected_edges * 200, 1000)

    while len(added_edges) // 2 < target_undirected_edges and attempts < max_attempts:
        src = int(source_order[attempts % len(source_order)])
        dst = int(candidate_array[rng.integers(0, len(candidate_array))])
        attempts += 1
        if src == dst:
            continue
        if (src, dst) in existing_edges or (dst, src) in existing_edges:
            continue
        added_edges.append((src, dst))
        added_edges.append((dst, src))
        existing_edges.add((src, dst))
        existing_edges.add((dst, src))

    if not added_edges:
        return sparse_edge_index.cpu(), []

    added_edge_index = torch.tensor(added_edges, dtype=torch.long).t().contiguous()
    return torch.cat([sparse_edge_index.cpu(), added_edge_index], dim=1), added_edges


class HadamardEdgeMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, left, right):
        return self.net(left * right).view(-1)


class ConcatEdgeMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, left, right):
        return self.net(torch.cat([left, right], dim=1)).view(-1)


def sample_edge_scorer_pairs(sparse_edge_index, candidate_node_mask, max_positive_edges, seed):
    candidate_nodes = torch.nonzero(candidate_node_mask, as_tuple=False).view(-1).cpu().tolist()
    candidate_set = set(candidate_nodes)
    if not candidate_nodes:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0)

    sparse_edges = undirected_edge_set(sparse_edge_index)
    positives = [(src, dst) for src, dst in sparse_edges if src in candidate_set and dst in candidate_set]
    if not positives:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0)

    rng = np.random.default_rng(seed)
    if len(positives) > max_positive_edges:
        idx = rng.choice(len(positives), size=max_positive_edges, replace=False)
        positives = [positives[int(i)] for i in idx]

    negatives = []
    candidate_array = np.asarray(candidate_nodes, dtype=np.int64)
    attempts = 0
    max_attempts = max(len(positives) * 100, 1000)
    while len(negatives) < len(positives) and attempts < max_attempts:
        src = int(candidate_array[rng.integers(0, len(candidate_array))])
        dst = int(candidate_array[rng.integers(0, len(candidate_array))])
        attempts += 1
        if src == dst:
            continue
        edge = tuple(sorted((src, dst)))
        if edge in sparse_edges:
            continue
        negatives.append(edge)

    pairs = positives + negatives
    labels = [1.0] * len(positives) + [0.0] * len(negatives)
    if not pairs:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0)
    left = torch.tensor([src for src, _ in pairs], dtype=torch.long)
    right = torch.tensor([dst for _, dst in pairs], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float)
    perm = torch.randperm(y.numel(), generator=torch.Generator().manual_seed(seed))
    return left[perm], right[perm], y[perm]


def sample_original_graph_edge_pairs(original_edge_index, num_nodes, max_positive_edges, seed):
    original_edges = list(undirected_edge_set(original_edge_index))
    if not original_edges:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0)

    rng = np.random.default_rng(seed)
    if len(original_edges) > max_positive_edges:
        idx = rng.choice(len(original_edges), size=max_positive_edges, replace=False)
        positives = [original_edges[int(i)] for i in idx]
    else:
        positives = original_edges

    original_edge_set = set(original_edges)
    negatives = []
    attempts = 0
    max_attempts = max(len(positives) * 200, 1000)
    while len(negatives) < len(positives) and attempts < max_attempts:
        src = int(rng.integers(0, num_nodes))
        dst = int(rng.integers(0, num_nodes))
        attempts += 1
        if src == dst:
            continue
        edge = tuple(sorted((src, dst)))
        if edge in original_edge_set:
            continue
        negatives.append(edge)

    pairs = positives + negatives
    labels = [1.0] * len(positives) + [0.0] * len(negatives)
    left = torch.tensor([src for src, _ in pairs], dtype=torch.long)
    right = torch.tensor([dst for _, dst in pairs], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float)
    perm = torch.randperm(y.numel(), generator=torch.Generator().manual_seed(seed))
    return left[perm], right[perm], y[perm]


def train_edge_scorer(
    x_llm,
    sparse_edge_index,
    candidate_node_mask,
    max_positive_edges,
    hidden_channels,
    dropout,
    num_epochs,
    batch_size,
    seed,
):
    left, right, labels = sample_edge_scorer_pairs(
        sparse_edge_index=sparse_edge_index,
        candidate_node_mask=candidate_node_mask,
        max_positive_edges=max_positive_edges,
        seed=seed,
    )
    if labels.numel() == 0:
        return None

    device = config.DEVICE
    x = F.normalize(x_llm.detach(), p=2, dim=1).to(device)
    model = HadamardEdgeMLP(x.size(1), hidden_channels, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    left = left.to(device)
    right = right.to(device)
    labels = labels.to(device)

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(labels.numel(), device=device)
        for start in range(0, labels.numel(), batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(x[left[idx]], x[right[idx]])
            loss = F.binary_cross_entropy_with_logits(logits, labels[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def train_original_graph_edge_scorer(
    x_llm,
    original_edge_index,
    num_nodes,
    max_positive_edges,
    hidden_channels,
    dropout,
    num_epochs,
    batch_size,
    seed,
):
    left, right, labels = sample_original_graph_edge_pairs(
        original_edge_index=original_edge_index,
        num_nodes=num_nodes,
        max_positive_edges=max_positive_edges,
        seed=seed,
    )
    if labels.numel() == 0:
        return None

    device = config.DEVICE
    x = F.normalize(x_llm.detach(), p=2, dim=1).to(device)
    model = ConcatEdgeMLP(x.size(1), hidden_channels, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    left = left.to(device)
    right = right.to(device)
    labels = labels.to(device)

    for epoch in range(num_epochs):
        model.train()
        perm = torch.randperm(labels.numel(), device=device)
        for start in range(0, labels.numel(), batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            logits_forward = model(x[left[idx]], x[right[idx]])
            logits_backward = model(x[right[idx]], x[left[idx]])
            logits = 0.5 * (logits_forward + logits_backward)
            loss = F.binary_cross_entropy_with_logits(logits, labels[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def build_mlp_matched_edges(
    x_llm,
    sparse_edge_index,
    source_node_mask,
    candidate_node_mask,
    edge_scorer,
    target_undirected_edges,
    max_edges_per_node,
    k_prescreen,
    seed,
):
    if edge_scorer is None or target_undirected_edges <= 0 or not source_node_mask.any():
        return sparse_edge_index.cpu(), []

    source_nodes = torch.nonzero(source_node_mask, as_tuple=False).view(-1).cpu()
    candidate_nodes = torch.nonzero(candidate_node_mask, as_tuple=False).view(-1).cpu()
    if source_nodes.numel() == 0 or candidate_nodes.numel() == 0:
        return sparse_edge_index.cpu(), []

    rng = np.random.default_rng(seed)
    order = torch.tensor(rng.permutation(source_nodes.numpy()), dtype=torch.long)
    x_cpu = F.normalize(x_llm.detach().cpu(), p=2, dim=1)
    candidate_x = x_cpu[candidate_nodes].t().contiguous()
    existing_edges = directed_edge_set(sparse_edge_index)
    added_edges = []
    device = config.DEVICE
    x_device = x_cpu.to(device)
    topk = min(int(k_prescreen), candidate_nodes.numel())
    if topk <= 0:
        return sparse_edge_index.cpu(), []

    with torch.no_grad():
        for start in range(0, order.numel(), 128):
            batch_nodes = order[start : start + 128]
            sims = x_cpu[batch_nodes] @ candidate_x
            _, top_idx = torch.topk(sims, k=topk, dim=1, largest=True)
            for row_idx, node in enumerate(batch_nodes.tolist()):
                candidate_idx = top_idx[row_idx]
                candidate_ids = candidate_nodes[candidate_idx]
                src_batch = torch.full_like(candidate_ids, int(node))
                scores = edge_scorer(x_device[src_batch.to(device)], x_device[candidate_ids.to(device)]).detach().cpu()
                score_order = torch.argsort(scores, descending=True)
                repaired = 0
                for candidate_pos in score_order.tolist():
                    candidate = int(candidate_ids[candidate_pos].item())
                    if candidate == node:
                        continue
                    src, dst = int(node), candidate
                    if (src, dst) in existing_edges or (dst, src) in existing_edges:
                        continue
                    added_edges.append((src, dst))
                    added_edges.append((dst, src))
                    existing_edges.add((src, dst))
                    existing_edges.add((dst, src))
                    repaired += 1
                    if repaired >= max_edges_per_node:
                        break
                    if len(added_edges) // 2 >= target_undirected_edges:
                        break
                if len(added_edges) // 2 >= target_undirected_edges:
                    break
            del sims, top_idx
            if len(added_edges) // 2 >= target_undirected_edges:
                break

    if not added_edges:
        return sparse_edge_index.cpu(), []

    added_edge_index = torch.tensor(added_edges, dtype=torch.long).t().contiguous()
    return torch.cat([sparse_edge_index.cpu(), added_edge_index], dim=1), added_edges


def build_original_graph_mlp_threshold_edges(
    x_llm,
    sparse_edge_index,
    source_node_mask,
    candidate_node_mask,
    edge_scorer,
    target_undirected_edges,
    max_edges_per_node,
    similarity_threshold,
    seed,
):
    if edge_scorer is None or target_undirected_edges <= 0 or not source_node_mask.any():
        return sparse_edge_index.cpu(), []

    source_nodes = torch.nonzero(source_node_mask, as_tuple=False).view(-1).cpu()
    candidate_nodes = torch.nonzero(candidate_node_mask, as_tuple=False).view(-1).cpu()
    if source_nodes.numel() == 0 or candidate_nodes.numel() == 0:
        return sparse_edge_index.cpu(), []

    rng = np.random.default_rng(seed)
    source_order = torch.tensor(rng.permutation(source_nodes.numpy()), dtype=torch.long)
    x_cpu = F.normalize(x_llm.detach().cpu(), p=2, dim=1)
    candidate_x = x_cpu[candidate_nodes].t().contiguous()
    existing_edges = directed_edge_set(sparse_edge_index)
    added_edges = []
    device = config.DEVICE
    x_device = x_cpu.to(device)

    with torch.no_grad():
        for start in range(0, source_order.numel(), 128):
            batch_nodes = source_order[start : start + 128]
            sims = x_cpu[batch_nodes] @ candidate_x
            for row_idx, node in enumerate(batch_nodes.tolist()):
                valid_mask = sims[row_idx] >= similarity_threshold
                if not valid_mask.any():
                    continue
                candidate_ids = candidate_nodes[valid_mask]
                if candidate_ids.numel() == 0:
                    continue
                src_batch = torch.full_like(candidate_ids, int(node))
                forward = edge_scorer(x_device[src_batch.to(device)], x_device[candidate_ids.to(device)])
                backward = edge_scorer(x_device[candidate_ids.to(device)], x_device[src_batch.to(device)])
                scores = (forward + backward).detach().cpu()
                score_order = torch.argsort(scores, descending=True)
                repaired = 0
                for candidate_pos in score_order.tolist():
                    candidate = int(candidate_ids[candidate_pos].item())
                    if candidate == node:
                        continue
                    src, dst = int(node), candidate
                    if (src, dst) in existing_edges or (dst, src) in existing_edges:
                        continue
                    added_edges.append((src, dst))
                    added_edges.append((dst, src))
                    existing_edges.add((src, dst))
                    existing_edges.add((dst, src))
                    repaired += 1
                    if repaired >= max_edges_per_node:
                        break
                    if len(added_edges) // 2 >= target_undirected_edges:
                        break
                if len(added_edges) // 2 >= target_undirected_edges:
                    break
            del sims
            if len(added_edges) // 2 >= target_undirected_edges:
                break

    if not added_edges:
        return sparse_edge_index.cpu(), []

    added_edge_index = torch.tensor(added_edges, dtype=torch.long).t().contiguous()
    return torch.cat([sparse_edge_index.cpu(), added_edge_index], dim=1), added_edges


def build_oracle_original_edges(sparse_edge_index, original_edge_index, recovered_label_mask, active_node_mask):
    if not recovered_label_mask.any():
        return sparse_edge_index.cpu(), []

    existing_edges = directed_edge_set(sparse_edge_index)
    added_edges = []
    rows = original_edge_index[0].cpu().tolist()
    cols = original_edge_index[1].cpu().tolist()
    for src, dst in zip(rows, cols):
        src = int(src)
        dst = int(dst)
        if src == dst:
            continue
        if not (active_node_mask[src].item() and active_node_mask[dst].item()):
            continue
        if not (recovered_label_mask[src].item() or recovered_label_mask[dst].item()):
            continue
        if (src, dst) in existing_edges:
            continue
        added_edges.append((src, dst))
        existing_edges.add((src, dst))

    if not added_edges:
        return sparse_edge_index.cpu(), []

    added_edge_index = torch.tensor(added_edges, dtype=torch.long).t().contiguous()
    return torch.cat([sparse_edge_index.cpu(), added_edge_index], dim=1), added_edges


def diagnose_added_edges(added_edges, recovered_label_mask, original_edges, sparse_edges):
    added_undirected = {tuple(sorted((src, dst))) for src, dst in added_edges if src != dst}
    novel_original_edges = original_edges - sparse_edges
    hit_edges = added_undirected & novel_original_edges
    recovered_nodes = set(torch.nonzero(recovered_label_mask, as_tuple=False).view(-1).cpu().tolist())
    hit_nodes = {node for edge in hit_edges for node in edge if node in recovered_nodes}
    successful_count = len(recovered_nodes)
    added_count = len(added_undirected)
    return {
        "added_edges": added_count,
        "edge_hit_rate": len(hit_edges) / added_count if added_count > 0 else 0.0,
        "node_hit_rate": len(hit_nodes) / successful_count if successful_count > 0 else 0.0,
        "avg_added_degree_per_successful": added_count / successful_count if successful_count > 0 else 0.0,
    }


def build_model(backbone, in_channels, out_channels):
    if backbone == "gcn":
        return GCN(in_channels, config.GNN_HIDDEN_DIM, out_channels)
    if backbone == "gat":
        return GAT(in_channels, config.GNN_HIDDEN_DIM, out_channels, heads=config.GNN_HEADS)
    if backbone == "sage":
        return GraphSAGE(in_channels, config.GNN_HIDDEN_DIM, out_channels)
    raise ValueError(f"Unknown backbone: {backbone}")


def train_gnn(data, edge_index, train_mask, num_epochs, backbone, feature_space):
    device = config.DEVICE
    x = data.x_llm if feature_space == "llm" else data.x
    x = x.to(device)
    y = data.y.to(device)
    edge_index = edge_index.to(device)
    edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
    model = build_model(backbone, x.size(1), int(data.y.max().item() + 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    train_mask = train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)
    best_val = 0.0
    best_test = 0.0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            model.eval()
            with torch.no_grad():
                out = model(x, edge_index)
                pred = out.argmax(dim=1)
                val_acc = (pred[val_mask] == y[val_mask]).float().mean().item() if val_mask.any() else 0.0
                test_acc = (pred[test_mask] == y[test_mask]).float().mean().item() if test_mask.any() else 0.0
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return best_test


def run_mode(
    data,
    dataset_name,
    backbone,
    mode,
    node_drop_rate,
    seed,
    num_epochs,
    edge_index,
    train_mask,
    feature_space,
    sparse_edge_index,
    added_edges,
    recovered_label_mask,
    original_edges,
    sparse_edges,
    repair_time,
):
    cuda_enabled = torch.cuda.is_available() and config.DEVICE.type == "cuda"
    if cuda_enabled:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    config.set_seed(seed + 10000)
    acc = train_gnn(data, edge_index, train_mask, num_epochs, backbone, feature_space)
    if cuda_enabled:
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t0
    peak_gpu_mem = float(torch.cuda.max_memory_allocated() / (1024**2)) if cuda_enabled else float("nan")
    diag = diagnose_added_edges(
        added_edges=added_edges,
        recovered_label_mask=recovered_label_mask,
        original_edges=original_edges,
        sparse_edges=sparse_edges,
    )
    return {
        "dataset": dataset_name,
        "backbone": backbone,
        "mode": MODE_DISPLAY[mode],
        "node_drop_rate": node_drop_rate,
        "seed": seed,
        "accuracy": acc,
        "train_nodes": int(train_mask.sum().item()),
        "sparse_edges": len(sparse_edges),
        **diag,
        "repair_time_sec": repair_time,
        "train_time_sec": train_time,
        "total_time_sec": repair_time + train_time,
        "peak_gpu_mem_mb": peak_gpu_mem,
    }


def main():
    parser = argparse.ArgumentParser(description="Equal-label-budget controls for node recovery")
    parser.add_argument("--datasets", type=str, default="cora,pubmed,arxiv")
    parser.add_argument("--node-drop-rates", type=str, default="0.95")
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--recovery-ratio", type=float, default=0.5)
    parser.add_argument("--max-edges-per-recovered-node", type=int, default=10)
    parser.add_argument("--adaptive-threshold-alpha", type=float, default=0.0)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--arxiv-subgraph-size", type=int, default=0)
    parser.add_argument("--backbone", type=str, default="sage", choices=["gcn", "gat", "sage"])
    parser.add_argument("--modes", type=str, default=",".join(MODE_ORDER))
    parser.add_argument("--edge-scorer-epochs", type=int, default=50)
    parser.add_argument("--edge-scorer-hidden-dim", type=int, default=128)
    parser.add_argument("--edge-scorer-dropout", type=float, default=0.2)
    parser.add_argument("--edge-scorer-batch-size", type=int, default=8192)
    parser.add_argument("--edge-scorer-max-positive-edges", type=int, default=100000)
    parser.add_argument("--edge-scorer-prescreen-k", type=int, default=100)
    parser.add_argument("--original-graph-mlp-threshold", type=float, default=0.6)
    parser.add_argument("--output-prefix", type=str, default="label_budget_controls")
    args = parser.parse_args()

    datasets = parse_csv_strings(args.datasets)
    node_drop_rates = parse_csv_floats(args.node_drop_rates)
    requested_modes = parse_csv_strings(args.modes)
    unknown_modes = sorted(set(requested_modes) - set(MODE_ORDER))
    if unknown_modes:
        raise ValueError(f"Unknown modes: {unknown_modes}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.output_prefix.strip() or "label_budget_controls"
    raw_path = PROJECT_ROOT / "logs" / f"{prefix}_raw_{timestamp}.csv"
    summary_path = PROJECT_ROOT / "logs" / f"{prefix}_summary_{timestamp}.csv"
    rows = []
    raw_rows = []

    for dataset in datasets:
        data, dataset_name = load_dataset(dataset, args.arxiv_subgraph_size)
        print("=" * 80)
        print(f"Dataset: {dataset_name} nodes={data.num_nodes} edges={data.edge_index.size(1)}")
        print("=" * 80)

        for node_drop_rate in node_drop_rates:
            per_mode = {MODE_DISPLAY[mode]: [] for mode in requested_modes}

            for seed in range(args.num_runs):
                config.set_seed(seed)
                sparse_edge_index, dropped_node_mask = build_corrupted_graph(
                    data,
                    edge_drop_rate=args.drop_rate,
                    node_drop_rate=node_drop_rate,
                )
                sampled_recovered_mask = sample_recovered_nodes(
                    data,
                    dropped_node_mask=dropped_node_mask,
                    recovery_ratio=args.recovery_ratio,
                )
                observed_train_mask = data.train_mask & ~dropped_node_mask
                candidate_node_mask = ~dropped_node_mask
                if hasattr(data, "semantic_candidate_mask"):
                    candidate_node_mask = candidate_node_mask & data.semantic_candidate_mask
                observed_node_count = int((~dropped_node_mask).sum().item())
                original_edges = undirected_edge_set(data.edge_index)
                sparse_edges = undirected_edge_set(sparse_edge_index)

                repair_t0 = time.perf_counter()
                semantic_edge_index, successful_recovered_mask, semantic_added_edges = build_semantic_repair_edges(
                    x_llm=data.x_llm,
                    sparse_edge_index=sparse_edge_index,
                    recovered_node_mask=sampled_recovered_mask,
                    candidate_node_mask=candidate_node_mask,
                    k_neighbors=args.k_neighbors,
                    max_edges_per_node=args.max_edges_per_recovered_node,
                    adaptive_threshold_alpha=args.adaptive_threshold_alpha,
                    observed_node_count=observed_node_count,
                )
                semantic_repair_time = time.perf_counter() - repair_t0
                recovered_label_mask = successful_recovered_mask & data.train_mask
                recovered_train_mask = observed_train_mask | recovered_label_mask

                random_t0 = time.perf_counter()
                random_edge_index, random_added_edges = build_random_matched_edges(
                    sparse_edge_index=sparse_edge_index,
                    source_node_mask=recovered_label_mask,
                    candidate_node_mask=candidate_node_mask,
                    target_undirected_edges=len(semantic_added_edges) // 2,
                    seed=seed + 20000,
                )
                random_repair_time = time.perf_counter() - random_t0

                mlp_edge_index = sparse_edge_index
                mlp_added_edges = []
                mlp_repair_time = 0.0
                if "recovered_labels_mlp_edges" in requested_modes:
                    mlp_t0 = time.perf_counter()
                    edge_scorer = train_edge_scorer(
                        x_llm=data.x_llm,
                        sparse_edge_index=sparse_edge_index,
                        candidate_node_mask=candidate_node_mask,
                        max_positive_edges=args.edge_scorer_max_positive_edges,
                        hidden_channels=args.edge_scorer_hidden_dim,
                        dropout=args.edge_scorer_dropout,
                        num_epochs=args.edge_scorer_epochs,
                        batch_size=args.edge_scorer_batch_size,
                        seed=seed + 30000,
                    )
                    mlp_edge_budget = adaptive_edge_budget(
                        sparse_edge_index=sparse_edge_index,
                        observed_node_count=observed_node_count,
                        max_edges_per_node=args.max_edges_per_recovered_node,
                        k_neighbors=args.k_neighbors,
                    )
                    mlp_edge_index, mlp_added_edges = build_mlp_matched_edges(
                        x_llm=data.x_llm,
                        sparse_edge_index=sparse_edge_index,
                        source_node_mask=recovered_label_mask,
                        candidate_node_mask=candidate_node_mask,
                        edge_scorer=edge_scorer,
                        target_undirected_edges=len(semantic_added_edges) // 2,
                        max_edges_per_node=mlp_edge_budget,
                        k_prescreen=args.edge_scorer_prescreen_k,
                        seed=seed + 40000,
                    )
                    mlp_repair_time = time.perf_counter() - mlp_t0
                    del edge_scorer
                    torch.cuda.empty_cache()

                original_mlp_edge_index = sparse_edge_index
                original_mlp_added_edges = []
                original_mlp_repair_time = 0.0
                if "recovered_labels_original_graph_mlp_edges" in requested_modes:
                    original_mlp_t0 = time.perf_counter()
                    original_edge_scorer = train_original_graph_edge_scorer(
                        x_llm=data.x_llm,
                        original_edge_index=data.edge_index,
                        num_nodes=data.num_nodes,
                        max_positive_edges=args.edge_scorer_max_positive_edges,
                        hidden_channels=args.edge_scorer_hidden_dim,
                        dropout=args.edge_scorer_dropout,
                        num_epochs=args.edge_scorer_epochs,
                        batch_size=args.edge_scorer_batch_size,
                        seed=seed + 50000,
                    )
                    original_mlp_edge_budget = adaptive_edge_budget(
                        sparse_edge_index=sparse_edge_index,
                        observed_node_count=observed_node_count,
                        max_edges_per_node=args.max_edges_per_recovered_node,
                        k_neighbors=args.k_neighbors,
                    )
                    original_mlp_edge_index, original_mlp_added_edges = build_original_graph_mlp_threshold_edges(
                        x_llm=data.x_llm,
                        sparse_edge_index=sparse_edge_index,
                        source_node_mask=recovered_label_mask,
                        candidate_node_mask=candidate_node_mask,
                        edge_scorer=original_edge_scorer,
                        target_undirected_edges=len(semantic_added_edges) // 2,
                        max_edges_per_node=original_mlp_edge_budget,
                        similarity_threshold=args.original_graph_mlp_threshold,
                        seed=seed + 60000,
                    )
                    original_mlp_repair_time = time.perf_counter() - original_mlp_t0
                    del original_edge_scorer
                    torch.cuda.empty_cache()

                active_node_mask = (~dropped_node_mask) | recovered_label_mask
                oracle_t0 = time.perf_counter()
                oracle_edge_index, oracle_added_edges = build_oracle_original_edges(
                    sparse_edge_index=sparse_edge_index,
                    original_edge_index=data.edge_index,
                    recovered_label_mask=recovered_label_mask,
                    active_node_mask=active_node_mask,
                )
                oracle_repair_time = time.perf_counter() - oracle_t0

                mode_configs = {
                    "sparse_llm": (
                        sparse_edge_index,
                        observed_train_mask,
                        "llm",
                        [],
                        torch.zeros_like(recovered_label_mask),
                        0.0,
                    ),
                    "semantic_edges_no_recovered_labels": (
                        semantic_edge_index,
                        observed_train_mask,
                        "llm",
                        semantic_added_edges,
                        recovered_label_mask,
                        semantic_repair_time,
                    ),
                    "recovered_labels_no_edges": (
                        sparse_edge_index,
                        recovered_train_mask,
                        "llm",
                        [],
                        recovered_label_mask,
                        0.0,
                    ),
                    "recovered_labels_random_edges": (
                        random_edge_index,
                        recovered_train_mask,
                        "llm",
                        random_added_edges,
                        recovered_label_mask,
                        random_repair_time,
                    ),
                    "recovered_labels_mlp_edges": (
                        mlp_edge_index,
                        recovered_train_mask,
                        "llm",
                        mlp_added_edges,
                        recovered_label_mask,
                        mlp_repair_time,
                    ),
                    "recovered_labels_original_graph_mlp_edges": (
                        original_mlp_edge_index,
                        recovered_train_mask,
                        "llm",
                        original_mlp_added_edges,
                        recovered_label_mask,
                        original_mlp_repair_time,
                    ),
                    "ours_semantic_edges": (
                        semantic_edge_index,
                        recovered_train_mask,
                        "llm",
                        semantic_added_edges,
                        recovered_label_mask,
                        semantic_repair_time,
                    ),
                    "oracle_original_edges": (
                        oracle_edge_index,
                        recovered_train_mask,
                        "llm",
                        oracle_added_edges,
                        recovered_label_mask,
                        oracle_repair_time,
                    ),
                }

                dropped_train_nodes = int((dropped_node_mask & data.train_mask).sum().item())
                sampled_recovered_train_nodes = int((sampled_recovered_mask & data.train_mask).sum().item())
                successful_recovered_train_nodes = int(recovered_label_mask.sum().item())
                for mode in requested_modes:
                    edge_index, train_mask, feature_space, added_edges, mode_recovered_mask, repair_time = mode_configs[mode]
                    row = run_mode(
                        data=data,
                        dataset_name=dataset_name,
                        backbone=args.backbone,
                        mode=mode,
                        node_drop_rate=node_drop_rate,
                        seed=seed,
                        num_epochs=args.num_epochs,
                        edge_index=edge_index,
                        train_mask=train_mask,
                        feature_space=feature_space,
                        sparse_edge_index=sparse_edge_index,
                        added_edges=added_edges,
                        recovered_label_mask=mode_recovered_mask,
                        original_edges=original_edges,
                        sparse_edges=sparse_edges,
                        repair_time=repair_time,
                    )
                    row.update(
                        {
                            "dropped_train_nodes": dropped_train_nodes,
                            "observed_train_nodes": int(observed_train_mask.sum().item()),
                            "sampled_recovered_train_nodes": sampled_recovered_train_nodes,
                            "successful_recovered_train_nodes": successful_recovered_train_nodes
                            if mode != "sparse_llm"
                            else 0,
                        }
                    )
                    raw_rows.append(row)
                    append_raw_row(raw_path, row)
                    per_mode[MODE_DISPLAY[mode]].append(row)
                    print(
                        f"{dataset_name} drop={node_drop_rate:.2f} seed={seed} "
                        f"{MODE_DISPLAY[mode]} acc={row['accuracy'] * 100:.2f} "
                        f"train={row['train_nodes']} recovered={row['successful_recovered_train_nodes']} "
                        f"edges={row['added_edges']} repair={row['repair_time_sec']:.2f}s "
                        f"train_time={row['train_time_sec']:.2f}s"
                    )

                gc.collect()

            for mode_name, stats in per_mode.items():
                if not stats:
                    continue
                rows.append(
                    {
                        "dataset": dataset_name,
                        "backbone": args.backbone,
                        "mode": mode_name,
                        "node_drop_rate": node_drop_rate,
                        "accuracy": float(np.mean([item["accuracy"] for item in stats])),
                        "std": float(np.std([item["accuracy"] for item in stats])),
                        "dropped_train_nodes_mean": float(np.mean([item["dropped_train_nodes"] for item in stats])),
                        "observed_train_nodes_mean": float(np.mean([item["observed_train_nodes"] for item in stats])),
                        "sampled_recovered_train_nodes_mean": float(
                            np.mean([item["sampled_recovered_train_nodes"] for item in stats])
                        ),
                        "successful_recovered_train_nodes_mean": float(
                            np.mean([item["successful_recovered_train_nodes"] for item in stats])
                        ),
                        "train_nodes_mean": float(np.mean([item["train_nodes"] for item in stats])),
                        "sparse_edges_mean": float(np.mean([item["sparse_edges"] for item in stats])),
                        "added_edges_mean": float(np.mean([item["added_edges"] for item in stats])),
                        "edge_hit_rate_mean": float(np.mean([item["edge_hit_rate"] for item in stats])),
                        "node_hit_rate_mean": float(np.mean([item["node_hit_rate"] for item in stats])),
                        "avg_added_degree_per_successful_mean": float(
                            np.mean([item["avg_added_degree_per_successful"] for item in stats])
                        ),
                        "repair_time_sec_mean": float(np.mean([item["repair_time_sec"] for item in stats])),
                        "train_time_sec_mean": float(np.mean([item["train_time_sec"] for item in stats])),
                        "total_time_sec_mean": float(np.mean([item["total_time_sec"] for item in stats])),
                        "peak_gpu_mem_mb_mean": safe_nanmean([item["peak_gpu_mem_mb"] for item in stats]),
                        "num_runs": args.num_runs,
                        "num_epochs": args.num_epochs,
                        "k_neighbors": args.k_neighbors,
                        "recovery_ratio": args.recovery_ratio,
                        "max_edges_per_recovered_node": args.max_edges_per_recovered_node,
                        "adaptive_threshold_alpha": args.adaptive_threshold_alpha,
                        "edge_scorer_epochs": args.edge_scorer_epochs,
                        "edge_scorer_hidden_dim": args.edge_scorer_hidden_dim,
                        "edge_scorer_prescreen_k": args.edge_scorer_prescreen_k,
                        "original_graph_mlp_threshold": args.original_graph_mlp_threshold,
                    }
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)
    print("\nSummary")
    print(
        summary[
            [
                "dataset",
                "backbone",
                "mode",
                "node_drop_rate",
                "accuracy",
                "std",
                "observed_train_nodes_mean",
                "successful_recovered_train_nodes_mean",
                "train_nodes_mean",
                "added_edges_mean",
                "edge_hit_rate_mean",
                "train_time_sec_mean",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved raw rows to: {raw_path}")


if __name__ == "__main__":
    main()
