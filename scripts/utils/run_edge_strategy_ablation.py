"""
Compare semantic edge construction strategies under the same node-drop protocol.

This ablation keeps the dropped/recovered training nodes fixed across strategies
and varies only how semantic recovery edges are selected.
"""
import argparse
import csv
import gc
import time
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.serialization
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import EdgeStorage, GlobalStorage, NodeStorage
from torch_geometric.utils import add_self_loops

from scripts.preprocess.preprocess_arxiv import preprocess_arxiv, sample_arxiv_subgraph
from scripts.preprocess.preprocess_citation import preprocess_citation
from scripts.utils.run_single_dataset_pilot import (
    build_corrupted_graph,
    sample_recovered_nodes,
)
from src import config
from src.models import GAT, GCN, GraphSAGE

torch.serialization.add_safe_globals(
    [DataEdgeAttr, DataTensorAttr, Data, GlobalStorage, NodeStorage, EdgeStorage]
)


def undirected_edge_set(edge_index):
    if edge_index.numel() == 0:
        return set()
    rows = edge_index[0].cpu().tolist()
    cols = edge_index[1].cpu().tolist()
    return {tuple(sorted((int(src), int(dst)))) for src, dst in zip(rows, cols) if src != dst}


def make_cluster_labels(x_llm, num_clusters, seed):
    from sklearn.cluster import MiniBatchKMeans

    x = F.normalize(x_llm.detach().cpu(), p=2, dim=1).numpy()
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=seed,
        batch_size=4096,
        n_init="auto",
        max_iter=100,
    )
    return torch.tensor(kmeans.fit_predict(x), dtype=torch.long)


def adaptive_edge_budget(sparse_edge_index, observed_node_count, max_edges_per_node, k_neighbors):
    avg_observed_degree = (
        float(sparse_edge_index.size(1)) / max(int(observed_node_count), 1)
        if observed_node_count > 0
        else 0.0
    )
    return max(1, min(int(max_edges_per_node), int(k_neighbors), int(np.ceil(avg_observed_degree))))


def inverse_adaptive_edge_budget(sparse_edge_index, observed_node_count, max_edges_per_node, k_neighbors):
    normal_budget = adaptive_edge_budget(
        sparse_edge_index=sparse_edge_index,
        observed_node_count=observed_node_count,
        max_edges_per_node=max_edges_per_node,
        k_neighbors=k_neighbors,
    )
    upper = min(int(max_edges_per_node), int(k_neighbors))
    return max(1, upper - normal_budget + 1)


def build_strategy_edges(
    x_llm,
    sparse_edge_index,
    recovered_node_mask,
    candidate_node_mask,
    strategy,
    k_neighbors,
    max_edges_per_node,
    adaptive_threshold_alpha,
    observed_node_count,
):
    if not recovered_node_mask.any() or max_edges_per_node <= 0:
        return sparse_edge_index, torch.zeros_like(recovered_node_mask), []

    recovered_nodes = torch.nonzero(recovered_node_mask, as_tuple=False).view(-1).cpu()
    candidate_nodes = torch.nonzero(candidate_node_mask, as_tuple=False).view(-1).cpu()
    if recovered_nodes.numel() == 0 or candidate_nodes.numel() == 0:
        return sparse_edge_index, torch.zeros_like(recovered_node_mask), []

    x = F.normalize(x_llm.detach().cpu(), p=2, dim=1)
    candidate_x = x[candidate_nodes].t().contiguous()
    if strategy == "inverse_adaptive":
        edge_budget = inverse_adaptive_edge_budget(
            sparse_edge_index=sparse_edge_index,
            observed_node_count=observed_node_count,
            max_edges_per_node=max_edges_per_node,
            k_neighbors=k_neighbors,
        )
    else:
        edge_budget = adaptive_edge_budget(
            sparse_edge_index=sparse_edge_index,
            observed_node_count=observed_node_count,
            max_edges_per_node=max_edges_per_node,
            k_neighbors=k_neighbors,
        )
    existing_edges = set()
    if sparse_edge_index.numel() > 0:
        existing_edges = set(zip(sparse_edge_index[0].cpu().tolist(), sparse_edge_index[1].cpu().tolist()))

    added_edges = []
    successful_recovered = torch.zeros_like(recovered_node_mask)
    batch_size = 256
    topk = min(int(k_neighbors), candidate_nodes.numel())
    if topk <= 0:
        return sparse_edge_index, successful_recovered, []

    for start in range(0, recovered_nodes.numel(), batch_size):
        batch_nodes = recovered_nodes[start : start + batch_size]
        batch_sims = x[batch_nodes] @ candidate_x
        top_values, top_idx = torch.topk(batch_sims, k=topk, dim=1, largest=True)

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

        del batch_sims, top_values, top_idx

    if not added_edges:
        return sparse_edge_index, successful_recovered, []

    added_edge_index = torch.tensor(added_edges, dtype=torch.long).t().contiguous()
    repaired_edge_index = torch.cat([sparse_edge_index.cpu(), added_edge_index], dim=1)
    return repaired_edge_index, successful_recovered, added_edges


def build_active_knn_edges(
    x_llm,
    sparse_edge_index,
    recovered_node_mask,
    active_node_mask,
    strategy,
    k_neighbors,
    cluster_labels=None,
):
    active_nodes = torch.nonzero(active_node_mask, as_tuple=False).view(-1).cpu()
    if active_nodes.numel() == 0 or k_neighbors <= 0:
        return sparse_edge_index, torch.zeros_like(recovered_node_mask), []

    x = F.normalize(x_llm.detach().cpu(), p=2, dim=1)
    existing_edges = set()
    if sparse_edge_index.numel() > 0:
        existing_edges = set(zip(sparse_edge_index[0].cpu().tolist(), sparse_edge_index[1].cpu().tolist()))

    added_edges = []
    successful_recovered = torch.zeros_like(recovered_node_mask)
    batch_size = 128

    if strategy == "global_active_knn":
        candidate_nodes = active_nodes
        candidate_x = x[candidate_nodes].t().contiguous()
        topk = min(int(k_neighbors) + 1, candidate_nodes.numel())
        for start in range(0, active_nodes.numel(), batch_size):
            batch_nodes = active_nodes[start : start + batch_size]
            sims = x[batch_nodes] @ candidate_x
            _, top_idx = torch.topk(sims, k=topk, dim=1, largest=True)
            for row_idx, src_node in enumerate(batch_nodes.tolist()):
                repaired = 0
                for candidate_pos in top_idx[row_idx].tolist():
                    dst_node = int(candidate_nodes[candidate_pos].item())
                    if dst_node == src_node:
                        continue
                    if (src_node, dst_node) in existing_edges or (dst_node, src_node) in existing_edges:
                        continue
                    added_edges.append((src_node, dst_node))
                    added_edges.append((dst_node, src_node))
                    existing_edges.add((src_node, dst_node))
                    existing_edges.add((dst_node, src_node))
                    if recovered_node_mask[src_node].item():
                        successful_recovered[src_node] = True
                    if recovered_node_mask[dst_node].item():
                        successful_recovered[dst_node] = True
                    repaired += 1
                    if repaired >= k_neighbors:
                        break
            del sims, top_idx
    elif strategy == "cluster_active_knn":
        if cluster_labels is None:
            raise ValueError("cluster_labels is required for cluster_active_knn")
        active_clusters = cluster_labels[active_nodes]
        cluster_to_nodes = {
            int(cluster_id.item()): active_nodes[active_clusters == cluster_id]
            for cluster_id in torch.unique(active_clusters)
        }
        for src_node in active_nodes.tolist():
            node_cluster = int(cluster_labels[src_node].item())
            candidate_nodes = cluster_to_nodes.get(node_cluster)
            if candidate_nodes is None:
                continue
            topk = min(int(k_neighbors) + 1, candidate_nodes.numel())
            if topk <= 1:
                continue
            sims = x[src_node].view(1, -1) @ x[candidate_nodes].t().contiguous()
            _, top_idx = torch.topk(sims.view(-1), k=topk, largest=True)
            repaired = 0
            for candidate_pos in top_idx.tolist():
                dst_node = int(candidate_nodes[candidate_pos].item())
                if dst_node == src_node:
                    continue
                if (src_node, dst_node) in existing_edges or (dst_node, src_node) in existing_edges:
                    continue
                added_edges.append((src_node, dst_node))
                added_edges.append((dst_node, src_node))
                existing_edges.add((src_node, dst_node))
                existing_edges.add((dst_node, src_node))
                if recovered_node_mask[src_node].item():
                    successful_recovered[src_node] = True
                if recovered_node_mask[dst_node].item():
                    successful_recovered[dst_node] = True
                repaired += 1
                if repaired >= k_neighbors:
                    break
            del sims, top_idx
    else:
        raise ValueError(f"Unknown active-node kNN strategy: {strategy}")

    if not added_edges:
        return sparse_edge_index, successful_recovered, []

    added_edge_index = torch.tensor(added_edges, dtype=torch.long).t().contiguous()
    repaired_edge_index = torch.cat([sparse_edge_index.cpu(), added_edge_index], dim=1)
    return repaired_edge_index, successful_recovered, added_edges


def train_gnn(data, edge_index, train_mask, num_epochs, backbone):
    device = config.DEVICE
    x = data.x_llm.to(device)
    y = data.y.to(device)
    edge_index = edge_index.to(device)
    edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)
    in_channels = data.x_llm.size(1)
    out_channels = int(data.y.max().item() + 1)
    if backbone == "gcn":
        model = GCN(in_channels, config.GNN_HIDDEN_DIM, out_channels).to(device)
    elif backbone == "gat":
        model = GAT(in_channels, config.GNN_HIDDEN_DIM, out_channels).to(device)
    elif backbone == "sage":
        model = GraphSAGE(in_channels, config.GNN_HIDDEN_DIM, out_channels).to(device)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
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


def diagnose_added_edges(added_edges, successful_recovered_mask, original_edges, sparse_edges, active_node_count):
    added_undirected = {tuple(sorted((src, dst))) for src, dst in added_edges if src != dst}
    novel_original_edges = original_edges - sparse_edges
    hit_edges = added_undirected & novel_original_edges
    successful_nodes = torch.nonzero(successful_recovered_mask, as_tuple=False).view(-1).cpu().tolist()
    hit_nodes = {node for edge in hit_edges for node in edge if successful_recovered_mask[node].item()}
    semantic_added_edges = len(added_undirected)
    successful_count = len(successful_nodes)
    return {
        "semantic_added_edges": semantic_added_edges,
        "successful_recovered_nodes": successful_count,
        "edge_hit_rate": len(hit_edges) / semantic_added_edges if semantic_added_edges > 0 else 0.0,
        "node_hit_rate": len(hit_nodes) / successful_count if successful_count > 0 else 0.0,
        "avg_added_degree_per_successful": semantic_added_edges / successful_count if successful_count > 0 else 0.0,
        "added_edges_per_active_node": semantic_added_edges / max(int(active_node_count), 1),
        "added_edges_over_sparse_edges": semantic_added_edges / max(len(sparse_edges), 1),
    }


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


def parse_csv_floats(value):
    return [float(item) for item in value.split(",") if item.strip()]


def parse_csv_strings(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


RAW_FIELDNAMES = [
    "dataset",
    "backbone",
    "strategy",
    "node_drop_rate",
    "seed",
    "accuracy",
    "dropped_nodes",
    "observed_train_nodes",
    "sampled_recovered_nodes",
    "successful_recovered_nodes",
    "semantic_added_edges",
    "edge_hit_rate",
    "node_hit_rate",
    "avg_added_degree_per_successful",
    "added_edges_per_active_node",
    "added_edges_over_sparse_edges",
    "repair_time_sec",
    "train_time_sec",
    "total_time_sec",
    "peak_gpu_mem_mb",
    "cluster_time_sec",
]


def append_raw_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Semantic edge construction strategy ablation")
    parser.add_argument("--datasets", type=str, default="cora,pubmed,arxiv")
    parser.add_argument("--node-drop-rates", type=str, default="0.95")
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--recovery-ratio", type=float, default=0.5)
    parser.add_argument("--max-edges-per-recovered-node", type=int, default=10)
    parser.add_argument("--adaptive-threshold-alpha", type=float, default=0.0)
    parser.add_argument("--support-size", type=int, default=10)
    parser.add_argument("--support-internal-edges", type=int, default=2)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--arxiv-subgraph-size", type=int, default=0)
    parser.add_argument("--include-sparse-baseline", action="store_true")
    parser.add_argument(
        "--backbone",
        type=str,
        default="sage",
        choices=["gcn", "gat", "sage"],
        help="Downstream GNN backbone.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="ours_adaptive,inverse_adaptive,global_active_knn,cluster_active_knn",
        help="Comma-separated strategy list.",
    )
    parser.add_argument("--output-prefix", type=str, default="")
    args = parser.parse_args()

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    node_drop_rates = parse_csv_floats(args.node_drop_rates)
    valid_strategies = {
        "ours_adaptive",
        "inverse_adaptive",
        "global_active_knn",
        "cluster_active_knn",
    }
    strategies = parse_csv_strings(args.strategies)
    unknown = sorted(set(strategies) - valid_strategies)
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}")
    rows = []
    raw_rows = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    raw_path = PROJECT_ROOT / "logs" / f"{prefix}edge_strategy_ablation_raw_{timestamp}.csv"
    summary_path = PROJECT_ROOT / "logs" / f"{prefix}edge_strategy_ablation_summary_{timestamp}.csv"

    for dataset in datasets:
        data, dataset_name = load_dataset(dataset, args.arxiv_subgraph_size)
        num_clusters = int(data.y.max().item() + 1)

        print("=" * 80)
        print(f"Dataset: {dataset_name} nodes={data.num_nodes} edges={data.edge_index.size(1)}")
        print("=" * 80)

        for node_drop_rate in node_drop_rates:
            per_mode = {strategy: [] for strategy in strategies}
            if args.include_sparse_baseline:
                per_mode[f"{args.backbone}_llm_sparse"] = []

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
                candidate_node_mask = ~dropped_node_mask
                if hasattr(data, "semantic_candidate_mask"):
                    candidate_node_mask = candidate_node_mask & data.semantic_candidate_mask
                observed_train_mask = data.train_mask & ~dropped_node_mask
                observed_node_count = int((~dropped_node_mask).sum().item())
                original_edges = undirected_edge_set(data.edge_index)
                sparse_edges = undirected_edge_set(sparse_edge_index)
                cluster_t0 = time.perf_counter()
                cluster_labels = make_cluster_labels(data.x_llm, num_clusters=num_clusters, seed=seed)
                cluster_time = time.perf_counter() - cluster_t0

                if args.include_sparse_baseline:
                    cuda_enabled = torch.cuda.is_available() and config.DEVICE.type == "cuda"
                    if cuda_enabled:
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    config.set_seed(seed + 10000)
                    acc = train_gnn(data, sparse_edge_index, observed_train_mask, args.num_epochs, args.backbone)
                    if cuda_enabled:
                        torch.cuda.synchronize()
                    train_time = time.perf_counter() - t0
                    peak_gpu_mem = (
                        float(torch.cuda.max_memory_allocated() / (1024**2)) if cuda_enabled else float("nan")
                    )
                    row = {
                        "dataset": dataset_name,
                        "backbone": args.backbone,
                        "strategy": f"{args.backbone}_llm_sparse",
                        "node_drop_rate": node_drop_rate,
                        "seed": seed,
                        "accuracy": acc,
                        "dropped_nodes": int(dropped_node_mask.sum().item()),
                        "observed_train_nodes": int(observed_train_mask.sum().item()),
                        "sampled_recovered_nodes": 0,
                        "successful_recovered_nodes": 0,
                        "semantic_added_edges": 0,
                        "edge_hit_rate": 0.0,
                        "node_hit_rate": 0.0,
                        "avg_added_degree_per_successful": 0.0,
                        "added_edges_per_active_node": 0.0,
                        "added_edges_over_sparse_edges": 0.0,
                        "repair_time_sec": 0.0,
                        "train_time_sec": train_time,
                        "total_time_sec": train_time,
                        "peak_gpu_mem_mb": peak_gpu_mem,
                        "cluster_time_sec": 0.0,
                    }
                    raw_rows.append(row)
                    append_raw_row(raw_path, row)
                    per_mode[f"{args.backbone}_llm_sparse"].append(row)
                    print(
                        f"{dataset_name} drop={node_drop_rate:.2f} seed={seed} "
                        f"sparse acc={acc * 100:.2f} train={train_time:.2f}s gpu={peak_gpu_mem:.1f}MB"
                    )

                for strategy in strategies:
                    cuda_enabled = torch.cuda.is_available() and config.DEVICE.type == "cuda"
                    if cuda_enabled:
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    if strategy in {"global_active_knn", "cluster_active_knn"}:
                        active_node_mask = (~dropped_node_mask) | sampled_recovered_mask
                        if hasattr(data, "semantic_candidate_mask"):
                            active_node_mask = active_node_mask & data.semantic_candidate_mask
                        repaired_edge_index, successful_recovered_mask, added_edges = build_active_knn_edges(
                            x_llm=data.x_llm,
                            sparse_edge_index=sparse_edge_index,
                            recovered_node_mask=sampled_recovered_mask,
                            active_node_mask=active_node_mask,
                            strategy=strategy,
                            k_neighbors=args.k_neighbors,
                            cluster_labels=cluster_labels,
                        )
                    else:
                        repaired_edge_index, successful_recovered_mask, added_edges = build_strategy_edges(
                            x_llm=data.x_llm,
                            sparse_edge_index=sparse_edge_index,
                            recovered_node_mask=sampled_recovered_mask,
                            candidate_node_mask=candidate_node_mask,
                            strategy=strategy,
                            k_neighbors=args.k_neighbors,
                            max_edges_per_node=args.max_edges_per_recovered_node,
                            adaptive_threshold_alpha=args.adaptive_threshold_alpha,
                            observed_node_count=observed_node_count,
                        )
                    if cuda_enabled:
                        torch.cuda.synchronize()
                    repair_time = time.perf_counter() - t0
                    if strategy.startswith("cluster_"):
                        repair_time += cluster_time
                    if cuda_enabled:
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    train_mask = observed_train_mask | (successful_recovered_mask & data.train_mask)
                    config.set_seed(seed + 10000)
                    acc = train_gnn(data, repaired_edge_index, train_mask, args.num_epochs, args.backbone)
                    if cuda_enabled:
                        torch.cuda.synchronize()
                    train_time = time.perf_counter() - t1
                    total_time = repair_time + train_time
                    peak_gpu_mem = (
                        float(torch.cuda.max_memory_allocated() / (1024**2)) if cuda_enabled else float("nan")
                    )
                    diag = diagnose_added_edges(
                        added_edges=added_edges,
                        successful_recovered_mask=successful_recovered_mask,
                        original_edges=original_edges,
                        sparse_edges=sparse_edges,
                        active_node_count=observed_node_count + int(sampled_recovered_mask.sum().item()),
                    )
                    row = {
                        "dataset": dataset_name,
                        "backbone": args.backbone,
                        "strategy": strategy,
                        "node_drop_rate": node_drop_rate,
                        "seed": seed,
                        "accuracy": acc,
                        "dropped_nodes": int(dropped_node_mask.sum().item()),
                        "observed_train_nodes": int(observed_train_mask.sum().item()),
                        "sampled_recovered_nodes": int(sampled_recovered_mask.sum().item()),
                        **diag,
                        "repair_time_sec": repair_time,
                        "train_time_sec": train_time,
                        "total_time_sec": total_time,
                        "peak_gpu_mem_mb": peak_gpu_mem,
                        "cluster_time_sec": cluster_time if strategy.startswith("cluster_") else 0.0,
                    }
                    raw_rows.append(row)
                    append_raw_row(raw_path, row)
                    per_mode[strategy].append(row)
                    print(
                        f"{dataset_name} drop={node_drop_rate:.2f} seed={seed} "
                        f"{strategy} acc={acc * 100:.2f} "
                        f"successful={diag['successful_recovered_nodes']} "
                        f"edges={diag['semantic_added_edges']} edge_hit={diag['edge_hit_rate'] * 100:.2f} "
                        f"repair={repair_time:.2f}s train={train_time:.2f}s gpu={peak_gpu_mem:.1f}MB"
                    )

                del cluster_labels
                gc.collect()

            for strategy, stats in per_mode.items():
                if not stats:
                    continue
                rows.append(
                    {
                        "dataset": dataset_name,
                        "backbone": args.backbone,
                        "strategy": strategy,
                        "node_drop_rate": node_drop_rate,
                        "accuracy": float(np.mean([item["accuracy"] for item in stats])),
                        "std": float(np.std([item["accuracy"] for item in stats])),
                        "dropped_nodes_mean": float(np.mean([item["dropped_nodes"] for item in stats])),
                        "observed_train_nodes_mean": float(np.mean([item["observed_train_nodes"] for item in stats])),
                        "sampled_recovered_nodes_mean": float(np.mean([item["sampled_recovered_nodes"] for item in stats])),
                        "successful_recovered_nodes_mean": float(np.mean([item["successful_recovered_nodes"] for item in stats])),
                        "semantic_added_edges_mean": float(np.mean([item["semantic_added_edges"] for item in stats])),
                        "edge_hit_rate_mean": float(np.mean([item["edge_hit_rate"] for item in stats])),
                        "node_hit_rate_mean": float(np.mean([item["node_hit_rate"] for item in stats])),
                        "avg_added_degree_per_successful_mean": float(
                            np.mean([item["avg_added_degree_per_successful"] for item in stats])
                        ),
                        "added_edges_per_active_node_mean": float(
                            np.mean([item["added_edges_per_active_node"] for item in stats])
                        ),
                        "added_edges_over_sparse_edges_mean": float(
                            np.mean([item["added_edges_over_sparse_edges"] for item in stats])
                        ),
                        "repair_time_sec_mean": float(np.mean([item["repair_time_sec"] for item in stats])),
                        "train_time_sec_mean": float(np.mean([item["train_time_sec"] for item in stats])),
                        "total_time_sec_mean": float(np.mean([item["total_time_sec"] for item in stats])),
                        "peak_gpu_mem_mb_mean": safe_nanmean([item["peak_gpu_mem_mb"] for item in stats]),
                        "cluster_time_sec_mean": float(np.mean([item["cluster_time_sec"] for item in stats])),
                        "num_runs": args.num_runs,
                        "k_neighbors": args.k_neighbors,
                        "recovery_ratio": args.recovery_ratio,
                        "max_edges_per_recovered_node": args.max_edges_per_recovered_node,
                        "adaptive_threshold_alpha": args.adaptive_threshold_alpha,
                        "cluster_count": num_clusters,
                        "support_size": args.support_size,
                        "support_internal_edges": args.support_internal_edges,
                    }
                )

    pd.DataFrame(rows).to_csv(summary_path, index=False)

    print("\nSummary")
    print(
        pd.DataFrame(rows)[
            [
                "dataset",
                "backbone",
                "strategy",
                "node_drop_rate",
                "accuracy",
                "std",
                "successful_recovered_nodes_mean",
                "semantic_added_edges_mean",
                "edge_hit_rate_mean",
                "node_hit_rate_mean",
                "added_edges_per_active_node_mean",
                "added_edges_over_sparse_edges_mean",
                "repair_time_sec_mean",
                "cluster_time_sec_mean",
                "train_time_sec_mean",
                "total_time_sec_mean",
                "peak_gpu_mem_mb_mean",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved raw rows to: {raw_path}")


if __name__ == "__main__":
    main()
