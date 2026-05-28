"""
Diagnose semantic recovery quality without training.

The script measures whether semantic recovery edges correspond to original
graph neighbors removed by node-drop corruption.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.serialization
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import EdgeStorage, GlobalStorage, NodeStorage

from scripts.preprocess.preprocess_arxiv import preprocess_arxiv, sample_arxiv_subgraph
from scripts.preprocess.preprocess_citation import preprocess_citation
from scripts.preprocess.preprocess_wikics import get_wikics_split, preprocess_wikics
from scripts.utils.run_single_dataset_pilot import build_corrupted_graph, sample_recovered_nodes
from src import config

torch.serialization.add_safe_globals(
    [DataEdgeAttr, DataTensorAttr, Data, GlobalStorage, NodeStorage, EdgeStorage]
)


def canonical_pair(src, dst):
    src = int(src)
    dst = int(dst)
    return (src, dst) if src < dst else (dst, src)


def build_undirected_edge_set(edge_index):
    if edge_index.numel() == 0:
        return set()
    rows = edge_index[0].cpu().tolist()
    cols = edge_index[1].cpu().tolist()
    return {canonical_pair(src, dst) for src, dst in zip(rows, cols) if src != dst}


def build_neighbor_sets(edge_index, num_nodes):
    neighbors = [set() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return neighbors
    rows = edge_index[0].cpu().tolist()
    cols = edge_index[1].cpu().tolist()
    for src, dst in zip(rows, cols):
        src = int(src)
        dst = int(dst)
        if src == dst:
            continue
        neighbors[src].add(dst)
        neighbors[dst].add(src)
    return neighbors


def get_dataset(dataset, arxiv_subgraph_size):
    if dataset == "wikics":
        data, _ = preprocess_wikics()
        data = get_wikics_split(data, 0)
        dataset_name = "WikiCS"
    elif dataset == "arxiv":
        data, _ = preprocess_arxiv()
        if arxiv_subgraph_size > 0:
            data = sample_arxiv_subgraph(data, num_nodes=arxiv_subgraph_size, seed=42)
            dataset_name = f"ogbn-arxiv-subgraph-{arxiv_subgraph_size}"
        else:
            dataset_name = "ogbn-arxiv-full"
    elif dataset in {"cora", "pubmed"}:
        data, _ = preprocess_citation(dataset)
        dataset_name = dataset.upper() if dataset == "cora" else "PubMed"
        if hasattr(data, "text_available_mask"):
            data.repair_target_mask = data.text_available_mask
            data.semantic_candidate_mask = data.text_available_mask
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return data, dataset_name


def diagnose_once(
    data,
    dataset_name,
    seed,
    node_drop_rate,
    edge_drop_rate,
    k_neighbors,
    recovery_ratio,
    repair_policy,
    similarity_threshold,
    adaptive_threshold_alpha,
    max_edges_per_recovered_node,
    chunk_size,
):
    sparse_edge_index, dropped_node_mask = build_corrupted_graph(
        data,
        edge_drop_rate=edge_drop_rate,
        node_drop_rate=node_drop_rate,
    )
    candidate_node_mask = ~dropped_node_mask
    if hasattr(data, "semantic_candidate_mask"):
        candidate_node_mask = candidate_node_mask & data.semantic_candidate_mask

    sampled_recovered_mask = sample_recovered_nodes(data, dropped_node_mask, recovery_ratio)
    recovered_nodes = torch.nonzero(sampled_recovered_mask, as_tuple=False).view(-1).cpu()
    candidate_nodes = torch.nonzero(candidate_node_mask, as_tuple=False).view(-1).cpu()
    observed_node_count = int((~dropped_node_mask).sum().item())

    original_edges = build_undirected_edge_set(data.edge_index)
    sparse_edges = build_undirected_edge_set(sparse_edge_index)
    original_neighbors = build_neighbor_sets(data.edge_index, data.num_nodes)
    candidate_node_set = set(candidate_nodes.tolist())
    x = F.normalize(data.x_llm.detach().cpu(), p=2, dim=1)

    edge_budget = max_edges_per_recovered_node
    if repair_policy == "adaptive":
        avg_observed_degree = float(sparse_edge_index.size(1)) / max(observed_node_count, 1)
        edge_budget = min(max_edges_per_recovered_node, max(1, int(np.ceil(avg_observed_degree))))

    added_edges = []
    successful_nodes = set()
    hit_nodes = set()
    existing_edges = set(sparse_edges)

    topk_total = 0
    topk_hit_total = 0
    topk_neighbor_total = 0
    topk_neighbor_hit_total = 0

    if recovered_nodes.numel() == 0 or candidate_nodes.numel() == 0:
        return {
            "dataset": dataset_name,
            "seed": seed,
            "node_drop_rate": node_drop_rate,
            "drop_rate": edge_drop_rate,
            "k_neighbors": k_neighbors,
            "recovery_ratio": recovery_ratio,
            "repair_policy": repair_policy,
            "similarity_threshold": similarity_threshold,
            "adaptive_threshold_alpha": adaptive_threshold_alpha,
            "max_edges_per_recovered_node": max_edges_per_recovered_node,
            "effective_edge_budget": edge_budget,
            "dropped_nodes": int(dropped_node_mask.sum().item()),
            "observed_train_nodes": int((data.train_mask & ~dropped_node_mask).sum().item()),
            "sampled_recovered_nodes": int(sampled_recovered_mask.sum().item()),
            "successful_recovered_nodes": 0,
            "recovery_edges": 0,
            "recovery_edge_hits": 0,
            "edge_hit_rate": 0.0,
            "node_hit_rate": 0.0,
            "topk_precision": 0.0,
            "topk_recall": 0.0,
            "avg_recovery_edges_per_successful_node": 0.0,
            "avg_hit_edges_per_successful_node": 0.0,
        }

    for start in range(0, recovered_nodes.numel(), chunk_size):
        end = min(start + chunk_size, recovered_nodes.numel())
        chunk_nodes = recovered_nodes[start:end]
        sims = x[chunk_nodes] @ x[candidate_nodes].t()
        topk = min(k_neighbors, candidate_nodes.numel()) if k_neighbors > 0 else candidate_nodes.numel()
        topk_values, topk_positions = torch.topk(sims, k=topk, dim=1, largest=True)

        for row_idx, node in enumerate(chunk_nodes.tolist()):
            top_positions = topk_positions[row_idx]
            top_values = topk_values[row_idx]
            top_candidates = candidate_nodes[top_positions].tolist()
            original_candidate_neighbors = original_neighbors[int(node)].intersection(candidate_node_set)

            topk_total += len(top_candidates)
            top_hits = sum(1 for candidate in top_candidates if candidate in original_neighbors[int(node)])
            topk_hit_total += top_hits
            topk_neighbor_total += len(original_candidate_neighbors)
            topk_neighbor_hit_total += len(set(top_candidates).intersection(original_candidate_neighbors))

            if repair_policy == "adaptive":
                threshold = top_values.mean()
                if top_values.numel() > 1:
                    threshold = threshold + adaptive_threshold_alpha * top_values.std(unbiased=False)
                valid_positions = top_positions[top_values >= threshold]
            else:
                valid_positions = top_positions[top_values >= similarity_threshold]

            if valid_positions.numel() == 0:
                continue

            valid_values = sims[row_idx, valid_positions]
            order = torch.argsort(valid_values, descending=True)
            repaired = 0
            for candidate_pos in valid_positions[order].tolist():
                candidate = int(candidate_nodes[candidate_pos].item())
                if candidate == node:
                    continue
                edge = canonical_pair(node, candidate)
                if edge in existing_edges:
                    continue
                added_edges.append(edge)
                existing_edges.add(edge)
                repaired += 1
                if edge in original_edges:
                    hit_nodes.add(int(node))
                if repaired >= edge_budget:
                    break
            if repaired > 0:
                successful_nodes.add(int(node))

    recovery_edges = len(added_edges)
    recovery_edge_hits = sum(1 for edge in added_edges if edge in original_edges)
    successful_recovered_nodes = len(successful_nodes)
    edge_hit_rate = recovery_edge_hits / recovery_edges if recovery_edges > 0 else 0.0
    node_hit_rate = len(hit_nodes) / successful_recovered_nodes if successful_recovered_nodes > 0 else 0.0
    topk_precision = topk_hit_total / topk_total if topk_total > 0 else 0.0
    topk_recall = topk_neighbor_hit_total / topk_neighbor_total if topk_neighbor_total > 0 else 0.0

    return {
        "dataset": dataset_name,
        "seed": seed,
        "node_drop_rate": node_drop_rate,
        "drop_rate": edge_drop_rate,
        "k_neighbors": k_neighbors,
        "recovery_ratio": recovery_ratio,
        "repair_policy": repair_policy,
        "similarity_threshold": similarity_threshold,
        "adaptive_threshold_alpha": adaptive_threshold_alpha,
        "max_edges_per_recovered_node": max_edges_per_recovered_node,
        "effective_edge_budget": edge_budget,
        "dropped_nodes": int(dropped_node_mask.sum().item()),
        "observed_train_nodes": int((data.train_mask & ~dropped_node_mask).sum().item()),
        "sampled_recovered_nodes": int(sampled_recovered_mask.sum().item()),
        "successful_recovered_nodes": successful_recovered_nodes,
        "recovery_edges": recovery_edges,
        "recovery_edge_hits": recovery_edge_hits,
        "edge_hit_rate": edge_hit_rate,
        "node_hit_rate": node_hit_rate,
        "topk_precision": topk_precision,
        "topk_recall": topk_recall,
        "avg_recovery_edges_per_successful_node": (
            recovery_edges / successful_recovered_nodes if successful_recovered_nodes > 0 else 0.0
        ),
        "avg_hit_edges_per_successful_node": (
            recovery_edge_hits / successful_recovered_nodes if successful_recovered_nodes > 0 else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose semantic recovery quality")
    parser.add_argument("--datasets", type=str, default="cora,pubmed,arxiv")
    parser.add_argument("--node-drop-rates", type=str, default="0.75,0.85,0.95")
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--recovery-ratio", type=float, default=0.5)
    parser.add_argument("--repair-policy", type=str, default="adaptive", choices=["fixed", "adaptive"])
    parser.add_argument("--similarity-threshold", type=float, default=0.6)
    parser.add_argument("--adaptive-threshold-alpha", type=float, default=0.0)
    parser.add_argument("--max-edges-per-recovered-node", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--arxiv-subgraph-size", type=int, default=0)
    parser.add_argument("--num-runs", type=int, default=1)
    args = parser.parse_args()

    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    node_drop_rates = [float(value) for value in args.node_drop_rates.split(",") if value.strip()]
    rows = []

    for dataset in datasets:
        data, dataset_name = get_dataset(dataset, args.arxiv_subgraph_size)
        print(f"Loaded {dataset_name}")
        for node_drop_rate in node_drop_rates:
            for seed in range(args.num_runs):
                config.set_seed(seed)
                row = diagnose_once(
                    data=data,
                    dataset_name=dataset_name,
                    seed=seed,
                    node_drop_rate=node_drop_rate,
                    edge_drop_rate=args.drop_rate,
                    k_neighbors=args.k_neighbors,
                    recovery_ratio=args.recovery_ratio,
                    repair_policy=args.repair_policy,
                    similarity_threshold=args.similarity_threshold,
                    adaptive_threshold_alpha=args.adaptive_threshold_alpha,
                    max_edges_per_recovered_node=args.max_edges_per_recovered_node,
                    chunk_size=args.chunk_size,
                )
                rows.append(row)
                print(
                    f"{dataset_name} node_drop={node_drop_rate:.2f} seed={seed} "
                    f"recovery_edges={row['recovery_edges']} "
                    f"edge_hit_rate={row['edge_hit_rate']:.4f} "
                    f"node_hit_rate={row['node_hit_rate']:.4f} "
                    f"topk_precision={row['topk_precision']:.4f} "
                    f"topk_recall={row['topk_recall']:.4f}"
                )

    result_df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROJECT_ROOT / "logs" / f"recovery_quality_{args.repair_policy}_{timestamp}.csv"
    result_df.to_csv(out_path, index=False)
    metric_cols = [
        "successful_recovered_nodes",
        "recovery_edges",
        "recovery_edge_hits",
        "edge_hit_rate",
        "node_hit_rate",
        "topk_precision",
        "topk_recall",
        "avg_recovery_edges_per_successful_node",
        "avg_hit_edges_per_successful_node",
    ]
    summary_df = (
        result_df.groupby(["dataset", "node_drop_rate"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary_df.columns
    ]
    summary_path = (
        PROJECT_ROOT / "logs" / f"recovery_quality_{args.repair_policy}_summary_{timestamp}.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    print("\nSummary")
    print(summary_df.to_string(index=False))
    print(f"\nSaved to: {out_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
