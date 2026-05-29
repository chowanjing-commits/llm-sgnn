"""
Run a single-dataset pilot experiment against the dropped-edge graph.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.serialization
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import EdgeStorage, GlobalStorage, NodeStorage

from src import config
from src.models import GAT, GCN, GraphSAGE, LLM_GNN, MLP
from scripts.preprocess.preprocess_arxiv import preprocess_arxiv, sample_arxiv_subgraph
from scripts.preprocess.preprocess_citation import preprocess_citation
from scripts.preprocess.preprocess_wikics import get_wikics_split, preprocess_wikics
from src.cold_start import ColdStartPipelineConfig, build_cold_start_training_state_from_config, safe_nanmean

torch.serialization.add_safe_globals(
    [DataEdgeAttr, DataTensorAttr, Data, GlobalStorage, NodeStorage, EdgeStorage]
)


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

    candidate_mask = data.train_mask.clone()
    if hasattr(data, "repair_target_mask"):
        candidate_mask = candidate_mask & data.repair_target_mask
    candidate_nodes = torch.nonzero(candidate_mask, as_tuple=False).view(-1)

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


def sample_recovered_nodes(data, dropped_node_mask, recovery_ratio):
    recovered = torch.zeros(data.num_nodes, dtype=torch.bool)
    if recovery_ratio <= 0:
        return recovered

    candidate_mask = data.train_mask & dropped_node_mask
    if hasattr(data, "repair_target_mask"):
        candidate_mask = candidate_mask & data.repair_target_mask
    candidate_nodes = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
    if candidate_nodes.numel() == 0:
        return recovered

    num_recover = int(candidate_nodes.numel() * recovery_ratio)
    if recovery_ratio > 0 and num_recover == 0:
        num_recover = 1
    num_recover = min(num_recover, candidate_nodes.numel())
    perm = torch.randperm(candidate_nodes.numel())[:num_recover]
    recovered[candidate_nodes[perm]] = True
    return recovered


def build_threshold_recovery_edges(
    x_llm,
    sparse_edge_index,
    recovered_node_mask,
    candidate_node_mask,
    k_neighbors,
    similarity_threshold,
    max_edges_per_node,
    repair_policy="fixed",
    adaptive_threshold_alpha=0.0,
    observed_node_count=None,
):
    if not recovered_node_mask.any() or max_edges_per_node <= 0:
        return sparse_edge_index, torch.zeros_like(recovered_node_mask), 0

    recovered_nodes = torch.nonzero(recovered_node_mask, as_tuple=False).view(-1)
    candidate_nodes = torch.nonzero(candidate_node_mask, as_tuple=False).view(-1)
    if recovered_nodes.numel() == 0 or candidate_nodes.numel() == 0:
        return sparse_edge_index, torch.zeros_like(recovered_node_mask), 0

    x = F.normalize(x_llm.detach().cpu(), p=2, dim=1)
    candidate_x = x[candidate_nodes].t().contiguous()
    added_edges = []
    successful_recovered = torch.zeros_like(recovered_node_mask)
    edge_budget = max_edges_per_node
    if repair_policy == "adaptive":
        if observed_node_count is None:
            observed_nodes = torch.unique(sparse_edge_index.cpu()) if sparse_edge_index.numel() > 0 else torch.empty(0)
            observed_node_count = int(observed_nodes.numel())
        avg_observed_degree = (
            float(sparse_edge_index.size(1)) / max(int(observed_node_count), 1)
            if observed_node_count > 0
            else 0.0
        )
        edge_budget = min(max_edges_per_node, max(1, int(np.ceil(avg_observed_degree))))

    existing_edges = set()
    if sparse_edge_index.numel() > 0:
        rows = sparse_edge_index[0].cpu().tolist()
        cols = sparse_edge_index[1].cpu().tolist()
        existing_edges = set(zip(rows, cols))

    batch_size = 256
    topk = min(k_neighbors, candidate_nodes.numel()) if k_neighbors > 0 else candidate_nodes.numel()
    for start in range(0, recovered_nodes.numel(), batch_size):
        batch_nodes = recovered_nodes[start : start + batch_size]
        batch_sims = x[batch_nodes] @ candidate_x
        if topk < candidate_nodes.numel():
            topk_sims, topk_idx = torch.topk(batch_sims, k=topk, dim=1, largest=True)
        else:
            topk_sims = batch_sims
            topk_idx = torch.arange(candidate_nodes.numel()).view(1, -1).expand(batch_sims.size(0), -1)

        for row_idx, node in enumerate(batch_nodes.tolist()):
            candidate_sims = topk_sims[row_idx]
            candidate_idx = topk_idx[row_idx]
            if repair_policy == "adaptive":
                threshold = candidate_sims.mean()
                if candidate_sims.numel() > 1:
                    threshold = threshold + adaptive_threshold_alpha * candidate_sims.std(unbiased=False)
                valid_mask = candidate_sims >= threshold
            else:
                valid_mask = candidate_sims >= similarity_threshold

            if not valid_mask.any():
                continue

            valid = candidate_idx[valid_mask]
            valid_sims = candidate_sims[valid_mask]
            order = torch.argsort(valid_sims, descending=True)
            repaired = 0
            for candidate_pos in valid[order].tolist():
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

        del batch_sims, topk_sims, topk_idx

    if not added_edges:
        return sparse_edge_index, successful_recovered, 0

    added_edge_index = torch.tensor(added_edges, dtype=torch.long).t().contiguous()
    repaired_edge_index = torch.cat([sparse_edge_index.cpu(), added_edge_index], dim=1)
    return repaired_edge_index, successful_recovered, len(added_edges) // 2


def is_repair_model(model_type):
    return model_type.startswith("LLM_GNN")


def repair_backbone(model_type):
    backbone_map = {
        "LLM_GNN": "gcn",
        "LLM_GNN_GCN": "gcn",
        "LLM_GNN_GAT": "gat",
        "LLM_GNN_SAGE": "sage",
    }
    return backbone_map[model_type]


def build_model(model_type, in_dim, num_classes, k_neighbors, beta):
    if model_type == "MLP":
        return MLP(in_dim, config.GNN_HIDDEN_DIM, num_classes)
    if model_type == "GCN":
        return GCN(in_dim, config.GNN_HIDDEN_DIM, num_classes)
    if model_type == "GAT":
        return GAT(in_dim, config.GNN_HIDDEN_DIM, num_classes, heads=config.GNN_HEADS)
    if model_type == "SAGE":
        return GraphSAGE(in_dim, config.GNN_HIDDEN_DIM, num_classes)
    if is_repair_model(model_type):
        return LLM_GNN(
            in_dim,
            config.GNN_HIDDEN_DIM,
            num_classes,
            k_neighbors=k_neighbors,
            beta=beta,
            backbone=repair_backbone(model_type),
            heads=config.GNN_HEADS,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def train_and_eval(
    data,
    model_type,
    use_llm,
    drop_rate,
    num_epochs,
    node_drop_rate=0.0,
    k_neighbors=config.DEFAULT_K_NEIGHBORS,
    beta=config.DEFAULT_BETA,
    recovery_ratio=0.0,
    similarity_threshold=0.6,
    max_edges_per_recovered_node=5,
    repair_policy="fixed",
    adaptive_threshold_alpha=0.0,
    cold_start_config=None,
    seed=0,
):
    device = config.DEVICE
    sparse_edge, dropped_node_mask = build_corrupted_graph(
        data,
        edge_drop_rate=drop_rate,
        node_drop_rate=node_drop_rate,
    )
    candidate_node_mask = ~dropped_node_mask
    if hasattr(data, "semantic_candidate_mask"):
        candidate_node_mask = candidate_node_mask & data.semantic_candidate_mask
    observed_train_mask = data.train_mask & ~dropped_node_mask
    train_mask = observed_train_mask
    train_y = data.y.clone()
    successful_recovered_mask = torch.zeros_like(dropped_node_mask)
    sampled_recovered_mask = torch.zeros_like(dropped_node_mask)
    added_recovery_edges = 0
    model_edge_index = sparse_edge
    cold_start_train_mask = data.train_mask & dropped_node_mask
    label_ready_mask = torch.zeros_like(dropped_node_mask)
    pseudo_train_mask = torch.zeros_like(dropped_node_mask)
    pseudo_label_accuracy = float("nan")
    pseudo_label_confidence_mean = float("nan")
    selected_center_distance_mean = float("nan")

    if is_repair_model(model_type) and cold_start_config is not None:
        if hasattr(data, "repair_target_mask"):
            cold_start_train_mask = cold_start_train_mask & data.repair_target_mask
        state = build_cold_start_training_state_from_config(
            x_llm=data.x_llm,
            y=data.y,
            sparse_edge_index=sparse_edge,
            cold_start_mask=cold_start_train_mask,
            observed_train_mask=observed_train_mask,
            candidate_node_mask=candidate_node_mask,
            num_classes=int(data.y.max().item() + 1),
            pipeline_config=cold_start_config,
            seed=seed,
            observed_node_count=int((~dropped_node_mask).sum().item()),
        )
        model_edge_index = state["edge_index"]
        train_mask = state["train_mask"]
        train_y = state["pseudo_y"]
        sampled_recovered_mask = state["selected_mask"]
        successful_recovered_mask = state["successful_recovered_mask"]
        label_ready_mask = state["label_ready_mask"]
        pseudo_train_mask = state["pseudo_train_mask"]
        added_recovery_edges = int(state["added_recovery_edges"])

        pseudo_nodes = torch.nonzero(pseudo_train_mask, as_tuple=False).view(-1)
        if pseudo_nodes.numel() > 0:
            pseudo_label_accuracy = (
                state["pseudo_y"][pseudo_nodes] == data.y[pseudo_nodes]
            ).float().mean().item()
            pseudo_label_confidence_mean = state["pseudo_confidence"][pseudo_nodes].mean().item()
            center_distance = state["center_distance"][pseudo_nodes]
            finite_center_distance = center_distance[torch.isfinite(center_distance)]
            if finite_center_distance.numel() > 0:
                selected_center_distance_mean = finite_center_distance.mean().item()
    elif is_repair_model(model_type):
        sampled_recovered_mask = sample_recovered_nodes(data, dropped_node_mask, recovery_ratio)
        model_edge_index, successful_recovered_mask, added_recovery_edges = build_threshold_recovery_edges(
            x_llm=data.x_llm,
            sparse_edge_index=sparse_edge,
            recovered_node_mask=sampled_recovered_mask,
            candidate_node_mask=candidate_node_mask,
            k_neighbors=k_neighbors,
            similarity_threshold=similarity_threshold,
            max_edges_per_node=max_edges_per_recovered_node,
            repair_policy=repair_policy,
            adaptive_threshold_alpha=adaptive_threshold_alpha,
            observed_node_count=int((~dropped_node_mask).sum().item()),
        )
        train_mask = observed_train_mask | (successful_recovered_mask & data.train_mask)

    in_dim = data.x_llm.size(1) if use_llm else data.x.size(1)
    num_classes = data.y.max().item() + 1

    model = build_model(model_type, in_dim, num_classes, k_neighbors=k_neighbors, beta=beta).to(device)
    exp_data = Data(
        x=data.x.to(device),
        x_llm=data.x_llm.to(device),
        edge_index=model_edge_index.to(device),
        original_edge_index=data.edge_index.to(device),
        y=data.y.to(device),
        train_y=train_y.to(device),
        train_mask=train_mask.to(device),
        val_mask=data.val_mask.to(device),
        test_mask=data.test_mask.to(device),
        dropped_node_mask=dropped_node_mask.to(device),
        candidate_node_mask=candidate_node_mask.to(device),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    x = exp_data.x_llm if use_llm else exp_data.x
    if not train_mask.any():
        raise RuntimeError(
            "No training nodes remain after graph corruption. "
            "Lower --node-drop-rate, raise --admission-ratio, or lower --pseudo-label-confidence."
        )
    best_val = 0.0
    best_test = 0.0
    effective_train_mask = train_mask.clone()
    gcn_edge_index = exp_data.edge_index
    if is_repair_model(model_type):
        gcn_edge_index, _ = add_self_loops(gcn_edge_index, num_nodes=exp_data.num_nodes)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        if is_repair_model(model_type):
            out = model.gnn(x, gcn_edge_index)
            effective_train_mask = train_mask
        else:
            out = model(x, exp_data.edge_index)
            effective_train_mask = train_mask
        loss = F.cross_entropy(out[effective_train_mask], exp_data.train_y[effective_train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            model.eval()
            with torch.no_grad():
                if is_repair_model(model_type):
                    out = model.gnn(x, gcn_edge_index)
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
    observed_train_count = int(observed_train_mask.sum().item())
    effective_train_count = int(effective_train_mask.sum().item())
    return {
        "accuracy": best_test,
        "sparse_edges": sparse_edge.size(1),
        "dropped_nodes": int(dropped_node_mask.sum().item()),
        "train_nodes": effective_train_count,
        "observed_train_nodes": observed_train_count,
        "cold_start_train_nodes": int(cold_start_train_mask.sum().item()),
        "sampled_recovered_nodes": int(sampled_recovered_mask.sum().item()),
        "label_ready_nodes": int(label_ready_mask.sum().item()),
        "successful_recovered_nodes": int(successful_recovered_mask.sum().item()),
        "pseudo_train_nodes": int(pseudo_train_mask.sum().item()),
        "recovery_edges": int(added_recovery_edges),
        "pseudo_label_accuracy": pseudo_label_accuracy,
        "pseudo_label_confidence_mean": pseudo_label_confidence_mean,
        "selected_center_distance_mean": selected_center_distance_mean,
    }


def main():
    parser = argparse.ArgumentParser(description="Run a single-dataset sparse-vs-repair pilot")
    parser.add_argument("--dataset", type=str, default="wikics", choices=["wikics", "arxiv", "cora", "pubmed"])
    parser.add_argument("--drop-rate", type=float, default=0.75)
    parser.add_argument("--node-drop-rate", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--split-indices", type=str, default="0")
    parser.add_argument("--k-neighbors", type=int, default=config.DEFAULT_K_NEIGHBORS)
    parser.add_argument("--beta", type=float, default=config.DEFAULT_BETA)
    parser.add_argument("--recovery-ratio", type=float, default=0.5)
    parser.add_argument("--similarity-threshold", type=float, default=0.6)
    parser.add_argument("--max-edges-per-recovered-node", type=int, default=5)
    parser.add_argument("--repair-policy", type=str, default="fixed", choices=["fixed", "adaptive"])
    parser.add_argument("--adaptive-threshold-alpha", type=float, default=0.0)
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="Use text-only cold-start admission and pseudo-labeling for recovered training nodes.",
    )
    parser.add_argument("--admission-ratio", type=float, default=0.5)
    parser.add_argument(
        "--admission-strategy",
        type=str,
        default="cluster_representative",
        choices=["cluster_representative", "random"],
    )
    parser.add_argument(
        "--pseudo-label-strategy",
        type=str,
        default="cluster_majority",
        choices=["cluster_majority", "nearest_labeled", "class_centroid"],
    )
    parser.add_argument("--pseudo-label-k", type=int, default=5)
    parser.add_argument("--pseudo-label-confidence", type=float, default=0.0)
    parser.add_argument(
        "--model-filter",
        type=str,
        default="all",
        choices=["all", "ours", "ours_gcn", "ours_gat", "ours_sage", "ours_gat_sage", "gat", "sage"],
    )
    parser.add_argument(
        "--arxiv-subgraph-size",
        type=int,
        default=10000,
        help="Use a sampled arXiv subgraph when >0; use the full graph when set to 0.",
    )
    args = parser.parse_args()

    config.set_seed()

    if args.dataset == "wikics":
        data, _ = preprocess_wikics()
        dataset_name = "WikiCS"
        split_indices = [int(x) for x in args.split_indices.split(",") if x.strip() != ""]
    elif args.dataset == "arxiv":
        data, _ = preprocess_arxiv()
        if args.arxiv_subgraph_size > 0:
            data = sample_arxiv_subgraph(data, num_nodes=args.arxiv_subgraph_size, seed=42)
            dataset_name = f"ogbn-arxiv-subgraph-{args.arxiv_subgraph_size}"
        else:
            dataset_name = "ogbn-arxiv-full"
        split_indices = [0]
    elif args.dataset in {"cora", "pubmed"}:
        data, _ = preprocess_citation(args.dataset)
        dataset_name = args.dataset.upper() if args.dataset == "cora" else "PubMed"
        if hasattr(data, "text_available_mask"):
            data.repair_target_mask = data.text_available_mask
            data.semantic_candidate_mask = data.text_available_mask
        split_indices = [0]


    base_edges = data.edge_index.size(1)
    print("=" * 70)
    print(f"Single-Dataset Pilot on {dataset_name}")
    print("=" * 70)
    print(f"Nodes: {data.num_nodes}")
    print(f"Original edges: {base_edges}")
    print(f"Drop rate: {args.drop_rate:.2f}")
    print(f"Node drop rate: {args.node_drop_rate:.2f}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Runs: {args.num_runs}")
    print(f"Splits: {split_indices}")
    print(f"k neighbors: {args.k_neighbors}")
    print(f"beta: {args.beta:.2f}")
    if args.cold_start:
        print("recovery ratio: unused with cold-start admission")
    else:
        print(f"recovery ratio: {args.recovery_ratio:.2f}")
    print(f"repair policy: {args.repair_policy}")
    print(f"similarity threshold: {args.similarity_threshold:.2f}")
    print(f"adaptive threshold alpha: {args.adaptive_threshold_alpha:.2f}")
    print(f"max edges per recovered node: {args.max_edges_per_recovered_node}")
    print(f"cold-start pipeline: {args.cold_start}")
    if args.cold_start:
        print(f"admission: {args.admission_strategy} ratio={args.admission_ratio:.2f}")
        print(
            f"pseudo labels: {args.pseudo_label_strategy} "
            f"k={args.pseudo_label_k} confidence>={args.pseudo_label_confidence:.2f}"
        )

    models = [
        ("MLP (Raw)", "MLP", False),
        ("MLP (LLM)", "MLP", True),
        ("GCN (Raw, Sparse)", "GCN", False),
        ("GCN (LLM, Sparse)", "GCN", True),
        ("GAT (Raw, Sparse)", "GAT", False),
        ("GAT (LLM, Sparse)", "GAT", True),
        ("GraphSAGE (Raw, Sparse)", "SAGE", False),
        ("GraphSAGE (LLM, Sparse)", "SAGE", True),
        ("Ours-GCN", "LLM_GNN_GCN", True),
        ("Ours-GAT", "LLM_GNN_GAT", True),
        ("Ours-GraphSAGE", "LLM_GNN_SAGE", True),
    ]
    if args.model_filter == "ours":
        models = [model for model in models if is_repair_model(model[1])]
    elif args.model_filter == "ours_gcn":
        models = [model for model in models if model[1] == "LLM_GNN_GCN"]
    elif args.model_filter == "ours_gat":
        models = [model for model in models if model[1] == "LLM_GNN_GAT"]
    elif args.model_filter == "ours_sage":
        models = [model for model in models if model[1] == "LLM_GNN_SAGE"]
    elif args.model_filter == "ours_gat_sage":
        models = [model for model in models if model[1] in {"LLM_GNN_GAT", "LLM_GNN_SAGE"}]
    elif args.model_filter == "gat":
        models = [model for model in models if model[1] == "GAT"]
    elif args.model_filter == "sage":
        models = [model for model in models if model[1] == "SAGE"]

    cold_start_config = None
    if args.cold_start:
        cold_start_config = ColdStartPipelineConfig(
            admission_ratio=args.admission_ratio,
            admission_strategy=args.admission_strategy,
            pseudo_label_strategy=args.pseudo_label_strategy,
            pseudo_label_k=args.pseudo_label_k,
            pseudo_label_confidence=args.pseudo_label_confidence,
            k_neighbors=args.k_neighbors,
            similarity_threshold=args.similarity_threshold,
            max_edges_per_node=args.max_edges_per_recovered_node,
            repair_policy=args.repair_policy,
            adaptive_threshold_alpha=args.adaptive_threshold_alpha,
        )

    rows = []
    for model_name, model_type, use_llm in models:
        print(f"\n{model_name}")
        accs = []
        kept_edges = []
        dropped_nodes = []
        train_nodes = []
        observed_train_nodes = []
        cold_start_train_nodes = []
        sampled_recovered_nodes = []
        label_ready_nodes = []
        successful_recovered_nodes = []
        pseudo_train_nodes = []
        recovery_edges = []
        pseudo_label_accuracies = []
        pseudo_label_confidences = []
        selected_center_distances = []
        for split_idx in split_indices:
            split_data = get_wikics_split(data, split_idx) if args.dataset == "wikics" else data
            split_accs = []
            for seed in range(args.num_runs):
                config.set_seed(seed + split_idx * 100)
                result = train_and_eval(
                    data=split_data,
                    model_type=model_type,
                    use_llm=use_llm,
                    drop_rate=args.drop_rate,
                    num_epochs=args.num_epochs,
                    node_drop_rate=args.node_drop_rate,
                    k_neighbors=args.k_neighbors,
                    beta=args.beta,
                    recovery_ratio=args.recovery_ratio,
                    similarity_threshold=args.similarity_threshold,
                    max_edges_per_recovered_node=args.max_edges_per_recovered_node,
                    repair_policy=args.repair_policy,
                    adaptive_threshold_alpha=args.adaptive_threshold_alpha,
                    cold_start_config=cold_start_config if is_repair_model(model_type) else None,
                    seed=seed + split_idx * 100,
                )
                acc = result["accuracy"]
                split_accs.append(acc)
                accs.append(acc)
                kept_edges.append(result["sparse_edges"])
                dropped_nodes.append(result["dropped_nodes"])
                train_nodes.append(result["train_nodes"])
                observed_train_nodes.append(result["observed_train_nodes"])
                cold_start_train_nodes.append(result["cold_start_train_nodes"])
                sampled_recovered_nodes.append(result["sampled_recovered_nodes"])
                label_ready_nodes.append(result["label_ready_nodes"])
                successful_recovered_nodes.append(result["successful_recovered_nodes"])
                pseudo_train_nodes.append(result["pseudo_train_nodes"])
                recovery_edges.append(result["recovery_edges"])
                pseudo_label_accuracies.append(result["pseudo_label_accuracy"])
                pseudo_label_confidences.append(result["pseudo_label_confidence_mean"])
                selected_center_distances.append(result["selected_center_distance_mean"])
                print(
                    f"  split={split_idx} seed={seed}: "
                    f"acc={acc * 100:.2f}% sparse_edges={result['sparse_edges']} "
                    f"dropped_nodes={result['dropped_nodes']} train_nodes={result['train_nodes']} "
                    f"observed_train_nodes={result['observed_train_nodes']} "
                    f"sampled_recovered={result['sampled_recovered_nodes']} "
                    f"successful_recovered={result['successful_recovered_nodes']} "
                    f"pseudo_train={result['pseudo_train_nodes']} "
                    f"recovery_edges={result['recovery_edges']}"
                )
            print(
                f"  split={split_idx} mean={np.mean(split_accs) * 100:.2f}% "
                f"std={np.std(split_accs) * 100:.2f}%"
            )

        rows.append(
            {
                "dataset": dataset_name,
                "drop_rate": args.drop_rate,
                "model": model_name,
                "accuracy": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "node_drop_rate": args.node_drop_rate,
                "original_edges": base_edges,
                "sparse_edges_mean": float(np.mean(kept_edges)),
                "dropped_nodes_mean": float(np.mean(dropped_nodes)),
                "train_nodes_mean": float(np.mean(train_nodes)),
                "observed_train_nodes_mean": float(np.mean(observed_train_nodes)),
                "cold_start_train_nodes_mean": float(np.mean(cold_start_train_nodes)),
                "sampled_recovered_nodes_mean": float(np.mean(sampled_recovered_nodes)),
                "label_ready_nodes_mean": float(np.mean(label_ready_nodes)),
                "successful_recovered_nodes_mean": float(np.mean(successful_recovered_nodes)),
                "pseudo_train_nodes_mean": float(np.mean(pseudo_train_nodes)),
                "recovery_edges_mean": float(np.mean(recovery_edges)),
                "pseudo_label_accuracy_mean": safe_nanmean(pseudo_label_accuracies),
                "pseudo_label_confidence_mean": safe_nanmean(pseudo_label_confidences),
                "selected_center_distance_mean": safe_nanmean(selected_center_distances),
                "splits": ",".join(str(x) for x in split_indices),
                "num_runs": args.num_runs,
                "k_neighbors": args.k_neighbors,
                "beta": args.beta,
                "recovery_ratio": np.nan if args.cold_start else args.recovery_ratio,
                "cold_start": bool(args.cold_start),
                "admission_ratio": args.admission_ratio if args.cold_start else np.nan,
                "admission_strategy": args.admission_strategy if args.cold_start else "",
                "pseudo_label_strategy": args.pseudo_label_strategy if args.cold_start else "",
                "pseudo_label_k": args.pseudo_label_k if args.cold_start else np.nan,
                "pseudo_label_confidence": args.pseudo_label_confidence if args.cold_start else np.nan,
                "repair_policy": args.repair_policy,
                "similarity_threshold": args.similarity_threshold,
                "adaptive_threshold_alpha": args.adaptive_threshold_alpha,
                "max_edges_per_recovered_node": args.max_edges_per_recovered_node,
            }
        )
        print(f"  mean={np.mean(accs) * 100:.2f}% std={np.std(accs) * 100:.2f}%")

    result_df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        PROJECT_ROOT
        / "logs"
        / (
            f"{args.dataset}_single_pilot{'_coldstart' if args.cold_start else ''}"
            f"_drop{int(args.drop_rate * 100)}"
            f"_node{int(args.node_drop_rate * 100)}_{timestamp}.csv"
        )
    )
    result_df.to_csv(out_path, index=False)

    print("\nSummary")
    print(
        result_df[
            [
                "model",
                "accuracy",
                "std",
                "sparse_edges_mean",
                "dropped_nodes_mean",
                "observed_train_nodes_mean",
                "train_nodes_mean",
                "sampled_recovered_nodes_mean",
                "pseudo_train_nodes_mean",
                "successful_recovered_nodes_mean",
                "recovery_edges_mean",
                "pseudo_label_accuracy_mean",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
