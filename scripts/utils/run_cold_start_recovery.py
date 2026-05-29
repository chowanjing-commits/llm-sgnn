"""
Run cold-start node recovery with text-only new nodes.

The protocol simulates nodes that enter the graph with only raw-text embeddings:
their incident edges and training labels are hidden, candidate nodes are selected
by unsupervised clustering, pseudo labels are inferred from observed labeled
nodes, and selected nodes are connected back to the sparse graph by semantic kNN.
"""
import argparse
import gc
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
from scripts.preprocess.preprocess_wikics import get_wikics_split, preprocess_wikics
from scripts.utils.run_single_dataset_pilot import (
    build_corrupted_graph,
    build_model,
    build_threshold_recovery_edges,
    is_repair_model,
)
from src import config

torch.serialization.add_safe_globals(
    [DataEdgeAttr, DataTensorAttr, Data, GlobalStorage, NodeStorage, EdgeStorage]
)


def load_dataset(dataset, arxiv_subgraph_size, force_regenerate_embeddings=False):
    if dataset == "wikics":
        data, _ = preprocess_wikics(force_regenerate=force_regenerate_embeddings)
        return data, "WikiCS", "wikics"
    if dataset == "arxiv":
        data, _ = preprocess_arxiv(force_regenerate=force_regenerate_embeddings)
        if arxiv_subgraph_size > 0:
            data = sample_arxiv_subgraph(data, num_nodes=arxiv_subgraph_size, seed=42)
            return data, f"ogbn-arxiv-subgraph-{arxiv_subgraph_size}", "single"
        return data, "ogbn-arxiv-full", "single"
    if dataset in {"cora", "pubmed"}:
        data, _ = preprocess_citation(dataset, force_regenerate=force_regenerate_embeddings)
        if hasattr(data, "text_available_mask"):
            data.repair_target_mask = data.text_available_mask
            data.semantic_candidate_mask = data.text_available_mask
        return data, "Cora" if dataset == "cora" else "PubMed", "single"
    raise ValueError(f"Unsupported dataset: {dataset}")


def parse_csv_ints(value):
    return [int(item) for item in value.split(",") if item.strip()]


def parse_csv_floats(value):
    return [float(item) for item in value.split(",") if item.strip()]


def safe_nanmean(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def normalized_cpu(x):
    return F.normalize(x.detach().cpu().float(), p=2, dim=1)


def fit_kmeans(x, num_clusters, seed):
    from sklearn.cluster import MiniBatchKMeans

    if x.size(0) == 0:
        return torch.empty(0, dtype=torch.long), torch.empty((0, x.size(1)))
    k = max(1, min(int(num_clusters), x.size(0)))
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=min(4096, max(256, x.size(0))),
        n_init="auto",
        max_iter=100,
    )
    labels = torch.tensor(kmeans.fit_predict(x.numpy()), dtype=torch.long)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
    return labels, F.normalize(centers, p=2, dim=1)


def admission_budget(num_candidates, admission_ratio):
    if num_candidates == 0 or admission_ratio <= 0:
        return 0
    total = int(num_candidates * admission_ratio)
    if admission_ratio > 0 and total == 0:
        total = 1
    return min(total, num_candidates)


def select_random_admission(cold_start_mask, admission_ratio, seed):
    """Random admission baseline under the same cold-start node budget."""
    selected = torch.zeros_like(cold_start_mask)
    candidate_nodes = torch.nonzero(cold_start_mask, as_tuple=False).view(-1).cpu()
    total = admission_budget(candidate_nodes.numel(), admission_ratio)
    center_distance = torch.full((cold_start_mask.size(0),), float("nan"), dtype=torch.float)
    cluster_labels = torch.full((cold_start_mask.size(0),), -1, dtype=torch.long)
    if total == 0:
        return selected, cluster_labels, center_distance

    generator = torch.Generator().manual_seed(seed)
    keep_positions = torch.randperm(candidate_nodes.numel(), generator=generator)[:total]
    selected[candidate_nodes[keep_positions]] = True
    return selected, cluster_labels, center_distance


def select_cluster_representatives(x_llm, cold_start_mask, num_clusters, admission_ratio, seed):
    """Cluster cold-start candidates and keep nodes closest to their cluster centers."""
    selected = torch.zeros_like(cold_start_mask)
    candidate_nodes = torch.nonzero(cold_start_mask, as_tuple=False).view(-1).cpu()
    total = admission_budget(candidate_nodes.numel(), admission_ratio)
    if total == 0:
        cluster_labels = torch.full((x_llm.size(0),), -1, dtype=torch.long)
        center_distance = torch.full((x_llm.size(0),), float("nan"), dtype=torch.float)
        return selected, cluster_labels, center_distance

    x = normalized_cpu(x_llm)
    candidate_x = x[candidate_nodes]
    cluster_labels, centers = fit_kmeans(candidate_x, num_clusters=num_clusters, seed=seed)
    sims = (candidate_x * centers[cluster_labels]).sum(dim=1)
    distances = 1.0 - sims

    selected_positions = set()
    cluster_ids = torch.unique(cluster_labels).tolist()
    for cluster_id in cluster_ids:
        positions = torch.nonzero(cluster_labels == int(cluster_id), as_tuple=False).view(-1)
        quota = admission_budget(positions.numel(), admission_ratio)
        order = positions[torch.argsort(distances[positions])]
        selected_positions.update(order[:quota].tolist())

    if len(selected_positions) < total:
        remaining = [
            pos
            for pos in torch.argsort(distances).tolist()
            if pos not in selected_positions
        ]
        selected_positions.update(remaining[: total - len(selected_positions)])
    selected_positions = sorted(selected_positions, key=lambda pos: float(distances[pos].item()))
    keep_positions = torch.tensor(selected_positions[:total], dtype=torch.long)
    selected[candidate_nodes[keep_positions]] = True

    all_cluster_labels = torch.full((x_llm.size(0),), -1, dtype=torch.long)
    all_cluster_labels[candidate_nodes] = cluster_labels
    center_distance = torch.full((x_llm.size(0),), float("nan"), dtype=torch.float)
    center_distance[candidate_nodes] = distances
    return selected, all_cluster_labels, center_distance


def select_cold_start_nodes(x_llm, cold_start_mask, num_classes, admission_ratio, strategy, seed):
    if strategy == "cluster_representative":
        return select_cluster_representatives(
            x_llm=x_llm,
            cold_start_mask=cold_start_mask,
            num_clusters=num_classes,
            admission_ratio=admission_ratio,
            seed=seed,
        )
    if strategy == "random":
        return select_random_admission(
            cold_start_mask=cold_start_mask,
            admission_ratio=admission_ratio,
            seed=seed,
        )
    raise ValueError(f"Unknown admission strategy: {strategy}")


def pseudo_label_by_nearest_labeled(x_llm, observed_train_mask, selected_mask, y, num_classes, k):
    pseudo_y = y.clone()
    confidence = torch.zeros_like(y, dtype=torch.float)
    train_nodes = torch.nonzero(observed_train_mask, as_tuple=False).view(-1).cpu()
    selected_nodes = torch.nonzero(selected_mask, as_tuple=False).view(-1).cpu()
    if train_nodes.numel() == 0 or selected_nodes.numel() == 0:
        return pseudo_y, confidence

    x = normalized_cpu(x_llm)
    topk = min(max(1, int(k)), train_nodes.numel())
    sims = x[selected_nodes] @ x[train_nodes].t().contiguous()
    _, nn_idx = torch.topk(sims, k=topk, dim=1, largest=True)
    train_y = y.detach().cpu()[train_nodes]
    for row_idx, node in enumerate(selected_nodes.tolist()):
        labels = train_y[nn_idx[row_idx]]
        counts = torch.bincount(labels, minlength=num_classes).float()
        pseudo_y[node] = int(torch.argmax(counts).item())
        confidence[node] = float(counts.max().item() / max(labels.numel(), 1))
    return pseudo_y, confidence


def pseudo_label_by_class_centroid(x_llm, observed_train_mask, selected_mask, y, num_classes):
    pseudo_y = y.clone()
    confidence = torch.zeros_like(y, dtype=torch.float)
    selected_nodes = torch.nonzero(selected_mask, as_tuple=False).view(-1).cpu()
    if selected_nodes.numel() == 0:
        return pseudo_y, confidence

    x = normalized_cpu(x_llm)
    centroids = []
    valid_labels = []
    y_cpu = y.detach().cpu()
    observed_cpu = observed_train_mask.detach().cpu()
    for label in range(num_classes):
        nodes = torch.nonzero(observed_cpu & (y_cpu == label), as_tuple=False).view(-1)
        if nodes.numel() == 0:
            continue
        centroids.append(F.normalize(x[nodes].mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0))
        valid_labels.append(label)
    if not centroids:
        return pseudo_y, confidence

    centroid_x = torch.stack(centroids, dim=0)
    sims = x[selected_nodes] @ centroid_x.t().contiguous()
    best_sims, best_idx = torch.max(sims, dim=1)
    for row_idx, node in enumerate(selected_nodes.tolist()):
        pseudo_y[node] = valid_labels[int(best_idx[row_idx].item())]
        confidence[node] = float((best_sims[row_idx].item() + 1.0) / 2.0)
    return pseudo_y, confidence


def pseudo_label_by_cluster_majority(
    x_llm,
    observed_train_mask,
    selected_mask,
    y,
    num_classes,
    num_clusters,
    seed,
):
    """Cluster observed labeled nodes plus selected cold-start nodes, then map clusters to labels."""
    pseudo_y = y.clone()
    confidence = torch.zeros_like(y, dtype=torch.float)
    active_mask = observed_train_mask | selected_mask
    active_nodes = torch.nonzero(active_mask, as_tuple=False).view(-1).cpu()
    selected_nodes = torch.nonzero(selected_mask, as_tuple=False).view(-1).cpu()
    if active_nodes.numel() == 0 or selected_nodes.numel() == 0:
        return pseudo_y, confidence

    x = normalized_cpu(x_llm)
    cluster_labels, _ = fit_kmeans(x[active_nodes], num_clusters=num_clusters, seed=seed)
    node_to_cluster = torch.full((x_llm.size(0),), -1, dtype=torch.long)
    node_to_cluster[active_nodes] = cluster_labels

    y_cpu = y.detach().cpu()
    observed_cpu = observed_train_mask.detach().cpu()
    fallback_y, fallback_conf = pseudo_label_by_class_centroid(
        x_llm=x_llm,
        observed_train_mask=observed_train_mask,
        selected_mask=selected_mask,
        y=y,
        num_classes=num_classes,
    )
    for node in selected_nodes.tolist():
        cluster_id = int(node_to_cluster[node].item())
        cluster_nodes = active_nodes[cluster_labels == cluster_id]
        labeled_nodes = cluster_nodes[observed_cpu[cluster_nodes]]
        if labeled_nodes.numel() == 0:
            pseudo_y[node] = fallback_y[node]
            confidence[node] = fallback_conf[node]
            continue
        counts = torch.bincount(y_cpu[labeled_nodes], minlength=num_classes).float()
        pseudo_y[node] = int(torch.argmax(counts).item())
        confidence[node] = float(counts.max().item() / max(labeled_nodes.numel(), 1))
    return pseudo_y, confidence


def assign_pseudo_labels(
    x_llm,
    observed_train_mask,
    selected_mask,
    y,
    num_classes,
    strategy,
    pseudo_label_k,
    seed,
):
    if strategy == "cluster_majority":
        return pseudo_label_by_cluster_majority(
            x_llm=x_llm,
            observed_train_mask=observed_train_mask,
            selected_mask=selected_mask,
            y=y,
            num_classes=num_classes,
            num_clusters=num_classes,
            seed=seed,
        )
    if strategy == "nearest_labeled":
        return pseudo_label_by_nearest_labeled(
            x_llm=x_llm,
            observed_train_mask=observed_train_mask,
            selected_mask=selected_mask,
            y=y,
            num_classes=num_classes,
            k=pseudo_label_k,
        )
    if strategy == "class_centroid":
        return pseudo_label_by_class_centroid(
            x_llm=x_llm,
            observed_train_mask=observed_train_mask,
            selected_mask=selected_mask,
            y=y,
            num_classes=num_classes,
        )
    raise ValueError(f"Unknown pseudo-label strategy: {strategy}")


def train_gnn(data, edge_index, train_mask, train_y, num_epochs, model_type, k_neighbors, beta):
    device = config.DEVICE
    x = data.x_llm.to(device)
    y_train = train_y.to(device)
    y_eval = data.y.to(device)
    edge_index = edge_index.to(device)
    if is_repair_model(model_type):
        edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)

    model = build_model(
        model_type=model_type,
        in_dim=data.x_llm.size(1),
        num_classes=int(data.y.max().item() + 1),
        k_neighbors=k_neighbors,
        beta=beta,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    train_mask = train_mask.to(device)
    if not train_mask.any():
        raise RuntimeError(
            "No training nodes remain after cold-start admission. "
            "Lower --node-drop-rate, raise --admission-ratio, or lower --pseudo-label-confidence."
        )
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    best_val = 0.0
    best_test = 0.0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model.gnn(x, edge_index) if is_repair_model(model_type) else model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y_train[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            model.eval()
            with torch.no_grad():
                out = model.gnn(x, edge_index) if is_repair_model(model_type) else model(x, edge_index)
                pred = out.argmax(dim=1)
                val_acc = (pred[val_mask] == y_eval[val_mask]).float().mean().item() if val_mask.any() else 0.0
                test_acc = (pred[test_mask] == y_eval[test_mask]).float().mean().item() if test_mask.any() else 0.0
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = test_acc

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return best_test


def run_once(
    data,
    seed,
    node_drop_rate,
    drop_rate,
    admission_ratio,
    admission_strategy,
    pseudo_label_strategy,
    pseudo_label_k,
    pseudo_label_confidence,
    k_neighbors,
    similarity_threshold,
    max_edges_per_recovered_node,
    repair_policy,
    adaptive_threshold_alpha,
    model_type,
    beta,
    num_epochs,
):
    sparse_edge_index, dropped_node_mask = build_corrupted_graph(
        data,
        edge_drop_rate=drop_rate,
        node_drop_rate=node_drop_rate,
    )
    candidate_node_mask = ~dropped_node_mask
    if hasattr(data, "semantic_candidate_mask"):
        candidate_node_mask = candidate_node_mask & data.semantic_candidate_mask

    observed_train_mask = data.train_mask & ~dropped_node_mask
    cold_start_mask = data.train_mask & dropped_node_mask
    if hasattr(data, "repair_target_mask"):
        cold_start_mask = cold_start_mask & data.repair_target_mask

    num_classes = int(data.y.max().item() + 1)
    selected_mask, cluster_labels, center_distance = select_cold_start_nodes(
        x_llm=data.x_llm,
        cold_start_mask=cold_start_mask,
        num_classes=num_classes,
        admission_ratio=admission_ratio,
        strategy=admission_strategy,
        seed=seed,
    )
    pseudo_y, pseudo_confidence = assign_pseudo_labels(
        x_llm=data.x_llm,
        observed_train_mask=observed_train_mask,
        selected_mask=selected_mask,
        y=data.y,
        num_classes=num_classes,
        strategy=pseudo_label_strategy,
        pseudo_label_k=pseudo_label_k,
        seed=seed,
    )
    label_ready_mask = selected_mask & (pseudo_confidence >= pseudo_label_confidence)

    repaired_edge_index, successful_recovered_mask, added_recovery_edges = build_threshold_recovery_edges(
        x_llm=data.x_llm,
        sparse_edge_index=sparse_edge_index,
        recovered_node_mask=selected_mask,
        candidate_node_mask=candidate_node_mask,
        k_neighbors=k_neighbors,
        similarity_threshold=similarity_threshold,
        max_edges_per_node=max_edges_per_recovered_node,
        repair_policy=repair_policy,
        adaptive_threshold_alpha=adaptive_threshold_alpha,
        observed_node_count=int((~dropped_node_mask).sum().item()),
    )
    pseudo_train_mask = label_ready_mask & successful_recovered_mask
    train_mask = observed_train_mask | pseudo_train_mask
    accuracy = train_gnn(
        data=data,
        edge_index=repaired_edge_index,
        train_mask=train_mask,
        train_y=pseudo_y,
        num_epochs=num_epochs,
        model_type=model_type,
        k_neighbors=k_neighbors,
        beta=beta,
    )

    pseudo_nodes = torch.nonzero(pseudo_train_mask, as_tuple=False).view(-1)
    if pseudo_nodes.numel() > 0:
        pseudo_acc = (pseudo_y[pseudo_nodes] == data.y[pseudo_nodes]).float().mean().item()
        pseudo_conf_mean = pseudo_confidence[pseudo_nodes].mean().item()
        center_dist_mean = center_distance[pseudo_nodes].nanmean().item()
    else:
        pseudo_acc = float("nan")
        pseudo_conf_mean = float("nan")
        center_dist_mean = float("nan")

    del cluster_labels
    return {
        "accuracy": accuracy,
        "admission_strategy": admission_strategy,
        "sparse_edges": int(sparse_edge_index.size(1)),
        "repaired_edges": int(repaired_edge_index.size(1)),
        "dropped_nodes": int(dropped_node_mask.sum().item()),
        "cold_start_train_nodes": int(cold_start_mask.sum().item()),
        "observed_train_nodes": int(observed_train_mask.sum().item()),
        "selected_cold_start_nodes": int(selected_mask.sum().item()),
        "label_ready_nodes": int(label_ready_mask.sum().item()),
        "successful_recovered_nodes": int(successful_recovered_mask.sum().item()),
        "pseudo_train_nodes": int(pseudo_train_mask.sum().item()),
        "train_nodes": int(train_mask.sum().item()),
        "recovery_edges": int(added_recovery_edges),
        "pseudo_label_accuracy": pseudo_acc,
        "pseudo_label_confidence_mean": pseudo_conf_mean,
        "selected_center_distance_mean": center_dist_mean,
    }


def main():
    parser = argparse.ArgumentParser(description="Cold-start text-only node recovery experiment")
    parser.add_argument("--dataset", type=str, default="cora", choices=["cora", "pubmed", "wikics", "arxiv"])
    parser.add_argument("--node-drop-rate", type=float, default=0.75)
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--admission-ratio", type=float, default=None)
    parser.add_argument("--admission-ratios", type=str, default=None,
                        help="Comma-separated admission budgets. Overrides --admission-ratio.")
    parser.add_argument("--recovery-ratio", type=float, default=None,
                        help="Deprecated alias for --admission-ratio.")
    parser.add_argument("--admission-strategies", type=str, default="cluster_representative,random",
                        help="Comma-separated cold-start admission strategies: cluster_representative,random.")
    parser.add_argument("--pseudo-label-strategy", type=str, default="cluster_majority",
                        choices=["cluster_majority", "nearest_labeled", "class_centroid"])
    parser.add_argument("--pseudo-label-k", type=int, default=5)
    parser.add_argument("--pseudo-label-confidence", type=float, default=0.0)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--similarity-threshold", type=float, default=0.6)
    parser.add_argument("--max-edges-per-recovered-node", type=int, default=10)
    parser.add_argument("--repair-policy", type=str, default="adaptive", choices=["fixed", "adaptive"])
    parser.add_argument("--adaptive-threshold-alpha", type=float, default=0.0)
    parser.add_argument("--model-type", type=str, default="LLM_GNN_SAGE",
                        choices=["LLM_GNN_GCN", "LLM_GNN_GAT", "LLM_GNN_SAGE", "GCN", "GAT", "SAGE"])
    parser.add_argument("--beta", type=float, default=config.DEFAULT_BETA)
    parser.add_argument("--num-epochs", type=int, default=300)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--split-indices", type=str, default="0")
    parser.add_argument("--arxiv-subgraph-size", type=int, default=10000)
    parser.add_argument(
        "--force-regenerate-embeddings",
        action="store_true",
        help="Regenerate text-derived embeddings from raw text before running cold-start admission.",
    )
    parser.add_argument("--output-prefix", type=str, default="cold_start_recovery")
    args = parser.parse_args()
    if args.admission_ratios is not None:
        admission_ratios = parse_csv_floats(args.admission_ratios)
    else:
        admission_ratio = args.admission_ratio
        if admission_ratio is None:
            admission_ratio = args.recovery_ratio if args.recovery_ratio is not None else 0.5
        admission_ratios = [admission_ratio]
    if not admission_ratios:
        raise ValueError("At least one admission ratio is required.")
    admission_strategies = [item.strip() for item in args.admission_strategies.split(",") if item.strip()]
    unknown_strategies = sorted(set(admission_strategies) - {"cluster_representative", "random"})
    if unknown_strategies:
        raise ValueError(f"Unknown admission strategies: {unknown_strategies}")

    config.set_seed()
    data, dataset_name, split_mode = load_dataset(
        args.dataset,
        args.arxiv_subgraph_size,
        force_regenerate_embeddings=args.force_regenerate_embeddings,
    )
    split_indices = parse_csv_ints(args.split_indices) if split_mode == "wikics" else [0]

    print("=" * 80)
    print(f"Cold-start recovery on {dataset_name}")
    print(f"Nodes={data.num_nodes} edges={data.edge_index.size(1)} classes={int(data.y.max().item() + 1)}")
    print(f"Text feature generation={'regenerate' if args.force_regenerate_embeddings else 'cached_or_existing'}")
    print(f"Admission ratios={','.join(str(value) for value in admission_ratios)}")
    print(f"Admission strategies={','.join(admission_strategies)}, pseudo labels={args.pseudo_label_strategy}")
    print("=" * 80)

    rows = []
    raw_rows = []
    for split_idx in split_indices:
        split_data = get_wikics_split(data, split_idx) if split_mode == "wikics" else data
        for admission_ratio in admission_ratios:
            for admission_strategy in admission_strategies:
                accs = []
                stats = []
                for seed in range(args.num_runs):
                    config.set_seed(seed + split_idx * 100)
                    row = run_once(
                        data=split_data,
                        seed=seed + split_idx * 100,
                        node_drop_rate=args.node_drop_rate,
                        drop_rate=args.drop_rate,
                        admission_ratio=admission_ratio,
                        admission_strategy=admission_strategy,
                        pseudo_label_strategy=args.pseudo_label_strategy,
                        pseudo_label_k=args.pseudo_label_k,
                        pseudo_label_confidence=args.pseudo_label_confidence,
                        k_neighbors=args.k_neighbors,
                        similarity_threshold=args.similarity_threshold,
                        max_edges_per_recovered_node=args.max_edges_per_recovered_node,
                        repair_policy=args.repair_policy,
                        adaptive_threshold_alpha=args.adaptive_threshold_alpha,
                        model_type=args.model_type,
                        beta=args.beta,
                        num_epochs=args.num_epochs,
                    )
                    row = {"dataset": dataset_name, "split": split_idx, "seed": seed, **row}
                    raw_rows.append(row)
                    stats.append(row)
                    accs.append(row["accuracy"])
                    print(
                        f"split={split_idx} seed={seed} ratio={admission_ratio:.2f} "
                        f"admission={admission_strategy} acc={row['accuracy'] * 100:.2f}% "
                        f"selected={row['selected_cold_start_nodes']} pseudo_train={row['pseudo_train_nodes']} "
                        f"pseudo_acc={row['pseudo_label_accuracy']}"
                    )

                rows.append(
                    {
                        "dataset": dataset_name,
                        "split": split_idx,
                        "admission_strategy": admission_strategy,
                        "accuracy": float(np.mean(accs)),
                        "std": float(np.std(accs)),
                        "num_runs": args.num_runs,
                        "node_drop_rate": args.node_drop_rate,
                        "drop_rate": args.drop_rate,
                        "admission_ratio": admission_ratio,
                        "pseudo_label_strategy": args.pseudo_label_strategy,
                        "pseudo_label_k": args.pseudo_label_k,
                        "pseudo_label_confidence": args.pseudo_label_confidence,
                        "k_neighbors": args.k_neighbors,
                        "similarity_threshold": args.similarity_threshold,
                        "max_edges_per_recovered_node": args.max_edges_per_recovered_node,
                        "repair_policy": args.repair_policy,
                        "adaptive_threshold_alpha": args.adaptive_threshold_alpha,
                        "model_type": args.model_type,
                        "beta": args.beta,
                        "force_regenerate_embeddings": bool(args.force_regenerate_embeddings),
                        "sparse_edges_mean": float(np.mean([item["sparse_edges"] for item in stats])),
                        "repaired_edges_mean": float(np.mean([item["repaired_edges"] for item in stats])),
                        "cold_start_train_nodes_mean": float(np.mean([item["cold_start_train_nodes"] for item in stats])),
                        "observed_train_nodes_mean": float(np.mean([item["observed_train_nodes"] for item in stats])),
                        "selected_cold_start_nodes_mean": float(np.mean([item["selected_cold_start_nodes"] for item in stats])),
                        "label_ready_nodes_mean": float(np.mean([item["label_ready_nodes"] for item in stats])),
                        "successful_recovered_nodes_mean": float(np.mean([item["successful_recovered_nodes"] for item in stats])),
                        "pseudo_train_nodes_mean": float(np.mean([item["pseudo_train_nodes"] for item in stats])),
                        "recovery_edges_mean": float(np.mean([item["recovery_edges"] for item in stats])),
                        "pseudo_label_accuracy_mean": safe_nanmean([item["pseudo_label_accuracy"] for item in stats]),
                        "pseudo_label_confidence_mean": safe_nanmean([item["pseudo_label_confidence_mean"] for item in stats]),
                        "selected_center_distance_mean": safe_nanmean([item["selected_center_distance_mean"] for item in stats]),
                    }
                )
                print(
                    f"split={split_idx} ratio={admission_ratio:.2f} admission={admission_strategy} "
                    f"mean={np.mean(accs) * 100:.2f}% std={np.std(accs) * 100:.2f}%"
                )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = PROJECT_ROOT / "logs" / f"{args.output_prefix}_summary_{timestamp}.csv"
    raw_path = PROJECT_ROOT / "logs" / f"{args.output_prefix}_raw_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)

    print("\nSummary")
    print(
        pd.DataFrame(rows)[
            [
                "dataset",
                "split",
                "admission_ratio",
                "admission_strategy",
                "accuracy",
                "std",
                "selected_cold_start_nodes_mean",
                "pseudo_train_nodes_mean",
                "recovery_edges_mean",
                "pseudo_label_accuracy_mean",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved raw rows to: {raw_path}")


if __name__ == "__main__":
    main()
