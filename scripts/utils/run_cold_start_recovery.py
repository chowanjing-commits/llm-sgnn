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
    is_repair_model,
)
from src import config
from src.cold_start import build_cold_start_training_state, safe_nanmean

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
    state = build_cold_start_training_state(
        x_llm=data.x_llm,
        y=data.y,
        sparse_edge_index=sparse_edge_index,
        cold_start_mask=cold_start_mask,
        observed_train_mask=observed_train_mask,
        candidate_node_mask=candidate_node_mask,
        num_classes=num_classes,
        admission_ratio=admission_ratio,
        admission_strategy=admission_strategy,
        pseudo_label_strategy=pseudo_label_strategy,
        pseudo_label_k=pseudo_label_k,
        pseudo_label_confidence=pseudo_label_confidence,
        k_neighbors=k_neighbors,
        similarity_threshold=similarity_threshold,
        max_edges_per_node=max_edges_per_recovered_node,
        repair_policy=repair_policy,
        adaptive_threshold_alpha=adaptive_threshold_alpha,
        seed=seed,
        observed_node_count=int((~dropped_node_mask).sum().item()),
    )
    accuracy = train_gnn(
        data=data,
        edge_index=state["edge_index"],
        train_mask=state["train_mask"],
        train_y=state["pseudo_y"],
        num_epochs=num_epochs,
        model_type=model_type,
        k_neighbors=k_neighbors,
        beta=beta,
    )

    pseudo_train_mask = state["pseudo_train_mask"]
    pseudo_nodes = torch.nonzero(pseudo_train_mask, as_tuple=False).view(-1)
    if pseudo_nodes.numel() > 0:
        pseudo_acc = (state["pseudo_y"][pseudo_nodes] == data.y[pseudo_nodes]).float().mean().item()
        pseudo_conf_mean = state["pseudo_confidence"][pseudo_nodes].mean().item()
        center_dist_mean = state["center_distance"][pseudo_nodes].nanmean().item()
    else:
        pseudo_acc = float("nan")
        pseudo_conf_mean = float("nan")
        center_dist_mean = float("nan")

    return {
        "accuracy": accuracy,
        "admission_strategy": admission_strategy,
        "sparse_edges": int(sparse_edge_index.size(1)),
        "repaired_edges": int(state["edge_index"].size(1)),
        "dropped_nodes": int(dropped_node_mask.sum().item()),
        "cold_start_train_nodes": int(cold_start_mask.sum().item()),
        "observed_train_nodes": int(observed_train_mask.sum().item()),
        "selected_cold_start_nodes": int(state["selected_mask"].sum().item()),
        "label_ready_nodes": int(state["label_ready_mask"].sum().item()),
        "successful_recovered_nodes": int(state["successful_recovered_mask"].sum().item()),
        "pseudo_train_nodes": int(pseudo_train_mask.sum().item()),
        "train_nodes": int(state["train_mask"].sum().item()),
        "recovery_edges": int(state["added_recovery_edges"]),
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
