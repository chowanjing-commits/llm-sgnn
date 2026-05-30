"""
Screen cold-start pseudo-label reliability gates without training a GNN.

This diagnostic reuses the cold-start admission and edge-recovery pipeline, then
reports how confidence and support thresholds affect the pseudo-labeled subset.
Ground-truth labels are used only for offline pseudo-label accuracy reporting.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from scripts.preprocess.preprocess_wikics import get_wikics_split
from scripts.utils.run_cold_start_recovery import (
    load_dataset,
    parse_csv_floats,
    parse_csv_ints,
)
from scripts.utils.run_single_dataset_pilot import build_corrupted_graph
from src import config
from src.cold_start import (
    ColdStartPipelineConfig,
    build_cold_start_training_state_from_config,
    safe_nanmean,
)


def parse_csv_strings(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def summarize_mask(state, data, mask):
    nodes = torch.nonzero(mask, as_tuple=False).view(-1)
    if nodes.numel() == 0:
        return {
            "pseudo_train_nodes": 0,
            "pseudo_label_accuracy": float("nan"),
            "pseudo_label_confidence_mean": float("nan"),
            "pseudo_label_support_mean": float("nan"),
        }
    return {
        "pseudo_train_nodes": int(nodes.numel()),
        "pseudo_label_accuracy": (state["pseudo_y"][nodes] == data.y[nodes]).float().mean().item(),
        "pseudo_label_confidence_mean": state["pseudo_confidence"][nodes].float().mean().item(),
        "pseudo_label_support_mean": state["pseudo_label_support"][nodes].float().mean().item(),
    }


def run_screen_for_seed(
    data,
    seed,
    node_drop_rate,
    drop_rate,
    admission_ratio,
    admission_strategy,
    pseudo_label_strategy,
    pseudo_label_k,
    pseudo_label_agreement,
    k_neighbors,
    similarity_threshold,
    max_edges_per_recovered_node,
    repair_policy,
    adaptive_threshold_alpha,
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

    pipeline_config = ColdStartPipelineConfig(
        admission_ratio=admission_ratio,
        admission_strategy=admission_strategy,
        pseudo_label_strategy=pseudo_label_strategy,
        pseudo_label_k=pseudo_label_k,
        pseudo_label_confidence=0.0,
        min_pseudo_label_support=0,
        pseudo_label_agreement=pseudo_label_agreement,
        k_neighbors=k_neighbors,
        similarity_threshold=similarity_threshold,
        max_edges_per_node=max_edges_per_recovered_node,
        repair_policy=repair_policy,
        adaptive_threshold_alpha=adaptive_threshold_alpha,
    )
    state = build_cold_start_training_state_from_config(
        x_llm=data.x_llm,
        y=data.y,
        sparse_edge_index=sparse_edge_index,
        cold_start_mask=cold_start_mask,
        observed_train_mask=observed_train_mask,
        candidate_node_mask=candidate_node_mask,
        num_classes=int(data.y.max().item() + 1),
        pipeline_config=pipeline_config,
        seed=seed,
        observed_node_count=int((~dropped_node_mask).sum().item()),
    )
    return state, {
        "sparse_edges": int(sparse_edge_index.size(1)),
        "dropped_nodes": int(dropped_node_mask.sum().item()),
        "cold_start_train_nodes": int(cold_start_mask.sum().item()),
        "observed_train_nodes": int(observed_train_mask.sum().item()),
        "selected_cold_start_nodes": int(state["selected_mask"].sum().item()),
        "successful_recovered_nodes": int(state["successful_recovered_mask"].sum().item()),
        "recovery_edges": int(state["added_recovery_edges"]),
    }


def aggregate(rows, group_columns):
    grouped_rows = []
    for key, group in pd.DataFrame(rows).groupby(group_columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_columns, key))
        for column in [
            "sparse_edges",
            "dropped_nodes",
            "cold_start_train_nodes",
            "observed_train_nodes",
            "selected_cold_start_nodes",
            "successful_recovered_nodes",
            "recovery_edges",
            "pseudo_train_nodes",
        ]:
            row[f"{column}_mean"] = float(group[column].mean())
        row["pseudo_label_accuracy_mean"] = safe_nanmean(group["pseudo_label_accuracy"].tolist())
        row["pseudo_label_confidence_mean"] = safe_nanmean(group["pseudo_label_confidence_mean"].tolist())
        row["pseudo_label_support_mean"] = safe_nanmean(group["pseudo_label_support_mean"].tolist())
        grouped_rows.append(row)
    return pd.DataFrame(grouped_rows)


def main():
    parser = argparse.ArgumentParser(description="Screen cold-start pseudo-label reliability gates")
    parser.add_argument("--datasets", type=str, default="cora,pubmed,wikics")
    parser.add_argument("--arxiv-subgraph-size", type=int, default=10000)
    parser.add_argument("--node-drop-rate", type=float, default=0.75)
    parser.add_argument("--drop-rate", type=float, default=0.0)
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
    parser.add_argument(
        "--pseudo-label-agreements",
        type=str,
        default="nearest_and_centroid",
        help="Comma-separated agreement gates to screen.",
    )
    parser.add_argument("--confidence-thresholds", type=str, default="0.0,0.6,0.65,0.7,0.75,0.8")
    parser.add_argument("--min-supports", type=str, default="0,2,5")
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--similarity-threshold", type=float, default=0.6)
    parser.add_argument("--max-edges-per-recovered-node", type=int, default=5)
    parser.add_argument("--repair-policy", type=str, default="adaptive", choices=["fixed", "adaptive"])
    parser.add_argument("--adaptive-threshold-alpha", type=float, default=0.0)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--split-indices", type=str, default="0")
    parser.add_argument("--force-regenerate-embeddings", action="store_true")
    parser.add_argument("--output-prefix", type=str, default="cold_start_reliability_screen")
    args = parser.parse_args()

    datasets = parse_csv_strings(args.datasets)
    agreements = parse_csv_strings(args.pseudo_label_agreements)
    confidence_thresholds = parse_csv_floats(args.confidence_thresholds)
    min_supports = parse_csv_ints(args.min_supports)
    split_indices_arg = parse_csv_ints(args.split_indices)

    allowed_agreements = {"none", "nearest_labeled", "class_centroid", "nearest_or_centroid", "nearest_and_centroid"}
    unknown_agreements = sorted(set(agreements) - allowed_agreements)
    if unknown_agreements:
        raise ValueError(f"Unknown pseudo-label agreement modes: {unknown_agreements}")

    raw_rows = []
    config.set_seed()
    for dataset in datasets:
        data, dataset_name, split_mode = load_dataset(
            dataset,
            arxiv_subgraph_size=args.arxiv_subgraph_size,
            force_regenerate_embeddings=args.force_regenerate_embeddings,
        )
        split_indices = split_indices_arg if split_mode == "wikics" else [0]
        print(f"\n{dataset_name}: screening {len(agreements)} agreement gate(s)")
        for split_idx in split_indices:
            split_data = get_wikics_split(data, split_idx) if split_mode == "wikics" else data
            for agreement in agreements:
                for seed in range(args.num_runs):
                    run_seed = seed + split_idx * 100
                    config.set_seed(run_seed)
                    state, base_stats = run_screen_for_seed(
                        data=split_data,
                        seed=run_seed,
                        node_drop_rate=args.node_drop_rate,
                        drop_rate=args.drop_rate,
                        admission_ratio=args.admission_ratio,
                        admission_strategy=args.admission_strategy,
                        pseudo_label_strategy=args.pseudo_label_strategy,
                        pseudo_label_k=args.pseudo_label_k,
                        pseudo_label_agreement=agreement,
                        k_neighbors=args.k_neighbors,
                        similarity_threshold=args.similarity_threshold,
                        max_edges_per_recovered_node=args.max_edges_per_recovered_node,
                        repair_policy=args.repair_policy,
                        adaptive_threshold_alpha=args.adaptive_threshold_alpha,
                    )
                    for confidence in confidence_thresholds:
                        for min_support in min_supports:
                            pseudo_mask = (
                                state["selected_mask"]
                                & state["successful_recovered_mask"]
                                & state["pseudo_label_agreement_mask"]
                                & torch.isfinite(state["pseudo_confidence"])
                                & (state["pseudo_confidence"] >= confidence)
                                & (state["pseudo_label_support"] >= min_support)
                            )
                            row = {
                                "dataset": dataset_name,
                                "split": split_idx,
                                "seed": seed,
                                "pseudo_label_agreement": agreement,
                                "pseudo_label_confidence": confidence,
                                "min_pseudo_label_support": min_support,
                                "admission_strategy": args.admission_strategy,
                                "admission_ratio": args.admission_ratio,
                                "pseudo_label_strategy": args.pseudo_label_strategy,
                                "pseudo_label_k": args.pseudo_label_k,
                                "k_neighbors": args.k_neighbors,
                                "max_edges_per_recovered_node": args.max_edges_per_recovered_node,
                                "repair_policy": args.repair_policy,
                                **base_stats,
                                **summarize_mask(state, split_data, pseudo_mask),
                            }
                            raw_rows.append(row)
                print(f"  split={split_idx} agreement={agreement} done")

    group_columns = [
        "dataset",
        "split",
        "pseudo_label_agreement",
        "pseudo_label_confidence",
        "min_pseudo_label_support",
        "admission_strategy",
        "admission_ratio",
        "pseudo_label_strategy",
        "pseudo_label_k",
        "k_neighbors",
        "max_edges_per_recovered_node",
        "repair_policy",
    ]
    summary = aggregate(raw_rows, group_columns)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = PROJECT_ROOT / "logs" / f"{args.output_prefix}_summary_{timestamp}.csv"
    raw_path = PROJECT_ROOT / "logs" / f"{args.output_prefix}_raw_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)

    display_columns = [
        "dataset",
        "pseudo_label_agreement",
        "pseudo_label_confidence",
        "min_pseudo_label_support",
        "selected_cold_start_nodes_mean",
        "pseudo_train_nodes_mean",
        "pseudo_label_accuracy_mean",
        "pseudo_label_confidence_mean",
        "pseudo_label_support_mean",
    ]
    print("\nSummary")
    print(summary[display_columns].to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved raw rows to: {raw_path}")


if __name__ == "__main__":
    main()
