"""
Diagnose how many edges targeted repair adds under node-drop corruption.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch

from scripts.preprocess.preprocess_citation import preprocess_citation
from scripts.utils.run_single_dataset_pilot import build_corrupted_graph
from src import config
from src.models import StructureLearner


def diagnose(dataset, node_drop_rates, k_neighbors):
    data, _ = preprocess_citation(dataset)
    data.repair_target_mask = data.text_available_mask
    data.semantic_candidate_mask = data.text_available_mask

    for node_drop_rate in node_drop_rates:
        config.set_seed(0)
        sparse_edge_index, dropped_node_mask = build_corrupted_graph(data, 0.0, node_drop_rate)
        candidate_node_mask = (~dropped_node_mask) & data.semantic_candidate_mask

        learner = StructureLearner(k_neighbors=k_neighbors, beta=config.DEFAULT_BETA)
        repaired_edge_index = learner(
            data.x_llm,
            sparse_edge_index,
            data.num_nodes,
            "cpu",
            original_edge_index=data.edge_index,
            repair_node_mask=dropped_node_mask,
            candidate_node_mask=candidate_node_mask,
        )

        sparse_undirected = learner._coalesce_undirected_edges(sparse_edge_index, "cpu")
        repaired_undirected = learner._coalesce_undirected_edges(repaired_edge_index, "cpu")
        original_undirected = learner._coalesce_undirected_edges(data.edge_index, "cpu")

        degree_sparse = learner._compute_undirected_degree(sparse_undirected, data.num_nodes, "cpu")
        degree_original = learner._compute_undirected_degree(original_undirected, data.num_nodes, "cpu")
        missing_degree = torch.clamp(degree_original - degree_sparse, min=0) * dropped_node_mask.long()

        print(
            f"dataset={dataset} node_drop={node_drop_rate:.2f} "
            f"dropped={int(dropped_node_mask.sum())} "
            f"train_nodes={int(data.train_mask.sum())} "
            f"isolated_train_nodes={int((data.train_mask & dropped_node_mask).sum())} "
            f"sparse_undirected={sparse_undirected.size(1)} "
            f"repaired_undirected={repaired_undirected.size(1)} "
            f"added_undirected={repaired_undirected.size(1) - sparse_undirected.size(1)} "
            f"missing_degree_sum={int(missing_degree.sum())}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cora", "pubmed"], required=True)
    parser.add_argument("--node-drop-rates", default="0.75,0.85,0.95")
    parser.add_argument("--k-neighbors", type=int, default=10)
    args = parser.parse_args()

    rates = [float(value) for value in args.node_drop_rates.split(",") if value]
    diagnose(args.dataset, rates, args.k_neighbors)


if __name__ == "__main__":
    main()
