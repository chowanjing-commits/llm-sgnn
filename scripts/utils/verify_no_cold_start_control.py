"""
Verify the no-cold-start control used in cold-start experiments.

The control corresponds to admission_ratio=0.0: cold-start candidates are
structurally hidden, no candidate is admitted, no semantic recovery edge is
added, and training uses only the observed training nodes.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.cold_start import ColdStartPipelineConfig, build_cold_start_training_state_from_config


def build_toy_inputs():
    x_llm = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
            [0.2, 0.8, 0.0],
        ],
        dtype=torch.float,
    )
    y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    original_edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 3, 4, 4, 5],
            [1, 0, 2, 1, 4, 3, 5, 4],
        ],
        dtype=torch.long,
    )
    observed_train_mask = torch.tensor([True, True, False, True, True, False])
    cold_start_mask = torch.tensor([False, False, True, False, False, True])
    keep_edge_mask = ~cold_start_mask[original_edge_index[0]] & ~cold_start_mask[original_edge_index[1]]
    sparse_edge_index = original_edge_index[:, keep_edge_mask]
    candidate_node_mask = ~cold_start_mask
    return x_llm, y, original_edge_index, sparse_edge_index, observed_train_mask, cold_start_mask, candidate_node_mask


def main():
    (
        x_llm,
        y,
        original_edge_index,
        sparse_edge_index,
        observed_train_mask,
        cold_start_mask,
        candidate_node_mask,
    ) = build_toy_inputs()
    config = ColdStartPipelineConfig(
        admission_ratio=0.0,
        admission_strategy="cluster_representative",
        pseudo_label_strategy="cluster_majority",
        pseudo_label_k=3,
        pseudo_label_confidence=0.0,
        min_pseudo_label_support=0,
        pseudo_label_agreement="none",
        k_neighbors=3,
        similarity_threshold=0.0,
        max_edges_per_node=2,
        repair_policy="adaptive",
        adaptive_threshold_alpha=0.0,
    )
    state = build_cold_start_training_state_from_config(
        x_llm=x_llm,
        y=y,
        sparse_edge_index=sparse_edge_index,
        cold_start_mask=cold_start_mask,
        observed_train_mask=observed_train_mask,
        candidate_node_mask=candidate_node_mask,
        num_classes=2,
        pipeline_config=config,
        seed=11,
        observed_node_count=int(candidate_node_mask.sum().item()),
    )

    incident_to_hidden = (
        cold_start_mask[state["edge_index"][0]].sum().item()
        + cold_start_mask[state["edge_index"][1]].sum().item()
    )
    if not torch.equal(state["edge_index"], sparse_edge_index):
        raise AssertionError("no-cold-start control must not add or restore edges")
    if incident_to_hidden != 0:
        raise AssertionError("no-cold-start control must not keep edges incident to hidden cold-start nodes")
    if not torch.equal(state["train_mask"], observed_train_mask):
        raise AssertionError("no-cold-start control must train only on observed training nodes")
    if state["selected_mask"].any():
        raise AssertionError("no-cold-start control must not admit cold-start nodes")
    if state["pseudo_train_mask"].any():
        raise AssertionError("no-cold-start control must not add pseudo-training nodes")
    if (state["train_mask"] & cold_start_mask).any():
        raise AssertionError("hidden cold-start labels must not enter training")
    if state["added_recovery_edges"] != 0:
        raise AssertionError("no-cold-start control must not add semantic recovery edges")
    if original_edge_index.size(1) <= sparse_edge_index.size(1):
        raise AssertionError("toy graph should remove incident edges before the no-cold-start check")

    print("No-cold-start control verification passed.")


if __name__ == "__main__":
    main()
