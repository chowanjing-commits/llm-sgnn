"""
Verify cold-start protocol invariants on a small synthetic graph.

The check is intentionally lightweight: it does not train a GNN and does not
load datasets. It asserts that changing the hidden labels of cold-start nodes
does not affect admission, semantic edge recovery, or the pseudo labels that
enter the supervised loss.
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
            [1.00, 0.00, 0.00, 0.00],
            [0.95, 0.05, 0.00, 0.00],
            [0.90, 0.10, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.05, 0.95, 0.00, 0.00],
            [0.10, 0.90, 0.00, 0.00],
            [0.00, 0.00, 1.00, 0.00],
            [0.00, 0.00, 0.95, 0.05],
            [0.00, 0.00, 0.90, 0.10],
        ],
        dtype=torch.float,
    )
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
    sparse_edge_index = torch.tensor(
        [
            [0, 1, 3, 4, 6, 7],
            [1, 0, 4, 3, 7, 6],
        ],
        dtype=torch.long,
    )
    observed_train_mask = torch.tensor(
        [True, True, False, True, True, False, True, True, False],
        dtype=torch.bool,
    )
    cold_start_mask = torch.tensor(
        [False, False, True, False, False, True, False, False, True],
        dtype=torch.bool,
    )
    candidate_node_mask = ~cold_start_mask
    return x_llm, y, sparse_edge_index, observed_train_mask, cold_start_mask, candidate_node_mask


def run_pipeline(y, observed_train_mask_override=None):
    x_llm, _, sparse_edge_index, observed_train_mask, cold_start_mask, candidate_node_mask = build_toy_inputs()
    if observed_train_mask_override is not None:
        observed_train_mask = observed_train_mask_override
    config = ColdStartPipelineConfig(
        admission_ratio=1.0,
        admission_strategy="cluster_representative",
        pseudo_label_strategy="cluster_majority",
        pseudo_label_k=3,
        pseudo_label_confidence=0.0,
        k_neighbors=3,
        similarity_threshold=0.0,
        max_edges_per_node=2,
        repair_policy="adaptive",
        adaptive_threshold_alpha=0.0,
    )
    return build_cold_start_training_state_from_config(
        x_llm=x_llm,
        y=y,
        sparse_edge_index=sparse_edge_index,
        cold_start_mask=cold_start_mask,
        observed_train_mask=observed_train_mask,
        candidate_node_mask=candidate_node_mask,
        num_classes=3,
        pipeline_config=config,
        seed=7,
        observed_node_count=int(candidate_node_mask.sum().item()),
    )


def assert_equal(name, left, right):
    if torch.is_tensor(left):
        if not torch.equal(left, right):
            raise AssertionError(f"{name} changed unexpectedly")
    elif left != right:
        raise AssertionError(f"{name} changed unexpectedly")


def main():
    _, y, _, observed_train_mask, cold_start_mask, _ = build_toy_inputs()
    changed_hidden_y = y.clone()
    changed_hidden_y[cold_start_mask] = torch.tensor([2, 0, 1], dtype=torch.long)

    baseline = run_pipeline(y)
    changed = run_pipeline(changed_hidden_y)

    assert_equal("selected_mask", baseline["selected_mask"], changed["selected_mask"])
    assert_equal("edge_index", baseline["edge_index"], changed["edge_index"])
    assert_equal("pseudo_train_mask", baseline["pseudo_train_mask"], changed["pseudo_train_mask"])
    assert_equal("added_recovery_edges", baseline["added_recovery_edges"], changed["added_recovery_edges"])

    pseudo_nodes = baseline["pseudo_train_mask"]
    assert pseudo_nodes.any(), "toy protocol should admit pseudo-training nodes"
    assert_equal("pseudo_y_on_training_nodes", baseline["pseudo_y"][pseudo_nodes], changed["pseudo_y"][pseudo_nodes])

    hidden_train_nodes = cold_start_mask & baseline["train_mask"]
    if not torch.equal(hidden_train_nodes, baseline["pseudo_train_mask"]):
        raise AssertionError("cold-start training nodes must enter only through pseudo_train_mask")
    if not torch.equal(observed_train_mask | baseline["pseudo_train_mask"], baseline["train_mask"]):
        raise AssertionError("train_mask must be observed_train_mask union pseudo_train_mask")

    no_observed = run_pipeline(y, torch.zeros_like(observed_train_mask))
    if no_observed["pseudo_train_mask"].any():
        raise AssertionError("cold-start nodes without inferred pseudo labels must not enter training")
    if no_observed["train_mask"].any():
        raise AssertionError("train_mask must stay empty when no observed or pseudo-labeled nodes exist")

    print("Cold-start protocol verification passed.")


if __name__ == "__main__":
    main()
