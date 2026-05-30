"""
Verify selective pseudo-label filtering for cold-start training.

Admitting a text-only node into the recovered graph is separate from treating
its pseudo label as supervised training signal. This check raises the
pseudo-label confidence threshold on a toy graph and asserts that low-confidence
admitted nodes still keep recovery edges but are withheld from pseudo training.
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
            [1.00, 0.00, 0.00],
            [0.92, 0.08, 0.00],
            [0.50, 0.50, 0.00],
            [0.00, 0.00, 1.00],
            [0.02, 0.00, 0.98],
            [0.96, 0.04, 0.00],
            [0.65, 0.35, 0.00],
            [0.01, 0.00, 0.99],
        ],
        dtype=torch.float,
    )
    y = torch.tensor([0, 0, 1, 2, 2, 1, 2, 1], dtype=torch.long)
    sparse_edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 3, 4],
            [1, 0, 2, 1, 4, 3],
        ],
        dtype=torch.long,
    )
    observed_train_mask = torch.tensor(
        [True, True, True, True, True, False, False, False],
        dtype=torch.bool,
    )
    cold_start_mask = torch.tensor(
        [False, False, False, False, False, True, True, True],
        dtype=torch.bool,
    )
    candidate_node_mask = ~cold_start_mask
    return x_llm, y, sparse_edge_index, observed_train_mask, cold_start_mask, candidate_node_mask


def run_pipeline(pseudo_label_confidence, y_override=None):
    x_llm, y, sparse_edge_index, observed_train_mask, cold_start_mask, candidate_node_mask = build_toy_inputs()
    if y_override is not None:
        y = y_override
    config = ColdStartPipelineConfig(
        admission_ratio=1.0,
        admission_strategy="random",
        pseudo_label_strategy="nearest_labeled",
        pseudo_label_k=2,
        pseudo_label_confidence=pseudo_label_confidence,
        min_pseudo_label_support=0,
        pseudo_label_agreement="none",
        k_neighbors=3,
        similarity_threshold=0.0,
        max_edges_per_node=2,
        repair_policy="fixed",
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
        seed=11,
        observed_node_count=int(candidate_node_mask.sum().item()),
    )


def assert_tensor_equal(name, left, right):
    if not torch.equal(left, right):
        raise AssertionError(f"{name} changed unexpectedly")


def main():
    _, y, _, observed_train_mask, cold_start_mask, _ = build_toy_inputs()

    permissive = run_pipeline(pseudo_label_confidence=0.0)
    strict = run_pipeline(pseudo_label_confidence=0.8)

    assert_tensor_equal("selected_mask", permissive["selected_mask"], strict["selected_mask"])
    assert permissive["selected_mask"].sum().item() == 3, "toy graph should admit all cold-start nodes"
    assert permissive["pseudo_train_mask"].sum().item() == 3, "permissive threshold should pseudo-train all admitted nodes"

    withheld_mask = strict["selected_mask"] & strict["successful_recovered_mask"] & ~strict["pseudo_train_mask"]
    if withheld_mask.sum().item() < 1:
        raise AssertionError("strict threshold should withhold at least one admitted recovered node")
    if strict["pseudo_train_mask"].sum().item() >= permissive["pseudo_train_mask"].sum().item():
        raise AssertionError("strict threshold should reduce pseudo-training nodes")
    if not torch.all(strict["pseudo_confidence"][withheld_mask] < 0.8):
        raise AssertionError("withheld nodes should fail the pseudo-label confidence threshold")
    if not torch.equal(strict["train_mask"], observed_train_mask | strict["pseudo_train_mask"]):
        raise AssertionError("train_mask must be observed_train_mask union pseudo_train_mask")
    if strict["pseudo_y"][withheld_mask].any():
        raise AssertionError("withheld cold-start labels must be scrubbed from pseudo_y")

    changed_hidden_y = y.clone()
    changed_hidden_y[cold_start_mask] = torch.tensor([2, 0, 2], dtype=torch.long)
    changed = run_pipeline(pseudo_label_confidence=0.8, y_override=changed_hidden_y)

    assert_tensor_equal("selected_mask", strict["selected_mask"], changed["selected_mask"])
    assert_tensor_equal("successful_recovered_mask", strict["successful_recovered_mask"], changed["successful_recovered_mask"])
    assert_tensor_equal("pseudo_train_mask", strict["pseudo_train_mask"], changed["pseudo_train_mask"])
    assert_tensor_equal("train_mask", strict["train_mask"], changed["train_mask"])
    assert_tensor_equal("full_pseudo_y", strict["pseudo_y"], changed["pseudo_y"])

    pseudo_nodes = strict["pseudo_train_mask"]
    assert pseudo_nodes.any(), "strict threshold should keep a high-confidence pseudo-labeled subset"
    assert_tensor_equal(
        "pseudo_y_on_high_confidence_nodes",
        strict["pseudo_y"][pseudo_nodes],
        changed["pseudo_y"][pseudo_nodes],
    )

    print("Selective pseudo-label filter verification passed.")


if __name__ == "__main__":
    main()
