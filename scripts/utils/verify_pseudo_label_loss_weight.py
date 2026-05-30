"""
Verify pseudo-label loss weighting for cold-start training.

In edge-only cold-start runs, pseudo-labeled nodes stay in the recovered graph
for message passing but receive zero supervised-loss weight. This check asserts
that changing pseudo labels cannot change the supervised loss when that weight
is zero.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.cold_start import weighted_supervised_loss


def main():
    out = torch.tensor(
        [
            [3.0, 0.1, -0.2],
            [0.2, 2.5, -0.1],
            [2.0, 0.0, -1.0],
            [-0.5, 0.3, 2.0],
        ],
        dtype=torch.float,
    )
    train_mask = torch.tensor([True, True, True, True])
    pseudo_train_mask = torch.tensor([False, False, True, True])
    base_y = torch.tensor([0, 1, 0, 2], dtype=torch.long)
    changed_pseudo_y = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    edge_only_loss = weighted_supervised_loss(
        out=out,
        train_y=base_y,
        train_mask=train_mask,
        pseudo_train_mask=pseudo_train_mask,
        pseudo_label_loss_weight=0.0,
    )
    changed_edge_only_loss = weighted_supervised_loss(
        out=out,
        train_y=changed_pseudo_y,
        train_mask=train_mask,
        pseudo_train_mask=pseudo_train_mask,
        pseudo_label_loss_weight=0.0,
    )
    if not torch.allclose(edge_only_loss, changed_edge_only_loss):
        raise AssertionError("zero-weight pseudo labels must not affect the supervised loss")

    pseudo_supervised_loss = weighted_supervised_loss(
        out=out,
        train_y=base_y,
        train_mask=train_mask,
        pseudo_train_mask=pseudo_train_mask,
        pseudo_label_loss_weight=1.0,
    )
    changed_pseudo_supervised_loss = weighted_supervised_loss(
        out=out,
        train_y=changed_pseudo_y,
        train_mask=train_mask,
        pseudo_train_mask=pseudo_train_mask,
        pseudo_label_loss_weight=1.0,
    )
    if torch.allclose(pseudo_supervised_loss, changed_pseudo_supervised_loss):
        raise AssertionError("positive-weight pseudo labels should affect the supervised loss")

    pseudo_only_mask = torch.tensor([False, False, True, True])
    try:
        weighted_supervised_loss(
            out=out,
            train_y=base_y,
            train_mask=pseudo_only_mask,
            pseudo_train_mask=pseudo_train_mask,
            pseudo_label_loss_weight=0.0,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("zero-weight pseudo-only batches must fail without observed training nodes")

    print("Pseudo-label loss-weight verification passed.")


if __name__ == "__main__":
    main()
