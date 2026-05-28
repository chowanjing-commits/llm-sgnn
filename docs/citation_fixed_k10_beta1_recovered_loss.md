# Cora and PubMed Node Recovery Experiment

Date: 2026-05-22

## Setting

- Datasets: Cora, PubMed
- Split: Planetoid-style public split generated with seed 42
- Train/Val/Test: 20 nodes per class for training, 500 validation, 1000 test
- Seeds: 1
- Epochs: 300
- Edge drop rate: 0
- Node drop rates: 0.75, 0.85, 0.95
- Repair k: 10
- Repair beta: 1.0
- Runtime: all runs executed with `conda run -n LLM-SGNN`

## Current Node Recovery Protocol

- Node drop is sampled only from training nodes with real text.
- Dropped training nodes and their incident edges disappear from the observed sparse graph.
- Baselines train only on observed training nodes after node drop.
- `Ours (LLM-GNN Repair)` uses the dropped nodes' cached text-derived LLM embeddings to reconnect them to semantic neighbors.
- `Ours` computes supervised loss on observed training nodes plus successfully recovered training nodes.
- With `beta=1.0`, the repair budget for a node is its full missing degree.
- Repair candidates are initialized from non-dropped nodes with real text; once a dropped node is repaired, it becomes available as a candidate for later repaired nodes.

## Cora Results

Accuracy is test accuracy selected by best validation accuracy.

| Node drop | Observed train | Ours train | MLP Raw | MLP LLM | GCN Raw Sparse | GCN LLM Sparse | Ours Repair |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 40 | 140 | 43.40 | 59.20 | 71.10 | 71.40 | 77.10 |
| 0.85 | 27 | 140 | 37.30 | 53.50 | 70.00 | 70.80 | 77.40 |
| 0.95 | 13 | 140 | 29.20 | 48.60 | 61.30 | 56.00 | 77.20 |

## PubMed Results

| Node drop | Observed train | Ours train | MLP Raw | MLP LLM | GCN Raw Sparse | GCN LLM Sparse | Ours Repair |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 15 | 60 | 54.00 | 66.60 | 58.80 | 67.60 | 82.60 |
| 0.85 | 9 | 60 | 46.20 | 54.10 | 50.20 | 47.40 | 82.00 |
| 0.95 | 3 | 60 | 39.20 | 43.70 | 40.40 | 39.60 | 82.00 |

## Files

- Consolidated CSV: `logs/citation_fixed_k10_beta1_recovered_loss_20260522.csv`
- Cora raw runs: `logs/cora_single_pilot_drop0_node75_20260522_163721.csv`, `logs/cora_single_pilot_drop0_node85_20260522_163733.csv`, `logs/cora_single_pilot_drop0_node95_20260522_163745.csv`
- PubMed raw runs: `logs/pubmed_single_pilot_drop0_node75_20260522_163805.csv`, `logs/pubmed_single_pilot_drop0_node85_20260522_163828.csv`, `logs/pubmed_single_pilot_drop0_node95_20260522_163848.csv`

## Immediate Interpretation

Under the corrected node recovery protocol, `Ours` clearly outperforms baselines because baselines only see the post-drop observed training nodes, while `Ours` can recover dropped training nodes through LLM semantic edges and include successfully recovered nodes in supervised loss.

This protocol matches the intended setting: a sparse graph receives new or missing nodes with text, and the model uses LLM embeddings to connect them back into the graph before training.
