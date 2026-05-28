# arXiv Fixed-k Node Drop Experiment

Date: 2026-05-22

## Setting

- Dataset: full ogbn-arxiv
- Nodes: 169,343
- Original edges: 1,166,243
- Split: official ogbn-arxiv split
- Seeds: 1
- Epochs: 300
- Edge drop rate: 0
- Node drop rates: 0.75, 0.85, 0.95
- Repair k: 10
- LLM feature source: cached arXiv title/text embeddings in `embeddings/arxiv_llm_emb.pt`

## Node Drop Protocol

- Node drop is sampled only from training nodes.
- Dropped training nodes are removed from the supervised training loss.
- Incident edges of dropped training nodes are removed from the sparse graph.
- Validation and test masks follow the official split and are not dropped.
- Repair uses dropped nodes as repair targets.
- Repair candidates are initialized from non-dropped nodes; once a dropped node is repaired, it becomes available as a candidate for later repaired nodes.
- Repair similarity is computed from the original text-derived LLM embeddings passed to `LLM_GNN`.

## Results

Accuracy is test accuracy selected by best validation accuracy.

| Node drop | Train nodes | Sparse edges | MLP Raw | MLP LLM | GCN Raw Sparse | GCN LLM Sparse | Ours Repair |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 22,736 | 379,402 | 52.37 | 59.28 | 52.48 | 58.51 | 68.29 |
| 0.85 | 13,642 | 306,987 | 52.47 | 58.47 | 51.80 | 56.48 | 66.79 |
| 0.95 | 4,548 | 242,470 | 50.52 | 55.45 | 49.34 | 55.18 | 63.94 |

## Files

- Consolidated CSV: `logs/arxiv_full_fixed_k10_node_drop_20260522.csv`
- Node drop 0.75: `logs/arxiv_single_pilot_drop0_node75_20260522_113608.csv`
- Node drop 0.85: `logs/arxiv_single_pilot_drop0_node85_20260522_113947.csv`
- Node drop 0.95: `logs/arxiv_single_pilot_drop0_node95_20260522_114346.csv`

## Immediate Interpretation

Fixed `k=10` Repair consistently outperforms `GCN (LLM, Sparse)` by 8.76 to 10.31 points. Accuracy still decreases as node drop increases, likely because the supervised training set shrinks sharply and the fixed repair budget may under-compensate under extreme sparsity.

## k Sweep at Node Drop 0.95

After observing that accuracy decreases at higher node drop, we tested whether increasing the repair k improves the extreme `node_drop_rate=0.95` setting. Only `Ours (LLM-GNN Repair)` was rerun because MLP and sparse GCN baselines do not depend on k.

| k | Ours Repair |
|---:|---:|
| 10 | 63.94 |
| 20 | 63.91 |
| 30 | 63.89 |
| 50 | 64.03 |

CSV: `logs/arxiv_full_node95_k_sweep_20260522.csv`

Interpretation: increasing k from 10 to 20/30/50 does not materially improve accuracy at `node_drop_rate=0.95`. Under the current repair strategy, the performance drop is therefore less likely to be caused by k being too small, and more likely related to the sharply reduced supervised training set, noisy semantic candidates, or the repair budget/connection policy.
