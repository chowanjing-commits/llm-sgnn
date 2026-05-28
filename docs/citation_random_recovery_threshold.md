# Cora and PubMed Random Node Recovery with Thresholded Semantic Edges

Date: 2026-05-22

## Setting

- Datasets: Cora, PubMed
- Split: Planetoid-style public split generated with seed 42
- Seeds: 1
- Epochs: 300
- Edge drop rate: 0
- Node drop rates: 0.75, 0.85, 0.95
- Recovery ratio: 0.5 of dropped training nodes
- Semantic candidate k: 10
- Similarity threshold: 0.6
- Max edges per recovered node: 5
- Runtime: all runs executed with `conda run -n LLM-SGNN`

## Protocol

- Node drop is sampled only from training nodes with available raw text.
- Dropped training nodes and their incident edges disappear from the observed sparse graph.
- Baselines train only on observed training nodes after node drop.
- `Ours (LLM-GNN Repair)` randomly samples 50% of dropped training nodes as attempted recovered nodes.
- Recovery edges are built from cached text-derived LLM embeddings.
- For each attempted recovered node, the runner first selects top-10 semantic candidate nodes, then keeps candidates with cosine similarity at least 0.6, then adds at most 5 undirected edges.
- Only successfully connected recovered training nodes are added to the supervised loss.
- `Ours` loss mask is therefore `observed_train_nodes + successfully_recovered_train_nodes`, not the original full training mask.
- `beta` is not active in this runner-based threshold recovery path; it remains in the CSV only for compatibility with older experiments.

## Cora Results

Accuracy is test accuracy selected by best validation accuracy.

| Node drop | Observed train | Sampled recovered | Successful recovered | Recovery edges | Ours train | MLP Raw | MLP LLM | GCN Raw Sparse | GCN LLM Sparse | Ours Repair |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 40 | 50 | 50 | 250 | 90 | 43.40 | 59.20 | 71.10 | 71.40 | 79.80 |
| 0.85 | 27 | 56 | 56 | 280 | 83 | 37.30 | 53.50 | 70.00 | 70.80 | 71.80 |
| 0.95 | 13 | 63 | 63 | 315 | 76 | 29.20 | 48.60 | 61.30 | 56.00 | 75.20 |

## PubMed Results

| Node drop | Observed train | Sampled recovered | Successful recovered | Recovery edges | Ours train | MLP Raw | MLP LLM | GCN Raw Sparse | GCN LLM Sparse | Ours Repair |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 15 | 22 | 22 | 110 | 37 | 54.00 | 66.60 | 58.80 | 67.60 | 77.70 |
| 0.85 | 9 | 25 | 25 | 125 | 34 | 46.20 | 54.10 | 50.20 | 47.40 | 71.70 |
| 0.95 | 3 | 28 | 28 | 140 | 31 | 39.20 | 43.70 | 40.40 | 39.60 | 79.20 |

## Files

- Consolidated CSV: `logs/citation_random_recovery_r50_thr06_k10_20260522.csv`
- Cora raw runs: `logs/cora_single_pilot_drop0_node75_20260522_165740.csv`, `logs/cora_single_pilot_drop0_node85_20260522_165751.csv`, `logs/cora_single_pilot_drop0_node95_20260522_165803.csv`
- PubMed raw runs: `logs/pubmed_single_pilot_drop0_node75_20260522_165823.csv`, `logs/pubmed_single_pilot_drop0_node85_20260522_165843.csv`, `logs/pubmed_single_pilot_drop0_node95_20260522_165903.csv`

## Notes

- At threshold 0.6, every sampled recovered node succeeded and received the full 5 semantic edges. This means the threshold is currently not restrictive for Cora/PubMed under top-10 candidate search.
- Planetoid Cora/PubMed have very small training splits, especially PubMed with only 60 training nodes. High node drop rates leave 15, 9, and 3 observed training nodes, so one-seed results should be treated as pilot evidence rather than final claims.

## Threshold and Edge-Cap Sensitivity

Only `Ours` was rerun because baselines do not depend on recovery threshold or recovery edge cap.

| Dataset | Node drop | Threshold | Max edges | Successful recovered | Recovery edges | Ours train | Ours Repair |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 0.75 | 0.6 | 5 | 50 | 250 | 90 | 79.80 |
| Cora | 0.75 | 0.7 | 5 | 50 | 250 | 90 | 79.80 |
| Cora | 0.75 | 0.8 | 5 | 50 | 246 | 90 | 79.20 |
| Cora | 0.75 | 0.8 | 3 | 50 | 148 | 90 | 78.30 |
| Cora | 0.85 | 0.6 | 5 | 56 | 280 | 83 | 71.80 |
| Cora | 0.85 | 0.7 | 5 | 56 | 280 | 83 | 72.10 |
| Cora | 0.85 | 0.8 | 5 | 55 | 274 | 82 | 72.80 |
| Cora | 0.85 | 0.8 | 3 | 55 | 165 | 82 | 74.20 |
| Cora | 0.95 | 0.6 | 5 | 63 | 315 | 76 | 75.20 |
| Cora | 0.95 | 0.7 | 5 | 63 | 315 | 76 | 75.20 |
| Cora | 0.95 | 0.8 | 5 | 62 | 305 | 75 | 74.00 |
| Cora | 0.95 | 0.8 | 3 | 62 | 184 | 75 | 74.00 |
| PubMed | 0.75 | 0.6 | 5 | 22 | 110 | 37 | 77.70 |
| PubMed | 0.75 | 0.7 | 5 | 22 | 110 | 37 | 77.70 |
| PubMed | 0.75 | 0.8 | 5 | 22 | 110 | 37 | 77.70 |
| PubMed | 0.75 | 0.8 | 3 | 22 | 66 | 37 | 78.20 |
| PubMed | 0.85 | 0.6 | 5 | 25 | 125 | 34 | 71.70 |
| PubMed | 0.85 | 0.7 | 5 | 25 | 125 | 34 | 72.00 |
| PubMed | 0.85 | 0.8 | 5 | 25 | 125 | 34 | 72.00 |
| PubMed | 0.85 | 0.8 | 3 | 25 | 75 | 34 | 72.30 |
| PubMed | 0.95 | 0.6 | 5 | 28 | 140 | 31 | 79.20 |
| PubMed | 0.95 | 0.7 | 5 | 28 | 140 | 31 | 79.20 |
| PubMed | 0.95 | 0.8 | 5 | 28 | 140 | 31 | 79.20 |
| PubMed | 0.95 | 0.8 | 3 | 28 | 84 | 31 | 76.70 |

Sensitivity CSV: `logs/citation_random_recovery_threshold_sensitivity_20260522.csv`

Current interpretation:

- Thresholds 0.6 and 0.7 are effectively equivalent.
- Threshold 0.8 only starts filtering Cora slightly and still does not affect PubMed under top-10 candidate search.
- Reducing the edge cap from 5 to 3 cuts recovery edges substantially. It is not uniformly worse: Cora 0.85 and PubMed 0.75/0.85 improve, while PubMed 0.95 drops.
- If we want a conservative graph repair setting, `max_edges_per_recovered_node=3` is more defensible than 5 for Cora-like graphs.

## Extended Sensitivity at Thresholds 0.85 and 0.90

| Dataset | Node drop | Threshold | Max edges | Successful recovered | Recovery edges | Ours train | Ours Repair |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 0.75 | 0.85 | 5 | 47 | 228 | 87 | 79.30 |
| Cora | 0.75 | 0.85 | 3 | 47 | 138 | 87 | 78.80 |
| Cora | 0.75 | 0.90 | 5 | 33 | 100 | 73 | 78.00 |
| Cora | 0.75 | 0.90 | 3 | 33 | 75 | 73 | 77.30 |
| Cora | 0.85 | 0.85 | 5 | 54 | 241 | 81 | 76.30 |
| Cora | 0.85 | 0.85 | 3 | 54 | 151 | 81 | 75.10 |
| Cora | 0.85 | 0.90 | 5 | 35 | 104 | 62 | 78.20 |
| Cora | 0.85 | 0.90 | 3 | 35 | 78 | 62 | 76.90 |
| Cora | 0.95 | 0.85 | 5 | 58 | 270 | 71 | 74.60 |
| Cora | 0.95 | 0.85 | 3 | 58 | 166 | 71 | 75.90 |
| Cora | 0.95 | 0.90 | 5 | 39 | 126 | 52 | 77.00 |
| Cora | 0.95 | 0.90 | 3 | 39 | 94 | 52 | 76.50 |
| PubMed | 0.75 | 0.85 | 5 | 22 | 110 | 37 | 77.70 |
| PubMed | 0.75 | 0.85 | 3 | 22 | 66 | 37 | 78.20 |
| PubMed | 0.75 | 0.90 | 5 | 22 | 99 | 37 | 77.90 |
| PubMed | 0.75 | 0.90 | 3 | 22 | 61 | 37 | 78.50 |
| PubMed | 0.85 | 0.85 | 5 | 25 | 125 | 34 | 72.90 |
| PubMed | 0.85 | 0.85 | 3 | 25 | 75 | 34 | 71.70 |
| PubMed | 0.85 | 0.90 | 5 | 25 | 121 | 34 | 72.20 |
| PubMed | 0.85 | 0.90 | 3 | 25 | 73 | 34 | 71.40 |
| PubMed | 0.95 | 0.85 | 5 | 28 | 140 | 31 | 79.20 |
| PubMed | 0.95 | 0.85 | 3 | 28 | 84 | 31 | 76.70 |
| PubMed | 0.95 | 0.90 | 5 | 28 | 128 | 31 | 76.70 |
| PubMed | 0.95 | 0.90 | 3 | 28 | 78 | 31 | 78.10 |

Extended interpretation:

- Cora becomes threshold-sensitive at 0.85 and strongly threshold-sensitive at 0.90. The number of successful recovered nodes drops, but accuracy can improve at high node-drop rates, suggesting that stricter semantic filtering removes noisy recovered nodes or noisy edges.
- PubMed remains recovery-count insensitive up to threshold 0.90: all sampled recovered nodes still connect successfully. The threshold mostly reduces edge count, not recovered-node count.
- The best one-seed Cora settings in this grid are `0.75: threshold 0.6/0.7, cap 5`, `0.85: threshold 0.90, cap 5`, and `0.95: threshold 0.90, cap 5`.
- The best one-seed PubMed settings are `0.75: threshold 0.90, cap 3`, `0.85: threshold 0.85, cap 5`, and `0.95: threshold 0.6/0.7/0.8/0.85, cap 5`.
- For a single conservative default across Cora/PubMed, `threshold=0.85, max_edges_per_recovered_node=3` is defensible, but it is not uniformly best. For best pilot accuracy, the optimal threshold appears dataset- and drop-rate-dependent.

## Adaptive Repair Policy Pilot

To avoid selecting a global cosine threshold by grid search, we added an adaptive repair option:

- Candidate pool: top-10 semantic neighbors per sampled recovered node.
- Degree-aware edge budget: `b = min(max_edges_per_recovered_node, ceil(|E_sparse| / |V_observed|))`.
- Node-wise threshold: `tau_u = mean(S_u) + alpha * std(S_u)`, where `S_u` is the recovered node's top-10 similarity set.
- A candidate is connected only if its similarity is at least `tau_u`.

In these runs, `max_edges_per_recovered_node` was set to 10, so the effective budget is determined by observed graph density unless the density exceeds 10.

| Dataset | Node drop | Alpha | Successful recovered | Recovery edges | Ours train | Ours Repair |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.75 | 0.0 | 50 | 171 | 90 | 79.30 |
| Cora | 0.85 | 0.0 | 56 | 193 | 83 | 74.20 |
| Cora | 0.95 | 0.0 | 63 | 218 | 76 | 74.80 |
| PubMed | 0.75 | 0.0 | 22 | 84 | 37 | 77.30 |
| PubMed | 0.85 | 0.0 | 25 | 102 | 34 | 72.90 |
| PubMed | 0.95 | 0.0 | 28 | 105 | 31 | 80.20 |
| Cora | 0.75 | 0.5 | 50 | 119 | 90 | 79.20 |
| Cora | 0.85 | 0.5 | 56 | 132 | 83 | 74.80 |
| Cora | 0.95 | 0.5 | 63 | 167 | 76 | 75.50 |
| PubMed | 0.75 | 0.5 | 22 | 59 | 37 | 76.80 |
| PubMed | 0.85 | 0.5 | 25 | 73 | 34 | 72.30 |
| PubMed | 0.95 | 0.5 | 28 | 75 | 31 | 77.50 |

Adaptive CSV: `logs/citation_adaptive_recovery_policy_20260522.csv`

Adaptive interpretation:

- Adaptive repair avoids a global cosine threshold and reduces recovery edges compared with fixed cap 5 while keeping all sampled nodes recoverable in these Cora/PubMed runs.
- `alpha=0.0` is a strong simple default. It matches or improves several fixed-threshold settings and reaches the best PubMed 0.95 result in this pilot grid.
- `alpha=0.5` is more conservative. It reduces edge count further, but can under-connect PubMed at high node drop.
- This adaptive policy is more suitable for the paper's main method than validation-grid threshold selection because it is derived from observed graph density and node-wise similarity distributions.

## Adaptive Multi-Seed Main Results

Setting: adaptive repair, `alpha=0.0`, `k=10`, recovery ratio 0.5, max edge cap 10, 3 seeds, 300 epochs.

| Dataset | Node drop | MLP Raw | MLP LLM | GCN Raw Sparse | GCN LLM Sparse | Ours Repair |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.75 | 43.90 ± 2.76 | 58.33 ± 0.84 | 73.20 ± 1.55 | 72.50 ± 1.77 | 77.30 ± 1.93 |
| Cora | 0.85 | 36.67 ± 0.45 | 51.63 ± 2.64 | 67.27 ± 2.04 | 67.93 ± 3.77 | 76.53 ± 1.70 |
| Cora | 0.95 | 26.37 ± 2.25 | 40.07 ± 6.14 | 53.60 ± 7.30 | 56.53 ± 3.73 | 75.63 ± 0.60 |
| PubMed | 0.75 | 53.63 ± 1.37 | 62.33 ± 3.23 | 61.80 ± 2.30 | 68.70 ± 1.92 | 78.30 ± 1.49 |
| PubMed | 0.85 | 47.90 ± 3.44 | 53.17 ± 0.93 | 51.00 ± 1.65 | 52.07 ± 3.40 | 75.67 ± 3.30 |
| PubMed | 0.95 | 42.73 ± 5.57 | 46.27 ± 7.69 | 46.03 ± 9.42 | 46.17 ± 10.15 | 78.07 ± 2.88 |

CSV: `logs/main_multiseed_adaptive_all_models_20260522.csv`

## Recovery Quality Diagnostics

We diagnosed whether semantic recovery edges match original graph neighbors removed by node-drop corruption. This is not required for training, but it helps distinguish true topology recovery from generic semantic pseudo-neighborhood construction.

Setting: adaptive repair, `alpha=0.0`, `k=10`, recovery ratio 0.5, max edge cap 10.

| Dataset | Node drop | Recovery edges | Edge hits | Edge hit rate | Node hit rate | Top-k precision | Top-k recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 0.75 | 171 | 33 | 19.30 | 46.00 | 9.80 | 27.37 |
| Cora | 0.85 | 193 | 31 | 16.06 | 41.07 | 8.75 | 19.84 |
| Cora | 0.95 | 218 | 30 | 13.76 | 38.10 | 6.67 | 20.39 |
| PubMed | 0.75 | 84 | 10 | 11.90 | 36.36 | 5.00 | 8.15 |
| PubMed | 0.85 | 102 | 6 | 5.88 | 24.00 | 3.60 | 8.11 |
| PubMed | 0.95 | 105 | 8 | 7.62 | 28.57 | 3.21 | 9.38 |

CSV: `logs/recovery_quality_adaptive_multiseed_20260522.csv`
Summary CSV: `logs/recovery_quality_adaptive_summary_multiseed_20260522.csv`

Interpretation:

- Cora has the strongest recovery-quality signal: about 14-19% of recovery edges match original missing neighbors, and 38-46% of recovered nodes receive at least one original-neighbor edge.
- PubMed has weaker topology recovery signal, especially at node drop 0.85/0.95.
- Accuracy gains should therefore be described carefully. The method partially recovers original topology, but also constructs useful semantic pseudo-neighborhoods that may not exactly match original citation edges.
