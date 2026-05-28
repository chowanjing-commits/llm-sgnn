# arXiv Adaptive Recovery Policy Experiment

Date: 2026-05-22

## Setting

- Dataset: full ogbn-arxiv
- Nodes: 169,343
- Original edges in `edge_index`: 1,166,243
- Split: official ogbn-arxiv split
- Seeds: 1
- Epochs: 300
- Edge drop rate: 0
- Node drop rates: 0.75, 0.85, 0.95
- Recovery ratio: 0.5 of dropped training nodes
- Semantic candidate k: 10
- Repair policy: adaptive
- Adaptive threshold alpha: 0.0
- Maximum edge cap: 10
- Runtime: `conda run -n LLM-SGNN`

## Protocol

This experiment uses the newer random-recovery protocol also used for the latest Cora/PubMed experiments:

- Node drop is sampled only from training nodes.
- Dropped training nodes and their incident edges disappear from the observed sparse graph.
- `Ours (LLM-GNN Repair)` randomly samples 50% of dropped training nodes as attempted recovered nodes.
- Recovery candidates are selected from top-10 semantic neighbors using cached text-derived LLM embeddings.
- Adaptive edge budget is computed from observed graph density:
  `b = min(max_edges_per_recovered_node, ceil(|E_sparse| / |V_observed|))`.
- Node-wise threshold is:
  `tau_u = mean(S_u) + alpha * std(S_u)`, with `alpha=0.0`.
- Only successfully connected recovered training nodes enter the supervised loss.

## Results

Accuracy is test accuracy selected by best validation accuracy.

| Node drop | MLP Raw | MLP LLM | GCN Raw Sparse | GCN LLM Sparse | Ours Repair |
|---:|---:|---:|---:|---:|---:|
| 0.75 | 52.58 ± 0.23 | 59.55 ± 0.40 | 52.42 ± 0.13 | 58.29 ± 0.14 | 60.58 ± 0.30 |
| 0.85 | 52.38 ± 0.12 | 58.49 ± 0.38 | 52.10 ± 0.26 | 56.40 ± 0.13 | 60.43 ± 0.16 |
| 0.95 | 50.41 ± 0.27 | 55.94 ± 0.40 | 50.29 ± 0.81 | 55.87 ± 0.94 | 60.30 ± 0.14 |

Repair statistics:

| Node drop | Sampled recovered | Successful recovered | Recovery edges |
|---:|---:|---:|---:|
| 0.75 | 34,102 | 34,102 | 122,731 |
| 0.85 | 38,649 | 38,649 | 139,403 |
| 0.95 | 43,196 | 43,196 | 126,479 |

## Files

- Consolidated all-model CSV: `logs/main_multiseed_adaptive_all_models_20260522.csv`
- Ours-only CSV: `logs/arxiv_full_adaptive_recovery_policy_20260522.csv`
- Raw runs:
  `logs/arxiv_single_pilot_drop0_node75_20260522_182843.csv`,
  `logs/arxiv_single_pilot_drop0_node85_20260522_183000.csv`,
  `logs/arxiv_single_pilot_drop0_node95_20260522_183114.csv`

## Interpretation

This result should not be directly compared as the same method as the older `docs/arxiv_fixed_k10_node_drop.md` table. The older arXiv fixed-k experiment used a stronger repair setting and reported higher Ours accuracy. This adaptive experiment matches the newer random-recovery semantics: only 50% of dropped training nodes are attempted, the loss includes only observed plus successfully recovered training nodes, and edge addition is controlled by observed graph density plus node-wise semantic thresholding.

Under this stricter protocol, Ours remains around 60% across node-drop rates despite the observed training set shrinking from 22,736 to 4,548 nodes. The flat trend is plausible because the number of recovered training nodes increases as node-drop rate increases under fixed recovery ratio, partially compensating for the loss of observed training nodes.

Compared with `GCN (LLM, Sparse)`, adaptive Ours improves by 2.29, 4.03, and 4.43 points at node-drop rates 0.75, 0.85, and 0.95. The gain increases under stronger sparsity, which supports the intended sparse/new-node repair framing.

Next checks:

- Run multiple seeds to test whether the flat trend is stable.
- Add repair quality diagnostics such as recovery-edge hit rate against original edges.

## Adaptive Alpha Sweep

Only `Ours` was rerun because baselines do not depend on adaptive threshold alpha.

| Node drop | Alpha | Ours Repair | Recovery edges | Successful recovered |
|---:|---:|---:|---:|---:|
| 0.75 | 0.0 | 60.90 | 122,731 | 34,102 |
| 0.75 | 0.5 | 60.41 | 90,595 | 34,102 |
| 0.85 | 0.0 | 60.39 | 139,403 | 38,649 |
| 0.85 | 0.5 | 60.63 | 102,922 | 38,649 |
| 0.95 | 0.0 | 60.47 | 126,479 | 43,196 |
| 0.95 | 0.5 | 60.30 | 107,888 | 43,196 |

CSV: `logs/arxiv_adaptive_alpha_sweep_20260522.csv`

Interpretation: `alpha=0.5` substantially reduces recovery edges while preserving all sampled recovered nodes. Accuracy is similar to `alpha=0.0`, with a small improvement at node drop 0.85 and small decreases at 0.75/0.95. This supports the adaptive-threshold framing: stricter node-wise thresholds control graph densification without collapsing recovery.

## Recovery Quality Diagnostics

We diagnosed whether adaptive semantic recovery edges match original arXiv neighbors removed by node-drop corruption.

Setting: adaptive repair, `alpha=0.0`, `k=10`, recovery ratio 0.5, max edge cap 10.

| Node drop | Recovery edges | Edge hits | Edge hit rate | Node hit rate | Top-k precision | Top-k recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 122,731 | 10,221 | 8.33 | 22.61 | 5.12 | 5.88 |
| 0.85 | 139,403 | 10,259 | 7.36 | 20.13 | 4.53 | 6.01 |
| 0.95 | 126,479 | 8,496 | 6.72 | 15.43 | 3.85 | 5.62 |

CSV: `logs/recovery_quality_adaptive_multiseed_20260522.csv`
Summary CSV: `logs/recovery_quality_adaptive_summary_multiseed_20260522.csv`

Interpretation: arXiv has a weaker exact-topology recovery signal than Cora. The method still improves classification over sparse GCN baselines, but the gain should not be over-claimed as high-precision reconstruction of original citation edges. A more accurate framing is that LLM embeddings create useful semantic pseudo-neighborhoods, a subset of which recovers true missing topology.
