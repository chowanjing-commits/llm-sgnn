# Equal-Label-Budget Controls

## Purpose

This diagnostic checks whether the reported gains of the recovery pipeline come from semantic edge construction or from reintroducing additional labeled training nodes. Each run uses the same dropped nodes, sampled recovered nodes, successful recovered training-label set, and random seed across all modes. Only the graph structure changes.

This is a necessary control because the current Ours pipeline uses labels from successfully recovered dropped training nodes, while sparse baselines use only the remaining observed training nodes.

## Protocol

- Datasets: Cora, PubMed, ogbn-arxiv
- Node drop rate: 0.95
- Edge drop rate: 0.0
- Backbone: GraphSAGE
- Runs: 3 seeds
- Epochs: 300
- Recovery ratio: 0.5
- k: 10
- Repair policy: adaptive
- Adaptive alpha: 0.0
- Max edges per recovered node: 10

Artifacts:

- Raw log: `logs/label_budget_controls_minimal_raw_20260527_102921.csv`
- Summary log: `logs/label_budget_controls_minimal_summary_20260527_102921.csv`
- Script: `scripts/utils/run_label_budget_controls.py`

## Modes

- Sparse LLM: uses only observed training labels and the sparse graph.
- Semantic Edges, No Recovered Labels: adds Ours semantic edges but keeps the training loss on observed labels only.
- Recovered Labels, No Edges: uses the same successful recovered labels as Ours but does not add repair edges.
- Recovered Labels, Random Edges: uses the same recovered labels and adds the same number of random edges as Ours.
- Ours: uses recovered labels and adaptive LLM semantic edges.
- Oracle Original Edges: uses the same recovered labels and reconnects their original graph edges to active nodes.

## Results

| Dataset | Mode | Acc (%) | Std | Observed train | Recovered train | Train nodes | Added edges | Edge hit (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | Sparse LLM | 54.03 | 6.27 | 13.0 | 0.0 | 13.0 | 0.0 | 0.00 |
| Cora | Semantic Edges, No Recovered Labels | 56.33 | 6.16 | 13.0 | 63.0 | 13.0 | 215.7 | 14.82 |
| Cora | Recovered Labels, No Edges | 73.13 | 0.85 | 13.0 | 63.0 | 76.0 | 0.0 | 0.00 |
| Cora | Recovered Labels, Random Edges | 67.87 | 0.54 | 13.0 | 63.0 | 76.0 | 215.7 | 0.15 |
| Cora | Ours: Recovered Labels + LLM Semantic Edges | 73.67 | 1.64 | 13.0 | 63.0 | 76.0 | 215.7 | 14.82 |
| Cora | Recovered Labels, Oracle Original Edges | 76.07 | 0.75 | 13.0 | 63.0 | 76.0 | 235.7 | 100.00 |
| PubMed | Sparse LLM | 47.23 | 9.74 | 3.0 | 0.0 | 3.0 | 0.0 | 0.00 |
| PubMed | Semantic Edges, No Recovered Labels | 46.57 | 9.70 | 3.0 | 28.0 | 3.0 | 106.7 | 8.74 |
| PubMed | Recovered Labels, No Edges | 75.80 | 3.35 | 3.0 | 28.0 | 31.0 | 0.0 | 0.00 |
| PubMed | Recovered Labels, Random Edges | 71.90 | 3.89 | 3.0 | 28.0 | 31.0 | 106.7 | 0.00 |
| PubMed | Ours: Recovered Labels + LLM Semantic Edges | 77.43 | 1.77 | 3.0 | 28.0 | 31.0 | 106.7 | 8.74 |
| PubMed | Recovered Labels, Oracle Original Edges | 78.07 | 1.55 | 3.0 | 28.0 | 31.0 | 111.7 | 100.00 |
| ogbn-arxiv | Sparse LLM | 57.88 | 0.30 | 4548.0 | 0.0 | 4548.0 | 0.0 | 0.00 |
| ogbn-arxiv | Semantic Edges, No Recovered Labels | 57.41 | 0.54 | 4548.0 | 43196.0 | 4548.0 | 126533.7 | 6.60 |
| ogbn-arxiv | Recovered Labels, No Edges | 62.96 | 0.27 | 4548.0 | 43196.0 | 47744.0 | 0.0 | 0.00 |
| ogbn-arxiv | Recovered Labels, Random Edges | 60.30 | 0.05 | 4548.0 | 43196.0 | 47744.0 | 126533.7 | 0.01 |
| ogbn-arxiv | Ours: Recovered Labels + LLM Semantic Edges | 61.94 | 0.18 | 4548.0 | 43196.0 | 47744.0 | 126533.7 | 6.60 |
| ogbn-arxiv | Recovered Labels, Oracle Original Edges | 62.10 | 0.26 | 4548.0 | 43196.0 | 47744.0 | 371591.0 | 100.00 |

## Interpretation

The main gain is largely explained by the recovered training-label budget. On Cora and PubMed, Ours is only slightly above Recovered Labels, No Edges. On ogbn-arxiv, Recovered Labels, No Edges is higher than Ours.

Semantic edges alone do not improve performance under this setting. Semantic Edges, No Recovered Labels is close to or below Sparse LLM, especially on PubMed and ogbn-arxiv. This means the edge construction is not sufficient by itself when the training loss does not include recovered labels.

Random matched edges are consistently worse than Ours, which suggests that semantic edges are less harmful and more useful than random structural noise under the same edge budget. However, this does not establish semantic edges as the primary source of the overall improvement.

Oracle original edges are only moderately above Ours on Cora and PubMed, and are not clearly better on ogbn-arxiv despite adding many more edges. This supports the conservative claim that the method builds task-useful pseudo-neighborhoods rather than reconstructing the original topology.

## Consequence for the Paper

The main claim should be revised. The method should not be presented as primarily proving that semantic edge repair drives the gains. A more defensible framing is:

> Under label-available training-node structural missingness, LLM representations support a controlled recovery pipeline that restores usable training supervision and constructs semantic pseudo-neighborhoods. The dominant gain comes from recovering labeled training nodes, while semantic edges provide a smaller, dataset-dependent regularization or connectivity effect and are preferable to random edge insertion under the same budget.

Recommended changes:

- Make equal-label-budget controls a core experiment, not an appendix-only check.
- Reduce claims about semantic edge construction being the main driver.
- Keep semantic edge construction as a controlled pseudo-neighborhood mechanism that avoids random or dense graph augmentation.
- Do not run the full 0.75/0.85/0.95 equal-label table before deciding whether the paper should shift to a label-available recovery pipeline framing.

## Hadamard-MLP Edge Scorer Check

We additionally tested a learned semantic edge scorer under the same equal-label-budget setting. The scorer is trained only on the corrupted graph: visible sparse edges are positive pairs, randomly sampled visible non-edges are negative pairs, and the input is the Hadamard product of two frozen LLM embeddings. It does not use node labels. At repair time, it reranks candidate neighbors for the same recovered training-label set, and the number of added edges is matched to the cosine-based Ours setting.

Protocol:

- Datasets: Cora, PubMed, ogbn-arxiv
- Node drop rate: 0.95
- Backbone: GraphSAGE
- Runs: 3 seeds
- Edge scorer epochs: 50
- Edge scorer hidden dim: 128
- Positive edge cap: 100,000
- Candidate prescreen: top-100 cosine candidates

Artifacts:

- Raw log: `logs/label_budget_controls_mlp_minimal_raw_20260528_110654.csv`
- Summary log: `logs/label_budget_controls_mlp_minimal_summary_20260528_110654.csv`

| Dataset | Mode | Acc (%) | Std | Added edges | Edge hit (%) | Repair time (s) |
|---|---|---:|---:|---:|---:|---:|
| Cora | Recovered Labels, No Edges | 73.27 | 0.78 | 0.0 | 0.00 | 0.00 |
| Cora | Recovered Labels, Random Edges | 68.17 | 0.12 | 215.7 | 0.15 | 0.01 |
| Cora | Recovered Labels, Hadamard-MLP Edges | 75.53 | 0.56 | 215.7 | 13.27 | 0.45 |
| Cora | Ours: Recovered Labels + LLM Semantic Edges | 74.90 | 1.07 | 215.7 | 14.82 | 0.02 |
| PubMed | Recovered Labels, No Edges | 75.80 | 3.35 | 0.0 | 0.00 | 0.00 |
| PubMed | Recovered Labels, Random Edges | 72.43 | 3.62 | 106.7 | 0.00 | 0.05 |
| PubMed | Recovered Labels, Hadamard-MLP Edges | 75.57 | 2.87 | 106.7 | 7.82 | 1.66 |
| PubMed | Ours: Recovered Labels + LLM Semantic Edges | 77.73 | 2.19 | 106.7 | 8.74 | 0.08 |
| ogbn-arxiv | Recovered Labels, No Edges | 63.03 | 0.27 | 0.0 | 0.00 | 0.00 |
| ogbn-arxiv | Recovered Labels, Random Edges | 60.53 | 0.10 | 126,533.7 | 0.01 | 0.42 |
| ogbn-arxiv | Recovered Labels, Hadamard-MLP Edges | 61.39 | 0.33 | 126,533.7 | 6.45 | 24.61 |
| ogbn-arxiv | Ours: Recovered Labels + LLM Semantic Edges | 62.30 | 0.27 | 126,533.7 | 6.60 | 9.56 |

Interpretation:

Hadamard-MLP is feasible, but it is not a clear replacement for cosine similarity in the current protocol. It improves Cora slightly over cosine Ours, but underperforms cosine Ours on PubMed and ogbn-arxiv while adding substantial repair-time overhead. This suggests that learned edge scoring may be useful as an ablation or future extension, but the main method should remain the simpler cosine-based adaptive repair unless further tuning or stronger edge-supervision design shows consistent gains.

## Hadamard-MLP Drop-Rate Curve

To test whether Hadamard-MLP underperforms mainly because the 0.95 setting leaves too little useful edge supervision, we repeated the equal-label-budget MLP check at node drop rates 0.75, 0.85, and 0.95.

Artifacts:

- Raw log: `logs/label_budget_controls_mlp_droprate_raw_20260528_113213.csv`
- Summary log: `logs/label_budget_controls_mlp_droprate_summary_20260528_113213.csv`

| Dataset | Drop | No edges | Random | Hadamard-MLP | Cosine Ours | MLP - Cosine |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.75 | 74.37 | 69.63 | 74.73 | 74.63 | +0.10 |
| Cora | 0.85 | 74.47 | 68.80 | 73.20 | 74.50 | -1.30 |
| Cora | 0.95 | 73.37 | 67.87 | 74.67 | 73.77 | +0.90 |
| PubMed | 0.75 | 78.53 | 79.37 | 78.73 | 79.33 | -0.60 |
| PubMed | 0.85 | 74.43 | 74.23 | 74.80 | 75.63 | -0.83 |
| PubMed | 0.95 | 75.80 | 72.50 | 76.10 | 77.83 | -1.73 |
| ogbn-arxiv | 0.75 | 63.42 | 61.23 | 62.52 | 62.70 | -0.18 |
| ogbn-arxiv | 0.85 | 63.04 | 61.01 | 61.81 | 62.59 | -0.78 |
| ogbn-arxiv | 0.95 | 62.89 | 60.20 | 61.76 | 62.04 | -0.28 |

This curve does not support the simple explanation that Hadamard-MLP fails only because the 0.95 setting has too little visible edge supervision. The learned scorer is not consistently better at lower drop rates. It is competitive on Cora, but remains below cosine Ours on PubMed and ogbn-arxiv. A more likely explanation is that the self-supervised edge prediction objective learned from visible sparse edges does not transfer reliably to recovered training nodes, and the Hadamard-only scorer adds complexity without consistently improving downstream classification.

## Original-Graph MLP Edge Scorer

We also tested a variant inspired by the LLM4NG edge predictor. Unlike the corrupted-graph Hadamard-MLP scorer above, this variant trains an MLP on the original graph: original edges are positives, and an equal number of non-edges are sampled as negatives. Candidate repair edges are first filtered by a cosine threshold and then reranked by the MLP. This is an oracle-style diagnostic because it trains the edge scorer using original topology that is unavailable under the corrupted training graph protocol.

Implementation details:

- Edge scorer input: concatenated frozen LLM embeddings \([z_u, z_v]\), following the LLM4NG-style predictor.
- Positive samples: original graph edges.
- Negative samples: same number of randomly sampled original non-edges.
- Prediction candidates: recovered training nodes to active candidate nodes with cosine similarity above 0.6.
- Added edge count: matched to cosine Ours.
- Node drop rate: 0.95.
- Backbone: GraphSAGE.
- Runs: 3 seeds.

Artifacts:

- Raw log: `logs/label_budget_controls_original_graph_mlp_minimal_raw_20260528_160054.csv`
- Summary log: `logs/label_budget_controls_original_graph_mlp_minimal_summary_20260528_160054.csv`

| Dataset | Mode | Acc (%) | Std | Added edges | Edge hit (%) | Repair time (s) |
|---|---|---:|---:|---:|---:|---:|
| Cora | Recovered Labels, No Edges | 73.30 | 0.82 | 0.0 | 0.00 | 0.00 |
| Cora | Recovered Labels, Original-Graph MLP Edges | 67.90 | 2.67 | 215.7 | 2.01 | 0.40 |
| Cora | Ours: Recovered Labels + LLM Semantic Edges | 74.47 | 1.36 | 215.7 | 14.82 | 0.01 |
| Cora | Recovered Labels, Oracle Original Edges | 76.07 | 0.75 | 235.7 | 100.00 | 0.06 |
| PubMed | Recovered Labels, No Edges | 75.80 | 3.35 | 0.0 | 0.00 | 0.00 |
| PubMed | Recovered Labels, Original-Graph MLP Edges | 71.57 | 6.68 | 106.7 | 0.32 | 1.62 |
| PubMed | Ours: Recovered Labels + LLM Semantic Edges | 77.43 | 1.77 | 106.7 | 8.74 | 0.08 |
| PubMed | Recovered Labels, Oracle Original Edges | 78.07 | 1.55 | 111.7 | 100.00 | 1.78 |
| ogbn-arxiv | Recovered Labels, No Edges | 62.84 | 0.22 | 0.0 | 0.00 | 0.00 |
| ogbn-arxiv | Recovered Labels, Original-Graph MLP Edges | 61.89 | 0.01 | 126,533.7 | 0.02 | 458.99 |
| ogbn-arxiv | Ours: Recovered Labels + LLM Semantic Edges | 61.96 | 0.16 | 126,533.7 | 6.60 | 18.84 |
| ogbn-arxiv | Recovered Labels, Oracle Original Edges | 62.39 | 0.40 | 371,591.0 | 100.00 | 15.20 |

This oracle-style MLP scorer does not improve the repair pipeline. It is worse than cosine Ours on Cora and PubMed, close but still lower on ogbn-arxiv, and substantially more expensive on the large graph. Its edge hit rate is also much lower than cosine Ours. This suggests that the LLM4NG-style edge prediction objective is not well aligned with our recovered-training-node pseudo-neighborhood task. The result further supports keeping the main method as a lightweight cosine-based adaptive repair rather than adding a learned edge predictor.
