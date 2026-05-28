# Cora and PubMed Fixed-k Node Drop Experiment

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
- Runtime: all runs executed with `conda run -n LLM-SGNN`

## Text Protocol

- Cora uses `Title + Abstract` from official McCallum extraction files.
- Cora text coverage is 2636/2708 nodes. Nodes without real extraction text are not sampled as node-drop repair targets and are not used as semantic repair candidates.
- PubMed uses `TI + AB` from `pubmed.json`.
- PubMed text coverage is 19716/19717 nodes. The missing-text node is excluded from node-drop repair targets and semantic repair candidates.
- Text is lightly cleaned by removing field prefixes, SGML/XML-like tags, and repeated whitespace. No stemming or stopword removal is applied before BERT tokenization.

## Node Drop Protocol

- Node drop is sampled only from training nodes with real text.
- Dropped training nodes are removed from supervised training loss.
- Incident edges of dropped training nodes are removed from the sparse graph.
- Dropped nodes remain repair targets for `Ours (LLM-GNN Repair)`.
- Repair candidates are initialized from non-dropped nodes with real text; once a dropped node is repaired, it becomes available as a candidate for later repaired nodes.
- Validation and test masks are unchanged.

## Cora Results

Accuracy is test accuracy selected by best validation accuracy.

| Node drop | Train nodes | Sparse edges | MLP Raw | MLP LLM | GCN Raw Sparse | GCN LLM Sparse | Ours Repair |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 34 | 9,742 | 40.90 | 58.50 | 66.40 | 74.40 | 73.20 |
| 0.85 | 21 | 9,672 | 31.80 | 54.00 | 56.10 | 66.30 | 66.80 |
| 0.95 | 7 | 9,516 | 22.20 | 47.50 | 45.30 | 53.50 | 53.40 |

## PubMed Results

| Node drop | Train nodes | Sparse edges | MLP Raw | MLP LLM | GCN Raw Sparse | GCN LLM Sparse | Ours Repair |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 15 | 88,198 | 54.00 | 66.60 | 58.80 | 67.60 | 67.70 |
| 0.85 | 9 | 88,162 | 46.20 | 54.10 | 50.20 | 47.40 | 46.80 |
| 0.95 | 3 | 88,140 | 39.20 | 43.70 | 40.40 | 39.60 | 39.60 |

## Files

- Consolidated CSV: `logs/citation_fixed_k10_node_drop_20260522.csv`
- Cora embeddings: `embeddings/cora_llm_emb.pt`
- PubMed embeddings: `embeddings/pubmed_llm_emb.pt`
- Cora raw runs: `logs/cora_single_pilot_drop0_node75_20260522_161006.csv`, `logs/cora_single_pilot_drop0_node85_20260522_161019.csv`, `logs/cora_single_pilot_drop0_node95_20260522_161033.csv`
- PubMed raw runs: `logs/pubmed_single_pilot_drop0_node75_20260522_161055.csv`, `logs/pubmed_single_pilot_drop0_node85_20260522_161116.csv`, `logs/pubmed_single_pilot_drop0_node95_20260522_161138.csv`

## Immediate Interpretation

These are pilot results under the standard small Planetoid-style training split. At high node-drop rates, the remaining supervised training set becomes extremely small: Cora has 34/21/7 train nodes, and PubMed has 15/9/3 train nodes. This makes the results much less stable than arXiv and means the experiment is dominated by low-label supervision as much as by graph repair.

Under this split, Cora shows a small repair gain only at `node_drop_rate=0.85`, while PubMed shows essentially no repair gain beyond `GCN (LLM, Sparse)`. For stronger evidence on Cora/PubMed, the next step should use a larger fixed training split before drawing conclusions about the repair mechanism.
