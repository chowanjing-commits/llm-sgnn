# Text-Only Cold-Start Node Admission

This note records the method boundary for the cold-start extension on
`method-next`. It is separate from the original structural recovery baseline.

## Problem Setting

The original method repairs sparse structure for nodes that already belong to the
training graph. The cold-start extension simulates a harder setting: newly
admitted nodes have text only.

For a cold-start node, the method can use:

- Its raw text embedding `x_llm`.
- The sparse observed graph after node/edge dropping.
- Labels of the remaining observed training nodes.

The method cannot use:

- The cold-start node's original incident edges.
- The cold-start node's ground-truth label.
- Validation or test labels.

Ground-truth labels of cold-start nodes are used only for offline diagnostics,
such as `pseudo_label_accuracy_mean`.

## Protocol

The experiment entry point is:

```powershell
python scripts\utils\run_cold_start_recovery.py
```

Formal experiments should use the project environment:

```powershell
conda run -n llm-sgnn python scripts\utils\run_cold_start_recovery.py
```

The reusable method components live in `src/cold_start.py`. The main pipeline
entry is `build_cold_start_training_state`, which performs admission,
pseudo-labeling, semantic edge recovery, and train-mask construction. The script
above is an experiment runner that loads datasets, calls this pipeline, trains
the GNN, and writes CSV results.

By default, the script reuses cached text-derived embeddings when available.
To exercise the full raw-text-to-feature path before admission and edge recovery,
pass:

```powershell
--force-regenerate-embeddings
```

This calls the dataset preprocessing pipeline to rebuild `x_llm` from raw text
before cold-start admission. It can be expensive on full arXiv, so cached
features remain the default for budget sweeps.

Environment snapshot checked on 2026-05-28:

- Python: `C:\Users\AnjingChow\miniconda3\envs\llm-sgnn\python.exe`
- PyTorch: `2.8.0+cu129`
- CUDA available: `True`
- scikit-learn: `1.7.2`
- pandas: `2.3.3`

The protocol first drops a subset of training nodes with `--node-drop-rate`.
Those dropped training nodes become text-only cold-start candidates. Their
incident edges are removed from the sparse graph, and their labels are hidden
from the method.

The number of admitted cold-start nodes is controlled by `--admission-ratio`.
This is a new-method budget and should not be interpreted as the old structural
recovery ratio.

## Admission Strategies

`cluster_representative` is the proposed strategy. It clusters cold-start
candidate embeddings with `num_clusters = num_classes`, ranks nodes in each
cluster by distance to the cluster center, and admits representative nodes under
the same global admission budget.

`random` is the budget-matched control. It admits the same number of cold-start
nodes by random sampling.

Typical comparison:

```powershell
python scripts\utils\run_cold_start_recovery.py `
  --dataset cora `
  --num-runs 3 `
  --num-epochs 100 `
  --node-drop-rate 0.75 `
  --admission-ratio 0.5 `
  --admission-strategies cluster_representative,random `
  --model-type LLM_GNN_SAGE
```

## Pseudo Labels

Cold-start labels are pseudo labels. The default strategy is
`cluster_majority`: cluster observed labeled training nodes together with
admitted cold-start nodes, then map each admitted node's cluster to the majority
label among observed training nodes in that cluster. If a cluster has no observed
training label, class-centroid pseudo labeling is used as fallback.

Other supported strategies:

- `nearest_labeled`: kNN vote from observed labeled training nodes.
- `class_centroid`: nearest observed class centroid in text-embedding space.

`--pseudo-label-confidence` filters pseudo labels before they enter the training
loss. Nodes below the threshold may still have recovery edges, but they are not
used as labeled training examples.

## Edge Recovery

Admitted cold-start nodes are connected back to the sparse graph by semantic kNN
using text embeddings. Candidate neighbors exclude dropped nodes, so cold-start
nodes are repaired against the observed graph. Existing sparse edges are not
duplicated.

## Key Metrics

The script writes summary and raw CSV files under `logs/`.

Important fields:

- `accuracy`: downstream test accuracy.
- `admission_strategy`: `cluster_representative` or `random`.
- `admission_ratio`: cold-start admission budget.
- `cold_start_train_nodes_mean`: candidate cold-start training nodes.
- `selected_cold_start_nodes_mean`: admitted nodes before pseudo-label filtering.
- `pseudo_train_nodes_mean`: admitted nodes that enter the supervised loss.
- `recovery_edges_mean`: semantic edges added for admitted nodes.
- `pseudo_label_accuracy_mean`: diagnostic only; not available to the method.
- `selected_center_distance_mean`: representativeness diagnostic for clustering.

## Boundary Against Previous Method

Previous recovery experiments answer: if known training nodes are structurally
damaged, can semantic edges recover their local topology?

This cold-start protocol answers: if new nodes arrive with text only and no
labels or edges, can we decide which nodes to admit, infer reliable pseudo
labels, and connect them to the sparse graph?

This is still a benchmark simulation, not a real online dynamic-graph deployment.

## Sanity Runs

Cora command:

```powershell
python scripts\utils\run_cold_start_recovery.py `
  --dataset cora `
  --num-runs 3 `
  --num-epochs 100 `
  --node-drop-rate 0.75 `
  --admission-ratio 0.5 `
  --admission-strategies cluster_representative,random `
  --model-type LLM_GNN_SAGE `
  --output-prefix cold_start_cora_sanity
```

Output files:

- `logs/cold_start_cora_sanity_summary_20260528_232940.csv`
- `logs/cold_start_cora_sanity_raw_20260528_232940.csv`

Summary:

| Dataset | Admission strategy | Accuracy | Std | Selected nodes | Pseudo-train nodes | Recovery edges | Pseudo-label accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | `cluster_representative` | 0.5027 | 0.0685 | 50.0 | 50.0 | 179.3 | 0.3800 |
| Cora | `random` | 0.4023 | 0.0683 | 50.0 | 50.0 | 174.3 | 0.2267 |

This is only a small sanity check, but it supports keeping
`cluster_representative` as the default admission strategy before broader
multi-dataset runs.

PubMed command:

```powershell
python scripts\utils\run_cold_start_recovery.py `
  --dataset pubmed `
  --num-runs 3 `
  --num-epochs 100 `
  --node-drop-rate 0.75 `
  --admission-ratio 0.5 `
  --admission-strategies cluster_representative,random `
  --model-type LLM_GNN_SAGE `
  --output-prefix cold_start_pubmed_sanity
```

Output files:

- `logs/cold_start_pubmed_sanity_summary_20260528_233218.csv`
- `logs/cold_start_pubmed_sanity_raw_20260528_233218.csv`

Summary:

| Dataset | Admission strategy | Accuracy | Std | Selected nodes | Pseudo-train nodes | Recovery edges | Pseudo-label accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| PubMed | `cluster_representative` | 0.5587 | 0.1083 | 22.0 | 22.0 | 85.7 | 0.4091 |
| PubMed | `random` | 0.4640 | 0.0815 | 22.0 | 22.0 | 79.7 | 0.3636 |

WikiCS command:

```powershell
python scripts\utils\run_cold_start_recovery.py `
  --dataset wikics `
  --split-indices 0 `
  --num-runs 3 `
  --num-epochs 100 `
  --node-drop-rate 0.75 `
  --admission-ratio 0.5 `
  --admission-strategies cluster_representative,random `
  --model-type LLM_GNN_SAGE `
  --output-prefix cold_start_wikics_sanity
```

Output files:

- `logs/cold_start_wikics_sanity_summary_20260528_233532.csv`
- `logs/cold_start_wikics_sanity_raw_20260528_233532.csv`

Summary:

| Dataset | Admission strategy | Accuracy | Std | Selected nodes | Pseudo-train nodes | Recovery edges | Pseudo-label accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| WikiCS | `cluster_representative` | 0.5816 | 0.0189 | 217.0 | 217.0 | 844.0 | 0.5069 |
| WikiCS | `random` | 0.5455 | 0.0308 | 217.0 | 217.0 | 832.0 | 0.3932 |

arXiv 10k subgraph command:

```powershell
python scripts\utils\run_cold_start_recovery.py `
  --dataset arxiv `
  --arxiv-subgraph-size 10000 `
  --num-runs 3 `
  --num-epochs 100 `
  --node-drop-rate 0.75 `
  --admission-ratio 0.5 `
  --admission-strategies cluster_representative,random `
  --model-type LLM_GNN_SAGE `
  --output-prefix cold_start_arxiv10k_sanity
```

Output files:

- `logs/cold_start_arxiv10k_sanity_summary_20260528_233709.csv`
- `logs/cold_start_arxiv10k_sanity_raw_20260528_233709.csv`

Summary:

| Dataset | Admission strategy | Accuracy | Std | Selected nodes | Pseudo-train nodes | Recovery edges | Pseudo-label accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| ogbn-arxiv-subgraph-10000 | `cluster_representative` | 0.4744 | 0.0042 | 1971.0 | 1971.0 | 1971.0 | 0.4668 |
| ogbn-arxiv-subgraph-10000 | `random` | 0.4593 | 0.0021 | 1971.0 | 1971.0 | 1971.0 | 0.4111 |

## Conda Budget Runs

The budget runs below were executed with `conda run -n llm-sgnn`.
They should be preferred over the earlier sanity runs for paper tables.

Output files:

- `logs/conda_cold_start_cora_budget_summary_20260528_235539.csv`
- `logs/conda_cold_start_cora_budget_raw_20260528_235539.csv`
- `logs/conda_cold_start_pubmed_budget_summary_20260528_235624.csv`
- `logs/conda_cold_start_pubmed_budget_raw_20260528_235624.csv`
- `logs/conda_cold_start_wikics_budget_summary_20260528_235716.csv`
- `logs/conda_cold_start_wikics_budget_raw_20260528_235716.csv`
- `logs/conda_cold_start_arxiv10k_budget_summary_20260529_000157.csv`
- `logs/conda_cold_start_arxiv10k_budget_raw_20260529_000157.csv`
- `logs/conda_cold_start_arxiv_full_budget_summary_20260529_001323.csv`
- `logs/conda_cold_start_arxiv_full_budget_raw_20260529_001323.csv`

Summary accuracy by admission ratio:

| Dataset | Ratio | Cluster representative | Random | Delta |
|---|---:|---:|---:|---:|
| Cora | 0.25 | 0.5520 | 0.5617 | -0.0097 |
| Cora | 0.50 | 0.4907 | 0.3997 | 0.0910 |
| Cora | 0.75 | 0.4113 | 0.4467 | -0.0353 |
| Cora | 1.00 | 0.3647 | 0.3647 | 0.0000 |
| PubMed | 0.25 | 0.5793 | 0.5690 | 0.0103 |
| PubMed | 0.50 | 0.5337 | 0.4657 | 0.0680 |
| PubMed | 0.75 | 0.5463 | 0.5243 | 0.0220 |
| PubMed | 1.00 | 0.5470 | 0.5470 | 0.0000 |
| WikiCS | 0.25 | 0.6259 | 0.5904 | 0.0355 |
| WikiCS | 0.50 | 0.5744 | 0.5393 | 0.0351 |
| WikiCS | 0.75 | 0.5573 | 0.5570 | 0.0003 |
| WikiCS | 1.00 | 0.5233 | 0.5233 | 0.0000 |
| arXiv-10k | 0.25 | 0.5137 | 0.4902 | 0.0236 |
| arXiv-10k | 0.50 | 0.4746 | 0.4651 | 0.0095 |
| arXiv-10k | 0.75 | 0.4420 | 0.4536 | -0.0116 |
| arXiv-10k | 1.00 | 0.4509 | 0.4509 | 0.0000 |
| arXiv-full | 0.25 | 0.5658 | 0.5588 | 0.0070 |
| arXiv-full | 0.50 | 0.5482 | 0.5067 | 0.0415 |
| arXiv-full | 0.75 | 0.5062 | 0.5263 | -0.0200 |
| arXiv-full | 1.00 | 0.5017 | 0.4992 | 0.0025 |

Pseudo-label accuracy by admission ratio:

| Dataset | Ratio | Cluster representative | Random | Delta |
|---|---:|---:|---:|---:|
| Cora | 0.25 | 0.2933 | 0.3733 | -0.0800 |
| Cora | 0.50 | 0.3800 | 0.2267 | 0.1533 |
| Cora | 0.75 | 0.3378 | 0.3067 | 0.0311 |
| Cora | 1.00 | 0.2667 | 0.2667 | 0.0000 |
| PubMed | 0.25 | 0.4242 | 0.3333 | 0.0909 |
| PubMed | 0.50 | 0.4091 | 0.3636 | 0.0455 |
| PubMed | 0.75 | 0.4343 | 0.4545 | -0.0202 |
| PubMed | 1.00 | 0.5481 | 0.5481 | 0.0000 |
| WikiCS | 0.25 | 0.5093 | 0.3333 | 0.1759 |
| WikiCS | 0.50 | 0.5069 | 0.3932 | 0.1137 |
| WikiCS | 0.75 | 0.4571 | 0.4489 | 0.0082 |
| WikiCS | 1.00 | 0.4406 | 0.4406 | 0.0000 |
| arXiv-10k | 0.25 | 0.4948 | 0.4112 | 0.0836 |
| arXiv-10k | 0.50 | 0.4668 | 0.4111 | 0.0556 |
| arXiv-10k | 0.75 | 0.4401 | 0.4285 | 0.0116 |
| arXiv-10k | 1.00 | 0.4109 | 0.4109 | 0.0000 |
| arXiv-full | 0.25 | 0.4777 | 0.4324 | 0.0452 |
| arXiv-full | 0.50 | 0.4699 | 0.4380 | 0.0319 |
| arXiv-full | 0.75 | 0.4578 | 0.4400 | 0.0179 |
| arXiv-full | 1.00 | 0.4315 | 0.4315 | 0.0000 |

Interpretation:

- Moderate admission budgets are the meaningful regime. At `0.25` and `0.50`,
  cluster-representative admission usually improves pseudo-label quality and
  downstream accuracy, with the clearest full-arXiv gain at `0.50`.
- High budgets weaken the advantage. At `0.75`, noisy pseudo labels and semantic
  edges can offset the benefit of representative selection.
- At `1.00`, both strategies admit every cold-start candidate, so the selection
  mechanism disappears and the two strategies should be treated as admission
  equivalent. Small downstream accuracy differences can still appear from later
  stochastic training.

## Review Notes

Protocol audit:

- Cold-start admission, pseudo-labeling, semantic edge recovery, and train-mask
  construction are implemented in `src/cold_start.py` so they can be reused
  outside the experiment runner.
- Cold-start admission uses only `x_llm` and the cold-start candidate mask.
- `x_llm` can be loaded from cache or regenerated from raw text with
  `--force-regenerate-embeddings`.
- Ground-truth labels of admitted nodes are not used for admission, pseudo-label
  assignment, or edge construction.
- Pseudo labels are inferred from observed training labels through
  `cluster_majority`, `nearest_labeled`, or `class_centroid`.
- Ground-truth labels of cold-start nodes are used only to compute
  `pseudo_label_accuracy_mean`.
- `cluster_representative` and `random` share the same `admission_ratio` budget.
- Full arXiv runs use `--arxiv-subgraph-size 0`.

Table audit:

- Tables D1 and D2 in `docs/paper_draft_zh.md` were regenerated from the conda
  budget CSV files and matched the manually inserted values.
- `scripts/utils/summarize_cold_start_budget.py` can regenerate the markdown
  tables from the saved summary CSV files.
