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

The reusable method components live in `src/cold_start.py`. The raw-text feature
entry is `encode_texts_with_transformer`, which maps text-only nodes to frozen
Transformer embeddings. Main-method callers should configure the graph pipeline
with `ColdStartPipelineConfig` and call `build_cold_start_training_state_from_config`;
the lower-level `build_cold_start_training_state` remains available for direct
parameter calls. The pipeline performs admission, pseudo-labeling, semantic edge
recovery, and train-mask construction. The script above is an experiment runner
that loads datasets, calls this pipeline, trains the GNN, and writes CSV results.

The main single-dataset method runner also supports the same cold-start pipeline:

```powershell
python scripts\utils\run_single_dataset_pilot.py --cold-start
```

When `--cold-start` is enabled, Ours/`LLM_GNN_*` models call
`build_cold_start_training_state_from_config` before training. The runner then
uses the pipeline's recovered `edge_index`, `pseudo_y`, and `train_mask` for the
supervised loss, with `--pseudo-label-loss-weight` controlling how much
pseudo-labeled cold-start nodes contribute. This is the main-method integration
path; it avoids the earlier random recovery path that reintroduced the dropped
training nodes' ground-truth labels. Non-repair baselines and runs without
`--cold-start` keep the previous behavior for backward-compatible comparisons.

To validate the raw-text-to-feature segment from the main runner, pass:

```powershell
--force-regenerate-embeddings
```

The flag is forwarded to the dataset preprocessors before admission and training,
so `run_single_dataset_pilot.py --cold-start --force-regenerate-embeddings`
exercises the full path from raw text features to cold-start admission, pseudo
labels, semantic edges, and GNN training.

By default, the script reuses cached text-derived embeddings when available.
To exercise the full raw-text-to-feature path before admission and edge recovery,
pass:

```powershell
--force-regenerate-embeddings
```

This calls the dataset preprocessing pipeline to rebuild `x_llm` from raw text
before cold-start admission. For external text-only nodes, call
`encode_texts_with_transformer` directly and then pass the resulting embeddings
to `build_cold_start_training_state`. Feature generation can be expensive on full
arXiv, so cached features remain the default for budget sweeps.

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

`--pseudo-label-loss-weight` controls the supervised-loss weight of
pseudo-labeled cold-start nodes that pass the confidence/support/agreement
filters. The default `1.0` preserves the original pseudo-supervised protocol.
Setting it to `0.0` keeps admitted nodes and recovered semantic edges in the
message-passing graph but removes pseudo labels from the supervised loss; this is
the edge-only cold-start variant.

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
- `sampled_recovered_nodes_mean`: legacy recovery column; in cold-start main-entry
  logs it is kept as a backward-compatible alias for selected cold-start nodes.
- `pseudo_train_nodes_mean`: admitted nodes that pass pseudo-label filters; their
  actual supervised-loss contribution is controlled by
  `pseudo_label_loss_weight`.
- `recovery_edges_mean`: semantic edges added for admitted nodes.
- `pseudo_label_accuracy_mean`: diagnostic only; not available to the method.
- `selected_center_distance_mean`: representativeness diagnostic for clustering.

`run_single_dataset_pilot.py --cold-start` additionally writes the same
cold-start control fields beside the existing sparse-graph pilot metrics, so the
main-method logs can be audited for admission, pseudo-label filtering, and edge
recovery counts.

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

## Main-Method Integration Runs

The following runs validate that the cold-start pipeline is reachable from the
main single-dataset runner rather than only from the standalone budget script.
All commands used `conda run -n llm-sgnn`, `--cold-start`,
`--admission-ratio 0.5`, `--node-drop-rate 0.75`, `--drop-rate 0.0`,
`--repair-policy adaptive`, `--max-edges-per-recovered-node 5`, 3 seeds, and
100 epochs.

Output files:

- `logs/cora_single_pilot_coldstart_drop0_node75_20260529_212853.csv`
- `logs/pubmed_single_pilot_coldstart_drop0_node75_20260529_212946.csv`
- `logs/wikics_single_pilot_coldstart_drop0_node75_20260529_213036.csv`
- `logs/arxiv_single_pilot_coldstart_drop0_node75_20260529_213555.csv`

| Dataset | Backbone | Accuracy | Std | Selected cold-start | Pseudo-train | Recovery edges | Pseudo-label acc. |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | GCN | 0.5550 | 0.0695 | 50.0 | 50.0 | 99.7 | 0.3800 |
| Cora | GAT | 0.5780 | 0.0914 | 50.0 | 50.0 | 99.7 | 0.3800 |
| Cora | GraphSAGE | 0.5023 | 0.0921 | 50.0 | 50.0 | 99.7 | 0.3800 |
| PubMed | GCN | 0.5423 | 0.1499 | 22.0 | 22.0 | 45.7 | 0.4091 |
| PubMed | GAT | 0.5593 | 0.0988 | 22.0 | 22.0 | 45.7 | 0.4091 |
| PubMed | GraphSAGE | 0.5310 | 0.1501 | 22.0 | 22.0 | 45.7 | 0.4091 |
| WikiCS | GCN | 0.6052 | 0.0371 | 217.0 | 217.0 | 461.3 | 0.5069 |
| WikiCS | GAT | 0.5922 | 0.0182 | 217.0 | 217.0 | 461.3 | 0.5069 |
| WikiCS | GraphSAGE | 0.5779 | 0.0247 | 217.0 | 217.0 | 461.3 | 0.5069 |
| arXiv-full | GCN | 0.5379 | 0.0033 | 34102.0 | 34102.0 | 71104.0 | 0.4699 |
| arXiv-full | GAT | 0.5358 | 0.0051 | 34102.0 | 34102.0 | 71104.0 | 0.4699 |
| arXiv-full | GraphSAGE | 0.5501 | 0.0046 | 34102.0 | 34102.0 | 71104.0 | 0.4699 |

These results are integration checks, not a replacement for the budget ablation
above. They show that the main runner can execute the full path:
cached/raw-text features -> admission -> pseudo labels -> semantic edges ->
training with pseudo labels. These early runs used the default
`--pseudo-label-loss-weight 1.0`, so they are a stress test of pseudo-supervised
training rather than the final recommended cold-start objective. The full arXiv
GraphSAGE run matches the standalone full-graph budget scale at ratio `0.50`,
which supports using the integrated runner for later main-method experiments.

### Edge-Only Main-Method Integration Runs

After adding the agreement filter and pseudo-label loss weighting, we reran the
main entry with the safer edge-only setting:

```powershell
--pseudo-label-agreement nearest_or_centroid --pseudo-label-loss-weight 0.0
```

The run keeps admitted cold-start nodes and recovered semantic edges in the
message-passing graph, but removes pseudo labels from the supervised loss.

Output files:

- `logs/cora_single_pilot_coldstart_drop0_node75_20260530_162230.csv`
- `logs/pubmed_single_pilot_coldstart_drop0_node75_20260530_162323.csv`
- `logs/wikics_single_pilot_coldstart_drop0_node75_20260530_162614.csv`
- `logs/arxiv_single_pilot_coldstart_drop0_node75_20260530_163630.csv`

Matched no-cold-start control files, using the same main entry with
`--recovery-ratio 0.0` and without `--cold-start`:

- `logs/cora_single_pilot_drop0_node75_20260530_164206.csv`
- `logs/pubmed_single_pilot_drop0_node75_20260530_164245.csv`
- `logs/wikics_single_pilot_drop0_node75_20260530_164326.csv`
- `logs/arxiv_single_pilot_drop0_node75_20260530_164516.csv`

| Dataset | Backbone | Accuracy | Std | Selected cold-start | Pseudo-train tracked | Recovery edges | Pseudo-label acc. |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | GCN | 0.7213 | 0.0170 | 50.0 | 23.7 | 99.7 | 0.6626 |
| Cora | GAT | 0.7100 | 0.0188 | 50.0 | 23.7 | 99.7 | 0.6626 |
| Cora | GraphSAGE | 0.7230 | 0.0085 | 50.0 | 23.7 | 99.7 | 0.6626 |
| PubMed | GCN | 0.6873 | 0.0202 | 22.0 | 14.3 | 45.7 | 0.5031 |
| PubMed | GAT | 0.7197 | 0.0284 | 22.0 | 14.3 | 45.7 | 0.5031 |
| PubMed | GraphSAGE | 0.6817 | 0.0458 | 22.0 | 14.3 | 45.7 | 0.5031 |
| WikiCS | GCN | 0.7010 | 0.0089 | 217.0 | 133.3 | 461.3 | 0.7407 |
| WikiCS | GAT | 0.6976 | 0.0072 | 217.0 | 133.3 | 461.3 | 0.7407 |
| WikiCS | GraphSAGE | 0.7114 | 0.0042 | 217.0 | 133.3 | 461.3 | 0.7407 |
| arXiv-full | GCN | 0.5804 | 0.0021 | 34102.0 | 22048.3 | 71104.0 | 0.6630 |
| arXiv-full | GAT | 0.5964 | 0.0024 | 34102.0 | 22048.3 | 71104.0 | 0.6630 |
| arXiv-full | GraphSAGE | 0.6070 | 0.0026 | 34102.0 | 22048.3 | 71104.0 | 0.6630 |

Compared with the default pseudo-supervised integration table above, the
edge-only objective improves every listed model and dataset. Against the matched
no-cold-start control, however, the effect is mostly neutral:

| Dataset | Backbone | No cold-start | Edge-only | Delta |
|---|---|---:|---:|---:|
| Cora | GCN | 0.7210 | 0.7213 | +0.0003 |
| Cora | GAT | 0.7247 | 0.7100 | -0.0147 |
| Cora | GraphSAGE | 0.7250 | 0.7230 | -0.0020 |
| PubMed | GCN | 0.6860 | 0.6873 | +0.0013 |
| PubMed | GAT | 0.6770 | 0.7197 | +0.0427 |
| PubMed | GraphSAGE | 0.6800 | 0.6817 | +0.0017 |
| WikiCS | GCN | 0.7005 | 0.7010 | +0.0005 |
| WikiCS | GAT | 0.6938 | 0.6976 | +0.0038 |
| WikiCS | GraphSAGE | 0.7132 | 0.7114 | -0.0018 |
| arXiv-full | GCN | 0.5792 | 0.5804 | +0.0012 |
| arXiv-full | GAT | 0.5965 | 0.5964 | -0.0001 |
| arXiv-full | GraphSAGE | 0.6108 | 0.6070 | -0.0038 |

The strongest full-arXiv backbone remains GraphSAGE. `Pseudo-train tracked`
counts nodes passing pseudo-label filters for diagnostics; with loss weight
`0.0`, they do not contribute supervised loss. These results strengthen the
positioning of cold-start as a safe appendix extension and protocol capability:
edge-only recovery removes the harm from noisy pseudo-label supervision, but it
does not yet deliver a consistent gain over simply keeping the sparse graph
without admitted cold-start nodes.

## Formal Cold-Start Controls

To decide whether cold-start should be promoted from an appendix extension to a
main contribution, we ran a stricter four-way control using the standalone
GraphSAGE path:

- `No cold-start`: `--admission-ratio 0.0`; train only on observed nodes.
- `Random admission`: `--admission-ratio 0.5 --admission-strategies random`.
- `Cluster representative`: `--admission-ratio 0.5 --admission-strategies cluster_representative`.
- `Admit all`: `--admission-ratio 1.0`; both admission strategies select the
  same candidate set, so the representative mechanism disappears.

All commands used `conda run -n llm-sgnn`, `--model-type LLM_GNN_SAGE`,
`--node-drop-rate 0.75`, `--drop-rate 0.0`, `--repair-policy adaptive`,
`--max-edges-per-recovered-node 5`, `--k-neighbors 5`, 3 seeds, and 100 epochs.
Full arXiv used `--arxiv-subgraph-size 0`.

The `No cold-start` control is still sparse. It uses the same corrupted graph as
the cold-start methods after incident edges of hidden training candidates have
been removed. With `--admission-ratio 0.0`, the pipeline admits no cold-start
node, adds no semantic recovery edge, and keeps `train_mask` equal to the
observed training mask. The script
`scripts/utils/verify_no_cold_start_control.py` checks these invariants on a toy
graph.

Output files:

- `logs/cold_start_formal_cora_sage_summary_20260530_142118.csv`
- `logs/cold_start_formal_pubmed_sage_summary_20260530_142216.csv`
- `logs/cold_start_formal_wikics_sage_summary_20260530_142300.csv`
- `logs/cold_start_formal_arxiv_full_sage_summary_20260530_143237.csv`

| Dataset | Control | Accuracy | Std | Selected cold-start | Pseudo-train | Recovery edges | Pseudo-label acc. |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | No cold-start | 0.7250 | 0.0073 | 0.0 | 0.0 | 0.0 | nan |
| Cora | Random admission | 0.4147 | 0.0782 | 50.0 | 50.0 | 102.0 | 0.2267 |
| Cora | Cluster representative | 0.5023 | 0.0921 | 50.0 | 50.0 | 99.7 | 0.3800 |
| Cora | Admit all | 0.3563 | 0.0647 | 100.0 | 100.0 | 198.3 | 0.2667 |
| PubMed | No cold-start | 0.6800 | 0.0463 | 0.0 | 0.0 | 0.0 | nan |
| PubMed | Random admission | 0.4737 | 0.0651 | 22.0 | 22.0 | 43.7 | 0.3636 |
| PubMed | Cluster representative | 0.5310 | 0.1501 | 22.0 | 22.0 | 45.7 | 0.4091 |
| PubMed | Admit all | 0.5563 | 0.0608 | 45.0 | 45.0 | 91.3 | 0.5481 |
| WikiCS | No cold-start | 0.7132 | 0.0070 | 0.0 | 0.0 | 0.0 | nan |
| WikiCS | Random admission | 0.5432 | 0.0312 | 217.0 | 217.0 | 449.3 | 0.3932 |
| WikiCS | Cluster representative | 0.5779 | 0.0247 | 217.0 | 217.0 | 461.3 | 0.5069 |
| WikiCS | Admit all | 0.5190 | 0.0376 | 435.0 | 435.0 | 898.3 | 0.4406 |
| arXiv-full | No cold-start | 0.6109 | 0.0035 | 0.0 | 0.0 | 0.0 | nan |
| arXiv-full | Random admission | 0.5081 | 0.0162 | 34102.0 | 34102.0 | 71009.7 | 0.4380 |
| arXiv-full | Cluster representative | 0.5489 | 0.0047 | 34102.0 | 34102.0 | 71104.0 | 0.4699 |
| arXiv-full | Admit all | 0.4989 | 0.0362 | 68205.0 | 68205.0 | 142105.0 | 0.4315 |

Conclusion:

- `Cluster representative` consistently improves over `Random admission` at the
  same budget, and usually improves over `Admit all`.
- It does not outperform `No cold-start` under this supervised-loss protocol.
  The likely reason is pseudo-label noise: the admitted nodes enter the
  supervised loss, but pseudo-label accuracy is only 38.0% on Cora, 40.9% on
  PubMed, 50.7% on WikiCS, and 47.0% on full arXiv.
- Therefore, the current evidence supports cold-start as a controlled appendix
  extension and implementation capability, not yet as a stronger main
  contribution. Promoting it would require a better pseudo-label reliability
  mechanism or a training objective that does not treat noisy pseudo labels as
  ordinary ground-truth supervision.

### Pseudo-Label Support Filtering

After reviewing the formal controls, we added a conservative pseudo-label
support diagnostic and optional filter:

```powershell
--min-pseudo-label-support 2
```

`pseudo_label_support` counts how many observed training labels support the
assigned pseudo label. For `cluster_majority`, this is the majority-label count
inside the mixed cluster. For `nearest_labeled`, it is the vote count among the
nearest observed labels. For `class_centroid`, it is the number of observed
training nodes in the selected class. A node enters `pseudo_train_mask` only if
its pseudo-label confidence is finite, exceeds `--pseudo-label-confidence`, and
meets `--min-pseudo-label-support`. The default is `0`, preserving previous
results.

Initial check at admission ratio `0.50`, GraphSAGE, 3 seeds, 100 epochs:

| Dataset | Min support | Accuracy | Std | Selected cold-start | Pseudo-train | Pseudo-label acc. | Support mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | 0 | 0.5023 | 0.0921 | 50.0 | 50.0 | 0.3800 | n/a |
| Cora | 2 | 0.4747 | 0.0610 | 50.0 | 43.0 | 0.3434 | 4.5102 |
| PubMed | 0 | 0.5310 | 0.1501 | 22.0 | 22.0 | 0.4091 | n/a |
| PubMed | 2 | 0.5413 | 0.1620 | 22.0 | 20.0 | 0.4413 | 3.6970 |

Support filtering helps PubMed slightly but hurts Cora in this first check, so
minimum support alone is not enough to promote cold-start to a main contribution.
It is still useful as a diagnostic and as a building block for stronger filters,
such as agreement between cluster-majority, nearest-labeled, and class-centroid
pseudo labels.

### Pseudo-Label Agreement Filtering

We also added an optional agreement filter:

```powershell
--pseudo-label-agreement nearest_or_centroid
```

The primary pseudo-label strategy remains `cluster_majority`. The agreement
filter computes auxiliary labels from `nearest_labeled` and/or `class_centroid`
and allows a selected node to enter `pseudo_train_mask` only when the primary
label agrees with the requested auxiliary strategy. Supported modes are `none`,
`nearest_labeled`, `class_centroid`, `nearest_or_centroid`, and
`nearest_and_centroid`. The default is `none`, preserving previous results.

Agreement runs at admission ratio `0.50`, GraphSAGE, 3 seeds, 100 epochs:

Output files:

- `logs/cold_start_agree_or_cora_sage_summary_20260530_145743.csv`
- `logs/cold_start_agree_or_pubmed_sage_summary_20260530_145827.csv`
- `logs/cold_start_agree_or_wikics_sage_summary_20260530_154715.csv`
- `logs/cold_start_agree_or_arxiv_full_sage_summary_20260530_155008.csv`

| Dataset | Agreement | Accuracy | Std | Selected cold-start | Agreement nodes | Pseudo-train | Pseudo-label acc. |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | none | 0.5023 | 0.0921 | 50.0 | n/a | 50.0 | 0.3800 |
| Cora | nearest_or_centroid | 0.6520 | 0.0410 | 50.0 | 23.7 | 23.7 | 0.6626 |
| PubMed | none | 0.5310 | 0.1501 | 22.0 | n/a | 22.0 | 0.4091 |
| PubMed | nearest_or_centroid | 0.6170 | 0.0746 | 22.0 | 14.3 | 14.3 | 0.5031 |
| WikiCS | none | 0.5779 | 0.0247 | 217.0 | n/a | 217.0 | 0.5069 |
| WikiCS | nearest_or_centroid | 0.6822 | 0.0187 | 217.0 | 133.3 | 133.3 | 0.7407 |
| arXiv-full | none | 0.5489 | 0.0047 | 34102.0 | n/a | 34102.0 | 0.4699 |
| arXiv-full | nearest_or_centroid | 0.6039 | 0.0033 | 34102.0 | 22048.3 | 22048.3 | 0.6630 |

Agreement filtering is more promising than minimum-support filtering in these
checks: it substantially improves pseudo-label accuracy and downstream accuracy
on all four datasets. It nearly closes the gap to the no-cold-start control on
full arXiv, but still remains below no-cold-start on the current benchmark
protocol, so it should be treated as a reliability improvement rather than
evidence to promote cold-start to the paper's main contribution.

### Pseudo-Label Loss Weighting

The agreement results suggested that pseudo-label noise, rather than admission
or edge recovery alone, was the main bottleneck. We therefore added
`--pseudo-label-loss-weight` and compared three settings under the same
agreement filter, admission ratio `0.50`, GraphSAGE, 3 seeds, and 100 epochs:

- `1.0`: pseudo labels are treated like ordinary training labels.
- `0.3`: pseudo labels contribute to the supervised loss with reduced weight.
- `0.0`: edge-only cold-start recovery; admitted nodes and semantic edges remain
  in the graph, but pseudo labels do not contribute to the supervised loss.

Output files:

- `logs/cold_start_agree_edgeonly_cora_sage_summary_20260530_160337.csv`
- `logs/cold_start_agree_edgeonly_pubmed_sage_summary_20260530_160611.csv`
- `logs/cold_start_agree_edgeonly_wikics_sage_summary_20260530_160634.csv`
- `logs/cold_start_agree_edgeonly_arxiv_full_sage_summary_20260530_161051.csv`
- `logs/cold_start_agree_w03_cora_sage_summary_20260530_161119.csv`
- `logs/cold_start_agree_w03_pubmed_sage_summary_20260530_161145.csv`
- `logs/cold_start_agree_w03_wikics_sage_summary_20260530_161210.csv`
- `logs/cold_start_agree_w03_arxiv_full_sage_summary_20260530_161438.csv`

| Dataset | No cold-start | Agreement, w=1.0 | Agreement, w=0.3 | Agreement, w=0.0 | Selected cold-start | Pseudo-train | Recovery edges | Pseudo-label acc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | 0.7250 +/- 0.0073 | 0.6520 +/- 0.0410 | 0.6737 +/- 0.0176 | 0.7230 +/- 0.0085 | 50.0 | 23.7 | 99.7 | 0.6626 |
| PubMed | 0.6800 +/- 0.0463 | 0.6170 +/- 0.0746 | 0.6247 +/- 0.0890 | 0.6817 +/- 0.0458 | 22.0 | 14.3 | 45.7 | 0.5031 |
| WikiCS | 0.7132 +/- 0.0070 | 0.6822 +/- 0.0187 | 0.7002 +/- 0.0113 | 0.7114 +/- 0.0042 | 217.0 | 133.3 | 461.3 | 0.7407 |
| arXiv-full | 0.6109 +/- 0.0035 | 0.6039 +/- 0.0033 | 0.6083 +/- 0.0027 | 0.6089 +/- 0.0028 | 34102.0 | 22048.3 | 71104.0 | 0.6630 |

The edge-only variant is the most stable setting in this check. It nearly
matches the no-cold-start control on all four datasets, while `0.3` remains
worse on Cora, PubMed, and WikiCS and does not improve full arXiv. The resulting
interpretation is narrower but cleaner: cold-start admission and semantic edge
recovery can safely add text-only nodes for message passing, but pseudo labels
should remain diagnostic by default. Any supervised use of pseudo labels should
be explicitly weighted and separately validated until a stronger reliability
mechanism consistently beats the no-cold-start control.

## Review Notes

Protocol audit:

- Raw-text feature generation, cold-start admission, pseudo-labeling, semantic
  edge recovery, and train-mask construction are implemented in
  `src/cold_start.py` so they can be reused outside the experiment runner.
- `ColdStartPipelineConfig` groups the admission, pseudo-label, and edge-recovery
  hyperparameters for main-method integration.
- `scripts/utils/run_single_dataset_pilot.py --cold-start` now routes Ours
  models through the cold-start pipeline and uses `pseudo_y` rather than the
  hidden labels of dropped cold-start nodes. The contribution of pseudo-labeled
  nodes is controlled by `--pseudo-label-loss-weight`.
- Cold-start admission uses only `x_llm` and the cold-start candidate mask.
- `x_llm` can be loaded from cache or regenerated from raw text with
  `--force-regenerate-embeddings`.
- Ground-truth labels of admitted nodes are not used for admission, pseudo-label
  assignment, or edge construction.
- Pseudo labels are inferred from observed training labels through
  `cluster_majority`, `nearest_labeled`, or `class_centroid`.
- Ground-truth labels of cold-start nodes are used only to compute
  `pseudo_label_accuracy_mean`.
- If no finite pseudo-label confidence is produced for an admitted node, that
  node is excluded from `pseudo_train_mask` even when the confidence threshold is
  `0.0`; this prevents default labels from leaking into the supervised loss.
- With `--pseudo-label-loss-weight 0.0`, nodes in `pseudo_train_mask` are still
  tracked for diagnostics, but their pseudo labels have zero supervised-loss
  weight. They can still influence message passing through recovered edges.
- `cluster_representative` and `random` share the same `admission_ratio` budget.
- Full arXiv runs use `--arxiv-subgraph-size 0`.

Table audit:

- Tables D1 and D2 in `docs/paper_draft_zh.md` were regenerated from the conda
  budget CSV files and matched the manually inserted values.
- `scripts/utils/summarize_cold_start_budget.py` can regenerate the markdown
  tables from the saved summary CSV files.

Integration smoke checks on 2026-05-29:

```powershell
conda run -n llm-sgnn python -m py_compile scripts\utils\run_single_dataset_pilot.py src\cold_start.py
conda run -n llm-sgnn python scripts\utils\run_single_dataset_pilot.py --dataset cora --model-filter ours_gcn --num-runs 1 --num-epochs 1 --drop-rate 0.1 --node-drop-rate 0.2 --cold-start --admission-ratio 0.5 --repair-policy adaptive --max-edges-per-recovered-node 5
conda run -n llm-sgnn python scripts\utils\run_single_dataset_pilot.py --dataset cora --model-filter ours_gcn --num-runs 1 --num-epochs 1 --drop-rate 0.1 --node-drop-rate 0.2 --recovery-ratio 0.5 --repair-policy adaptive --max-edges-per-recovered-node 5
conda run -n llm-sgnn python scripts\utils\run_single_dataset_pilot.py --dataset cora --model-filter ours_gcn --num-runs 1 --num-epochs 1 --drop-rate 0.1 --node-drop-rate 0.2 --cold-start --admission-ratio 0.5 --repair-policy adaptive --max-edges-per-recovered-node 5 --force-regenerate-embeddings
```

Protocol invariant check:

```powershell
conda run -n llm-sgnn python scripts\utils\verify_cold_start_protocol.py
conda run -n llm-sgnn python scripts\utils\verify_no_cold_start_control.py
```

This synthetic check changes the hidden ground-truth labels of cold-start nodes
and asserts that admission, recovered edges, pseudo-train membership, and the
pseudo labels used by the supervised loss do not change.
