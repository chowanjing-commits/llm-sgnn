# Adaptive Semantic Pseudo-Neighborhood Repair for Text-Attributed Sparse Graphs

Draft date: 2026-05-22

Status: working draft. This document summarizes the current research framing, method, and experiments. It intentionally avoids unverified citations; add verified related work before submission.

## Abstract

Graph neural networks rely on observed topology to propagate label information, but real text-attributed graphs often contain newly appearing, partially observed, or structurally isolated nodes. In this setting, treating missing nodes as merely edge-dropped nodes is insufficient: the node itself is absent from the observed training process, and its incident edges are unavailable. We study a training-node missingness setting in which a subset of training nodes and their incident edges are removed from the observed graph, while the raw text of selected missing nodes remains available. The proposed LLM-GNN Repair framework uses text-derived language-model embeddings to recover a controlled subset of missing training nodes and construct semantic pseudo-neighborhoods around them. Recovered nodes are included in supervised training only if they are successfully connected.

Across ogbn-arxiv, Cora, and PubMed 3-seed experiments, adaptive semantic pseudo-neighborhood repair improves over sparse GCN, GAT, and GraphSAGE message-passing baselines under severe node-drop settings. The repair step is a preprocessing module and can be followed by different downstream GNN backbones. Recovery-quality diagnostics show that only a subset of added semantic edges exactly match original graph edges, so the method should not be interpreted as high-precision topology reconstruction. Instead, it provides controlled semantic neighborhoods that improve classification under node-level sparsity. We further include a label-conditioned synthetic-text ablation showing that generated class-level text can partially mitigate missing text in some settings, but remains limited by generation quality and does not replace node-specific raw text.

## 1. Introduction

Text-attributed graphs appear in citation networks, document graphs, web graphs, and knowledge-intensive recommendation systems. In many deployments, the observed topology is incomplete. New papers, documents, or web pages may arrive with text but without reliable citation or interaction edges. Existing graph learning pipelines usually assume a fixed node set and only simulate sparsity by randomly dropping edges. This setting misses an important failure mode: the graph may lose nodes and all their incident edges.

This work studies a stricter setting. During corruption, a subset of training nodes is removed from the observed graph, and all incident edges are deleted. Baseline models train only on the remaining observed training nodes. The proposed method attempts to recover a controlled subset of the missing training nodes using their raw text: it computes language-model embeddings, connects recovered nodes to semantically similar observed nodes, and includes only successfully recovered nodes in the supervised loss.

The central hypothesis is that language-model embeddings provide a useful semantic prior for pseudo-neighborhood repair. However, the prior should be used conservatively. Adding too many semantic edges can turn repair into uncontrolled graph densification, while too strict a threshold may fail to recover useful nodes. The current experiments therefore examine both accuracy and repair statistics: observed training nodes, successfully recovered nodes, added recovery edges, and recovery-edge hits against the original graph.

## 2. Problem Setup

Let the original text-attributed graph be:

\[
G = (V, E, X, T, Y),
\]

where \(V\) is the node set, \(E\) is the observed topology, \(X\) denotes original node features, \(T\) denotes raw text, and \(Y\) denotes labels for supervised nodes.

We simulate node-level sparsity by selecting a subset of training nodes \(V_{drop} \subset V_{train}\). These nodes are removed from the observed training graph, and every edge incident to them is deleted:

\[
E_{sparse} = \{(u,v) \in E: u \notin V_{drop}, v \notin V_{drop}\}.
\]

Baselines train on:

\[
V_{train}^{obs} = V_{train} \setminus V_{drop}.
\]

The repair method samples a subset of dropped training nodes:

\[
V_{rec}^{sampled} \subseteq V_{drop},
\]

and attempts to reconnect them using text-derived embeddings. A recovered node enters the supervised loss only if at least one semantic edge is successfully added:

\[
V_{train}^{ours} = V_{train}^{obs} \cup V_{rec}^{success}.
\]

This loss mask is important. The method does not use the full original training set unless all dropped nodes are explicitly selected and successfully recovered.

This is a training-node missingness simulation rather than an unlabeled cold-start prediction task. Labels of dropped training nodes are part of the original training supervision, but baselines cannot use them because those nodes are absent after corruption. The proposed method uses a dropped training node's label only after that node is explicitly selected for recovery and successfully connected back into the graph.

### 2.1 Label-Conditioned Synthetic-Text Diagnostic

The main setting assumes that a recovered missing node has access to its node-specific raw text. We also study a stricter diagnostic setting in which this raw text is unavailable. In that case, we ask whether class-conditioned synthetic paper text can provide a partial semantic prior for repair.

This diagnostic is label-informed by construction: synthetic text is generated from the ground-truth class of a dropped training node. It is therefore an upper-bound ablation, not a deployable cold-start method. Its purpose is to test whether class-level semantic text can mitigate missing-sample effects when node-specific text is absent, and to quantify how far such synthetic priors remain from true raw text.

## 3. Method: LLM-GNN Repair

### 3.1 Text-Derived Node Embeddings

For each text-available node, we compute or load a cached language-model embedding from its raw text. In the current experiments, Cora and PubMed use title-plus-abstract text when available, while ogbn-arxiv uses real paper titles from the OGB `titleabs.tsv` file; the arXiv abstract column is not used in the current cached embeddings. These embeddings are used in two places:

- As semantic node features for LLM-based baselines and the proposed method.
- As the similarity space for recovering missing nodes.

### 3.2 Controlled Node Recovery

For a dropped training node \(u\), the repair module compares its language-model embedding against candidate nodes that remain in the observed graph and have valid text embeddings. Current experiments use cosine similarity.

The current recovery rule is:

1. Select the top-\(k\) semantic candidates for each sampled recovered node.
2. Keep only candidates whose cosine similarity exceeds a threshold \(\tau\).
3. Add at most \(b\) undirected recovery edges per recovered node.
4. Mark a recovered node as successful only if at least one edge is added.

Here \(k\), \(\tau\), and \(b\) control different aspects of repair:

- \(k\) controls the semantic candidate pool size.
- \(\tau\) controls semantic confidence.
- \(b\) controls graph densification per recovered node.

### 3.3 Adaptive Repair Policy

A global cosine threshold is not well calibrated across datasets. The current adaptive variant therefore determines the repair threshold and edge budget from the corrupted graph and each recovered node's local similarity distribution.

For the corrupted graph, define the observed average degree as:

\[
d_{obs} = \frac{|E_{sparse}|}{|V_{obs}|}.
\]

In the implementation, \(|E_{sparse}|\) counts `edge_index` entries. For undirected graphs stored with both directions, this corresponds to the usual average degree. By contrast, the recovery-edge counts reported in diagnostics are deduplicated undirected pairs, so their unit is not the same as raw `edge_index` entries.

The maximum recovery budget is:

\[
b = \min(k,\ b_{max},\ \lceil d_{obs} \rceil).
\]

This degree-aware budget is important when many missing nodes are recovered. A fixed top-\(k\) rule would add roughly the same number of semantic edges for every recovered node, which can quickly turn repair into uncontrolled graph densification. By tying \(b\) to the observed average degree of the corrupted graph, sparse observed graphs induce smaller repair budgets, while denser observed graphs allow more semantic edges. Thus the repaired graph grows in a controlled way rather than becoming a full semantic kNN graph.

For each recovered node \(u\), let \(S_u\) be the similarity scores of its top-\(k\) semantic candidates. The node-wise threshold is:

\[
\tau_u = \mu(S_u) + \alpha \sigma(S_u).
\]

The method then keeps candidates whose similarity is at least \(\tau_u\), sorted by similarity, and adds at most \(b\) undirected edges. This rule avoids validation-grid selection of a global threshold and ties repair density to the sparsity of the observed graph.

### 3.4 Training and Evaluation

All methods are trained on the corrupted graph setting. Baselines do not see dropped training nodes in the loss. The proposed method sees observed training nodes plus successfully recovered training nodes. Validation and test masks follow the original dataset split and are used only for model selection and final evaluation.

The reported accuracy is test accuracy at the epoch with best validation accuracy.

## 4. Experimental Setup

### 4.1 Datasets

Current experiments use:

| Dataset | Nodes | Edges in edge_index | Split | Text source |
|---|---:|---:|---|---|
| ogbn-arxiv | 169,343 | 1,166,243 | Official OGB split | Real paper title embeddings from OGB `titleabs.tsv`; abstract not used in current cache |
| Cora | 2,708 | 10,556 | Planetoid-style public split | Title + Abstract |
| PubMed | 19,717 | 88,648 | Planetoid-style public split | Title + Abstract |

Note: Cora and PubMed have small public training splits: 140 training nodes for Cora and 60 for PubMed. Under high node-drop rates, the observed training set becomes extremely small.

### 4.2 Baselines

The main experiments compare:

| Model | Description |
|---|---|
| MLP (Raw) | MLP using original node features, no graph propagation |
| MLP (LLM) | MLP using language-model embeddings, no graph propagation |
| GCN (Raw, Sparse) | GCN using original features on the corrupted sparse graph |
| GCN (LLM, Sparse) | GCN using language-model embeddings on the corrupted sparse graph |
| GAT (Raw, Sparse) | GAT using original features on the corrupted sparse graph |
| GAT (LLM, Sparse) | GAT using language-model embeddings on the corrupted sparse graph |
| GraphSAGE (Raw, Sparse) | GraphSAGE using original features on the corrupted sparse graph |
| GraphSAGE (LLM, Sparse) | GraphSAGE using language-model embeddings on the corrupted sparse graph |
| Ours-GCN | Adaptive semantic repair followed by GCN on the repaired graph |
| Ours-GAT | Adaptive semantic repair followed by GAT on the repaired graph |
| Ours-GraphSAGE | Adaptive semantic repair followed by GraphSAGE on the repaired graph |

### 4.3 Corruption Protocol

The current main node-drop rates are:

\[
0.75,\ 0.85,\ 0.95.
\]

The main protocol uses adaptive repair with:

- Recovery ratio: 0.5 of dropped training nodes.
- Candidate \(k\): 10.
- Maximum cap \(b_{max}\): 10.
- Effective cap: \(\min(k, b_{max}, \lceil d_{obs} \rceil)\).
- Node-wise threshold: \(\tau_u = \mu(S_u) + \alpha\sigma(S_u)\).
- \(\alpha\): 0.0.
- Seeds: 3.
- Epochs: 300.

Earlier fixed-threshold Cora/PubMed sweeps and fixed-k arXiv results are retained only as sensitivity analysis or preliminary evidence. They should be reported in an appendix or historical experiment log, not mixed into the final main table.

## 5. Main Results

Under the adaptive random-recovery protocol, the 3-seed mean/std results are shown below. Bold marks the best result in each dataset/drop setting.

| Dataset | Node drop | MLP Raw | MLP LLM | GCN Raw Sparse | GCN LLM Sparse | GAT Raw Sparse | GAT LLM Sparse | SAGE Raw Sparse | SAGE LLM Sparse | Ours-GCN | Ours-GAT | Ours-SAGE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CORA | 0.75 | 43.90 ± 2.76 | 58.33 ± 0.84 | 73.20 ± 1.55 | 72.50 ± 1.77 | 71.13 ± 2.35 | 73.07 ± 2.87 | 66.50 ± 0.86 | 70.50 ± 1.67 | **77.30 ± 1.93** | 76.97 ± 2.19 | 74.60 ± 2.94 |
| CORA | 0.85 | 36.67 ± 0.45 | 51.63 ± 2.64 | 67.27 ± 2.04 | 67.93 ± 3.77 | 68.50 ± 2.75 | 68.10 ± 1.56 | 60.10 ± 0.54 | 65.07 ± 1.39 | **76.53 ± 1.70** | 74.30 ± 0.65 | 73.00 ± 2.10 |
| CORA | 0.95 | 26.37 ± 2.25 | 40.07 ± 6.14 | 53.60 ± 7.30 | 56.53 ± 3.73 | 55.77 ± 9.10 | 54.07 ± 9.39 | 43.27 ± 5.22 | 54.50 ± 5.28 | **75.63 ± 0.60** | 75.37 ± 1.31 | 75.20 ± 1.67 |
| PubMed | 0.75 | 53.63 ± 1.37 | 62.33 ± 3.23 | 61.80 ± 2.30 | 68.70 ± 1.92 | 65.07 ± 2.27 | 70.90 ± 5.75 | 60.87 ± 0.62 | 67.60 ± 3.08 | 78.30 ± 1.49 | 76.50 ± 1.51 | **79.37 ± 1.58** |
| PubMed | 0.85 | 47.90 ± 3.44 | 53.17 ± 0.93 | 51.00 ± 1.65 | 52.07 ± 3.40 | 58.47 ± 0.48 | 55.10 ± 4.40 | 49.73 ± 2.59 | 57.10 ± 3.86 | 75.67 ± 3.30 | **77.90 ± 3.02** | 76.57 ± 4.46 |
| PubMed | 0.95 | 42.73 ± 5.57 | 46.27 ± 7.69 | 46.03 ± 9.42 | 46.17 ± 10.15 | 46.97 ± 10.53 | 49.90 ± 11.04 | 46.17 ± 10.84 | 45.57 ± 10.14 | **78.07 ± 2.88** | 74.50 ± 3.39 | 77.37 ± 1.81 |
| ogbn-arxiv | 0.75 | 52.58 ± 0.23 | 59.55 ± 0.40 | 52.42 ± 0.13 | 58.29 ± 0.14 | 54.02 ± 0.31 | 59.82 ± 0.47 | 54.22 ± 0.18 | 60.09 ± 0.37 | 60.58 ± 0.30 | 60.12 ± 0.21 | **62.40 ± 0.35** |
| ogbn-arxiv | 0.85 | 52.38 ± 0.12 | 58.49 ± 0.38 | 52.10 ± 0.26 | 56.40 ± 0.13 | 53.57 ± 0.47 | 58.70 ± 0.36 | 53.44 ± 0.14 | 58.95 ± 0.31 | 60.43 ± 0.16 | 59.74 ± 0.12 | **62.54 ± 0.21** |
| ogbn-arxiv | 0.95 | 50.41 ± 0.27 | 55.94 ± 0.40 | 50.29 ± 0.81 | 55.87 ± 0.94 | 52.21 ± 0.51 | 56.93 ± 0.89 | 50.87 ± 0.33 | 56.74 ± 0.19 | 60.30 ± 0.14 | 59.30 ± 0.08 | **61.83 ± 0.19** |

The table separates the repair module from the downstream GNN backbone. On Cora, all repaired variants outperform their corresponding sparse backbones, with the largest gains at node drop 0.95. On PubMed, repaired GCN, GAT, and GraphSAGE are all far stronger than sparse message passing in the high-drop regime. On ogbn-arxiv, `Ours-GraphSAGE` gives the strongest result across all drop rates, indicating that semantic repair is not tied to GCN and can benefit stronger inductive aggregation backbones.

A plausible explanation is that GraphSAGE's neighborhood aggregation is better suited to large-scale sparse graphs and local inductive-style aggregation, whereas GAT can be harder to optimize efficiently at this scale and GCN may be more sensitive to the repaired graph's degree distribution. This should be interpreted as an empirical observation rather than a general claim that GraphSAGE is always the best backbone. It also reveals a limitation: repair quality and downstream backbone choice are coupled. The repair step does not independently guarantee better performance; its benefit depends on how the downstream GNN uses the recovered semantic neighborhoods.

As a one-seed sensitivity analysis, an arXiv alpha sweep shows that increasing \(\alpha\) from 0.0 to 0.5 reduces recovery edges substantially while preserving all sampled recovered nodes. Accuracy changes only modestly: 60.90 to 60.41 at node drop 0.75, 60.39 to 60.63 at node drop 0.85, and 60.47 to 60.30 at node drop 0.95. This suggests that the adaptive threshold can control graph densification without collapsing node recovery.

## 6. Recovery Quality Diagnostics

To test whether semantic recovery corresponds to original topology, we compare added recovery edges against the original pre-drop graph. An edge hit means a semantic recovery edge matches an original graph edge that was unavailable in the corrupted graph. Recovery-edge counts in this table are deduplicated undirected pairs.

| Dataset | Node drop | Recovery edges | Edge hit rate | Node hit rate | Top-k precision | Top-k recall |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.75 | 174.00 ± 3.61 | 17.83 ± 1.29 | 46.00 ± 2.00 | 9.40 ± 0.40 | 25.14 ± 5.02 |
| Cora | 0.85 | 199.33 ± 6.51 | 16.07 ± 1.51 | 41.67 ± 4.49 | 9.11 ± 0.31 | 21.72 ± 1.65 |
| Cora | 0.95 | 215.67 ± 3.21 | 14.82 ± 3.67 | 38.10 ± 7.94 | 7.46 ± 1.80 | 20.29 ± 1.67 |
| PubMed | 0.75 | 79.67 ± 4.51 | 8.69 ± 2.81 | 25.76 ± 9.46 | 3.79 ± 1.14 | 7.81 ± 1.09 |
| PubMed | 0.85 | 99.00 ± 3.00 | 5.39 ± 1.18 | 18.67 ± 6.11 | 3.07 ± 0.92 | 6.63 ± 1.50 |
| PubMed | 0.95 | 106.67 ± 1.53 | 8.74 ± 1.31 | 29.76 ± 2.06 | 4.05 ± 0.90 | 10.13 ± 1.69 |
| ogbn-arxiv | 0.75 | 122,882.67 ± 145.03 | 8.32 ± 0.01 | 22.66 ± 0.05 | 5.10 ± 0.05 | 5.94 ± 0.09 |
| ogbn-arxiv | 0.85 | 139,396.33 ± 6.11 | 7.38 ± 0.12 | 20.15 ± 0.31 | 4.51 ± 0.06 | 5.89 ± 0.34 |
| ogbn-arxiv | 0.95 | 126,533.67 ± 54.50 | 6.60 ± 0.13 | 15.30 ± 0.16 | 3.80 ± 0.05 | 5.70 ± 0.35 |

The diagnostic shows that semantic recovery partially reconstructs original topology, but it is not high-precision edge reconstruction. Cora shows the strongest signal: about 18% of recovery edges match original neighbors, and 38-46% of recovered nodes obtain at least one true original-neighbor edge. PubMed and arXiv have weaker exact-edge hit rates, even though adaptive repair still improves classification in several high-drop settings.

This changes the safest paper framing. The method should not claim to perfectly reconstruct missing citation edges. Instead, it creates semantic pseudo-neighborhoods that are useful for classification, with a measurable subset of edges corresponding to true missing topology.

## 7. Sensitivity Analysis: Why Repair Should Be Adaptive

The threshold and edge-cap sensitivity experiments are useful for the paper because they show that semantic repair is not a single fixed hyperparameter trick. The best repair policy depends on dataset properties.

### 7.1 Threshold Effects

For Cora, thresholds 0.6 and 0.7 are effectively equivalent. Threshold 0.8 starts to filter a small number of candidates. Threshold 0.85 and 0.90 substantially reduce successful recovered nodes, but can improve accuracy at high node-drop rates. This suggests that stricter semantic filtering can remove noisy recovered nodes or noisy edges.

For PubMed, thresholds up to 0.90 still recover all sampled nodes. The threshold mostly reduces edge count rather than recovered-node count. This indicates that PubMed's embedding similarity distribution is different from Cora's: a global threshold has different selectivity across datasets.

### 7.2 Edge-Cap Effects

Adding 5 edges per recovered node is not large relative to total graph size, but it can be large relative to local graph degree. Cora has an average edge-index degree of about 3.90, while PubMed has about 4.50. A fixed cap of 5 therefore creates recovered nodes with relatively high local degree.

Lowering the cap from 5 to 3 cuts recovery edges substantially. It sometimes improves accuracy, especially in settings where excessive semantic edges may introduce noise. However, it is not uniformly best. PubMed at node drop 0.95 performs better with more recovery edges in some settings.

### 7.3 Current Best Settings in One-Seed Grid

| Dataset | Node drop | Best observed setting | Repaired setting |
|---|---:|---|---:|
| Cora | 0.75 | threshold 0.6/0.7, cap 5 | 79.80 |
| Cora | 0.85 | threshold 0.90, cap 5 | 78.20 |
| Cora | 0.95 | threshold 0.90, cap 5 | 77.00 |
| PubMed | 0.75 | threshold 0.90, cap 3 | 78.50 |
| PubMed | 0.85 | threshold 0.85, cap 5 | 72.90 |
| PubMed | 0.95 | threshold 0.6/0.7/0.8/0.85, cap 5 | 79.20 |

These results suggest that repair should be adjusted according to:

- Dataset size: global edge additions are negligible in large graphs but more visible in small graphs.
- Graph density and local degree: a fixed edge cap can over-connect recovered nodes in low-degree graphs.
- Semantic similarity distribution: the same cosine threshold can be strict on Cora but permissive on PubMed.
- Node-drop rate: higher drop rates reduce observed supervision and may require stricter filtering or different recovery budgets.
- Training split size: small public splits make results sensitive to which training nodes are dropped and recovered.

## 8. Label-Conditioned Synthetic Text Recovery Ablation

To probe whether class-level synthetic text can mitigate missing-sample effects, we ran an upper-bound ablation in which recovered dropped training nodes were assigned embeddings from class-conditioned synthetic paper texts generated by Qwen3-4B-Instruct-2507. This setting is label-informed by construction and therefore is not a deployable cold-start method. The goal is not to replace true node text, but to test whether lightweight class-level text generation can provide a partial semantic prior when node-specific raw text is unavailable.

For Cora and PubMed, the length-matched prompt asked the model to generate title-plus-abstract style paper records for a given class, with dataset-name filtering and diversity checks. For ogbn-arxiv, the current embedding cache uses paper titles only, so the synthetic ablation also generates title-length class-conditioned text. Synthetic text is generated per class, then one synthetic embedding is sampled from the matching class pool for each recovered node. Only recovered dropped training nodes are replaced; all other node embeddings remain unchanged.

| Dataset | Node drop | True-text Ours | Synthetic-label-text Ours | Gap | Successful recovered | Synthetic recovery edges |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 0.75 | 77.37 ± 1.95 | 70.80 ± 1.96 | -6.57 | 50.0 | 178.00 |
| Cora | 0.85 | 76.53 ± 1.70 | 67.77 ± 0.90 | -8.77 | 56.0 | 201.67 |
| Cora | 0.95 | 75.60 ± 0.57 | 62.53 ± 1.89 | -13.07 | 63.0 | 227.00 |
| PubMed | 0.75 | 78.30 ± 1.49 | 75.57 ± 1.94 | -2.73 | 22.0 | 88.33 |
| PubMed | 0.85 | 75.77 ± 3.44 | 69.47 ± 5.28 | -6.30 | 25.0 | 101.33 |
| PubMed | 0.95 | 78.07 ± 2.88 | 70.90 ± 2.20 | -7.17 | 28.0 | 116.00 |
| ogbn-arxiv | 0.75 | 60.43 ± 0.17 | 57.66 ± 0.28 | -2.77 | 34,102.0 | 126,253.33 |
| ogbn-arxiv | 0.85 | 60.23 ± 0.25 | 55.87 ± 0.26 | -4.35 | 38,649.0 | 141,793.67 |
| ogbn-arxiv | 0.95 | 60.18 ± 0.25 | 52.68 ± 1.13 | -7.51 | 43,196.0 | 127,985.33 |

This result shows that class-conditioned synthetic text can act as a partial recovery prior: it improves over sparse raw baselines in several Cora and PubMed settings, but consistently remains below true-text repair. On ogbn-arxiv, synthetic-label-text is also weaker than a stronger `GAT (LLM, Sparse)` baseline at all drop rates, while true-text repair remains stronger than the sparse message-passing baselines.

The generation-quality diagnostics explain this limitation. Cora synthetic text passes the quality filter at 91.7%, PubMed at 66.7%, and ogbn-arxiv at 57.5%. The arXiv condition is especially constrained because the current cache uses only short paper titles; the generated title-level texts have a mean length of 8.2 words and higher within-class lexical overlap. Therefore, this ablation supports a conservative deployment interpretation: class-conditioned generation may be useful for few-shot or lightweight missing-sample mitigation, but current generated text quality is insufficient to replace node-specific raw text.

For arXiv, the synthetic-text ablation also included GAT sparse baselines:

| Node drop | GCN LLM Sparse | GAT LLM Sparse | True-text Ours | Synthetic-label-text Ours |
|---:|---:|---:|---:|---:|
| 0.75 | 58.47 ± 0.11 | 59.84 ± 0.45 | 60.43 ± 0.17 | 57.66 ± 0.28 |
| 0.85 | 56.53 ± 0.07 | 58.67 ± 0.35 | 60.23 ± 0.25 | 55.87 ± 0.26 |
| 0.95 | 55.87 ± 0.94 | 56.91 ± 0.90 | 60.18 ± 0.25 | 52.68 ± 1.13 |

## 9. Discussion

The current evidence supports the value of adaptive semantic pseudo-neighborhood repair, but it also shows that repair parameters must be treated as part of the method rather than incidental implementation details. A fixed top-k or fixed threshold may not transfer across datasets because language-model embedding similarity is not calibrated across corpora.

The repair objective also differs from global semantic kNN construction. A global kNN graph adds semantic edges for all nodes, regardless of whether they were corrupted, and therefore adds \(O(|V|k)\) edges. This changes the intervention from missing-node repair to full-graph semantic densification. Clustered kNN reduces the candidate range by restricting search within clusters, but it introduces an additional clustering assumption: if a useful neighbor falls outside the assigned cluster, the method cannot select it. Our adaptive repair instead adds edges only around sampled recovered nodes, with at most \(O(|V_{rec}|b)\) added edges, where \(b\) is tied to the corrupted graph's observed degree. This avoids global densification without imposing hard cluster boundaries.

A more principled final method could replace fixed thresholds with adaptive policies:

- Node-wise similarity thresholds based on each recovered node's top-k similarity distribution.
- Degree-aware edge caps based on the observed sparse graph's average degree.
- Drop-rate-aware recovery budgets that increase only when the observed training set becomes too small.
- Percentile thresholding within each dataset or mini-batch of recovered nodes.

The current sensitivity analysis is retained as an ablation because it explains why a universal recovery threshold is not appropriate, while the main protocol is now standardized across datasets and reported over multiple seeds.

## 10. Limitations and Next Experiments

Current limitations:

- Main results are now reported over 3 seeds; sensitivity and alpha-sweep analyses are still one-seed ablations.
- Cora and PubMed have very small Planetoid training splits.
- The current ogbn-arxiv LLM cache uses real paper titles only. The arXiv synthetic-text ablation is therefore a title-level comparison; title-plus-abstract true-text and synthetic-text embeddings remain a useful follow-up.
- Related work now uses verified source links, but final BibTeX entries have not
  yet been generated.
- Thresholds are raw cosine thresholds and may not be calibrated across datasets.
- Recovery edge hit rates are moderate to low, so claims should emphasize semantic pseudo-neighborhood repair rather than exact topology reconstruction.

Next experiments:

- Extend the current 3-seed main results to 5 seeds and/or additional splits if variance remains a concern.
- Regenerate ogbn-arxiv embeddings with title plus abstract and rerun the true-text and synthetic-text comparisons to test whether richer text changes the arXiv conclusion.
- Add a percentile-threshold variant, for example keeping candidates above the 90th or 95th percentile of recovered-node similarities.
- Add degree-aware caps, for example \(b=\min(3, \lceil \bar{d}_{obs} \rceil)\) or class/dataset-specific caps based on observed degree statistics.

## 11. Related Work

Graph neural networks for semi-supervised node classification propagate features
and labels over the observed graph. The main backbones in this draft follow GCN
([Kipf and Welling, 2017](https://arxiv.org/abs/1609.02907)), GraphSAGE
([Hamilton et al., 2017](https://arxiv.org/abs/1706.02216)), and GAT
([Velickovic et al., 2018](https://arxiv.org/abs/1710.10903)). The Cora and
PubMed splits are commonly associated with Planetoid
([Yang et al., 2016](https://arxiv.org/abs/1603.08861)); ogbn-arxiv comes from
the [Open Graph Benchmark](https://arxiv.org/abs/2005.00687); WikiCS is from
[Mernyei and Cangea, 2020](https://arxiv.org/abs/2007.02901). This paper does
not propose a new message-passing layer. It studies how language-model
representations can provide controlled semantic neighborhoods when training
nodes and their incident edges are missing.

Graph structure learning and topology augmentation methods directly learn or
revise graph connectivity, including LDS-GNN
([Franceschi et al., 2019](https://arxiv.org/abs/1903.11960)), IDGL
([Chen et al., 2020](https://arxiv.org/abs/2006.13009)), and Pro-GNN
([Jin et al., 2020](https://arxiv.org/abs/2005.10203)). Our intervention is
narrower: we do not rewrite the whole graph or claim exact topology
reconstruction, but add budgeted semantic edges only around sampled missing
training nodes and report recovery-quality diagnostics.

Text-attributed graph learning increasingly combines pretrained language models
with graph neural networks. GLEM
([Zhao et al., 2022](https://arxiv.org/abs/2210.14709)) jointly leverages
language models and GNNs on large text-attributed graphs, while Patton
([Jin et al., 2023](https://aclanthology.org/2023.acl-long.387/)) pretrains
language models on text-rich networks. This draft uses frozen language-model
embeddings as a repair signal rather than training a new text-graph foundation
model.

Pseudo-labeling and self-training are relevant to the cold-start appendix.
M3S ([Sun et al., 2019](https://arxiv.org/abs/1902.11038)) uses multi-stage
self-supervised training to expand supervision on graphs. The cold-start
extension in this draft follows a conservative version of that idea: admitted
text-only nodes may be connected for message passing, but only nodes that pass
confidence, support, or agreement gates enter pseudo-label supervision; the
others remain unlabeled.

Learning with incomplete neighborhoods or cold-start nodes is another adjacent
line. Cold Brew ([Zheng et al., 2021](https://arxiv.org/abs/2111.04840))
distills node representations for incomplete or missing neighborhoods. Our
appendix D simulates text-only candidate admission, pseudo-label diagnostics,
and controlled edge recovery; it remains a boundary analysis rather than the
paper's main contribution.

The links above are verified source links for the draft stage. Before converting
to LaTeX, generate BibTeX entries from arXiv, ACL Anthology, DBLP, or official
benchmark pages rather than hand-writing them.

## 12. Current Artifact Links

- arXiv fixed-k node-drop record: `docs/arxiv_fixed_k10_node_drop.md`
- arXiv adaptive recovery record: `docs/arxiv_adaptive_recovery_policy.md`
- Main multi-seed adaptive CSV: `logs/main_multiseed_adaptive_all_models_20260522.csv`
- Main multi-seed GAT baselines CSV: `logs/main_multiseed_gat_baselines_20260523.csv`
- Main multi-seed GraphSAGE baselines CSV: `logs/main_multiseed_sage_baselines_20260524.csv`
- Main multi-seed Ours backbone variants CSV: `logs/main_multiseed_ours_backbones_20260524.csv`
- Main multi-seed all-backbone combined CSV: `logs/main_multiseed_all_backbones_20260524.csv`
- Main multi-seed all-backbone wide table CSV: `logs/main_multiseed_all_backbones_wide_20260524.csv`
- arXiv adaptive alpha sweep CSV: `logs/arxiv_adaptive_alpha_sweep_20260522.csv`
- Cora/PubMed thresholded random recovery record: `docs/citation_random_recovery_threshold.md`
- Cora/PubMed sensitivity CSV: `logs/citation_random_recovery_threshold_sensitivity_20260522.csv`
- Cora/PubMed main random recovery CSV: `logs/citation_random_recovery_r50_thr06_k10_20260522.csv`
- Recovery quality multi-seed CSV: `logs/recovery_quality_adaptive_multiseed_20260522.csv`
- Recovery quality multi-seed summary CSV: `logs/recovery_quality_adaptive_summary_multiseed_20260522.csv`
