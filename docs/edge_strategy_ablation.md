# Semantic Edge Construction Strategy Ablation

This note records the four-strategy ablation used to compare controlled repair with active-node semantic densification.

## Protocol

- Datasets: Cora, PubMed, ogbn-arxiv full graph.
- Node drop rate: 0.95.
- Runs: 3 seeds.
- Epochs: 300.
- Backbone: GraphSAGE with LLM embeddings.
- Recovery ratio: 0.5.
- k: 10.
- Adaptive alpha: 0.0.

The ablation uses the same dropped and sampled recovered training nodes across strategies within each seed. It compares:

- `ours_adaptive`: controlled repair around sampled recovered training nodes.
- `inverse_adaptive`: reverse-budget control, used as a negative control.
- `global_active_knn`: active-node global kNN densification.
- `cluster_active_knn`: active-node cluster-restricted kNN densification.

## Main Results

| Dataset | Strategy | Accuracy | Semantic added edges | Edge hit rate | Node hit rate | Total time |
|---|---|---:|---:|---:|---:|---:|
| Cora | ours_adaptive | 75.17 ¡À 1.34 | 215.7 | 14.82% | 38.10 ¡À 7.94% | 1.69s |
| Cora | inverse_adaptive | 74.00 ¡À 2.09 | 229.0 | 13.95% | 38.10 ¡À 7.94% | 1.52s |
| Cora | global_active_knn | 73.43 ¡À 0.76 | 19,946.7 | 0.32% | 60.32 ¡À 0.00% | 1.95s |
| Cora | cluster_active_knn | 71.90 ¡À 0.99 | 20,290.7 | 0.26% | 47.09 ¡À 3.99% | 4.15s |
| PubMed | ours_adaptive | 77.83 ¡À 1.67 | 106.7 | 8.74% | 29.76 ¡À 2.06% | 2.70s |
| PubMed | inverse_adaptive | 77.33 ¡À 1.64 | 107.0 | 8.71% | 29.76 ¡À 2.06% | 2.63s |
| PubMed | global_active_knn | 76.37 ¡À 1.45 | 147,124.7 | 0.01% | 50.00 ¡À 3.57% | 9.69s |
| PubMed | cluster_active_knn | 75.77 ¡À 2.41 | 148,405.3 | 0.01% | 48.81 ¡À 5.46% | 138.62s |
| ogbn-arxiv | ours_adaptive | 61.97 ¡À 0.32 | 126,533.7 | 6.60% | 15.30 ¡À 0.16% | 38.52s |
| ogbn-arxiv | inverse_adaptive | 61.98 ¡À 0.16 | 171,935.3 | 5.88% | 17.01 ¡À 0.11% | 38.90s |
| ogbn-arxiv | global_active_knn | 63.06 ¡À 0.26 | 995,315.3 | 2.88% | 42.10 ¡À 0.24% | 157.55s |
| ogbn-arxiv | cluster_active_knn | 61.93 ¡À 0.09 | 1,010,503.3 | 2.21% | 34.19 ¡À 0.05% | 511.43s |
## Interpretation

`inverse_adaptive` does not provide a stable gain over `ours_adaptive`. It is close on Cora, PubMed, and ogbn-arxiv, but it does not change the main picture.

`global_active_knn` gives the strongest accuracy on ogbn-arxiv, but it also adds nearly one million semantic edges. That makes it a densification diagnostic rather than a targeted repair method.

`cluster_active_knn` is similarly expensive and does not produce a consistent accuracy advantage. The clustering step also adds substantial overhead on PubMed and ogbn-arxiv.

Overall, the ablation supports the paper's conservative claim: the main method should stay focused on controlled repair around recovered nodes, while the active-node variants are best treated as diagnostics for semantic graph densification.

## Artifacts

- Cora/PubMed/ogbn-arxiv summary: `logs/citation_cuda_inverse_active_edge_strategy_ablation_summary_20260526_150608.csv`
- Cora/PubMed/ogbn-arxiv raw: `logs/citation_cuda_inverse_active_edge_strategy_ablation_raw_20260526_150608.csv`
- ogbn-arxiv summary: `logs/arxiv_full_cuda_inverse_active_edge_strategy_ablation_summary_20260526_151458.csv`
- ogbn-arxiv raw: `logs/arxiv_full_cuda_inverse_active_edge_strategy_ablation_raw_20260526_151458.csv`


