# Label-Conditioned Synthetic Text Recovery Ablation

Date updated: 2026-05-23

## Purpose

This is an independent ablation, not the main method.

Question:

If dropped training nodes do not have node-specific raw text, can class-conditioned synthetic text provide a useful semantic prior for pseudo-neighborhood repair?

Important interpretation: the synthetic text is generated from the ground-truth class label. This is label-informed and should be treated as an upper-bound diagnostic, not as a deployable cold-start method. The goal is missing-sample mitigation, not replacement of true node text.

## Generation Setup

- Generation model: `models/Qwen3-4B-Instruct-2507`
- Source: downloaded from ModelScope after direct HuggingFace download was incomplete and `hf-mirror.com` failed on metadata requests.
- Generation backend: HuggingFace local loading
- Texts per class: 12
- Temperature: 0.6
- Top-p: 0.9
- Repetition penalty: 1.15
- No-repeat ngram size: 3
- Main synthetic style: `synthetic_text_style=length_matched`
- Filtering: `ban_dataset_name=true`, `max_jaccard_similarity=0.72`
- Embedding model: existing citation text embedding pipeline, `sentence-transformers/all-MiniLM-L6-v2`

Text format:

- Cora/PubMed: length-matched title-plus-abstract style text.
- ogbn-arxiv: title-length text, because the current arXiv LLM cache uses only real paper titles from OGB `titleabs.tsv`.

## Training Protocol

- Datasets: Cora, PubMed, ogbn-arxiv
- Node drop rates: 0.75, 0.85, 0.95
- Edge drop rate: 0.0
- Repair policy: adaptive
- Candidate k: 10
- Recovery ratio: 0.5
- Max edges per recovered node: 10
- Adaptive threshold alpha: 0.0
- Runs: 3 seeds
- Epochs: 300

Modes:

- `true_text`: recovered dropped training nodes keep their original real-text embeddings.
- `label_text`: recovered dropped training nodes use embeddings of label-conditioned synthetic paper text.

## Main Results

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

## arXiv Diagnostic With GAT

The arXiv synthetic-text run also includes GAT sparse baselines. This matters because `label_text` should be interpreted against stronger sparse message-passing baselines, not only GCN.

| Node drop | GCN LLM Sparse | GAT LLM Sparse | True-text Ours | Synthetic-label-text Ours |
|---:|---:|---:|---:|---:|
| 0.75 | 58.47 ± 0.11 | 59.84 ± 0.45 | 60.43 ± 0.17 | 57.66 ± 0.28 |
| 0.85 | 56.53 ± 0.07 | 58.67 ± 0.35 | 60.23 ± 0.25 | 55.87 ± 0.26 |
| 0.95 | 55.87 ± 0.94 | 56.91 ± 0.90 | 60.18 ± 0.25 | 52.68 ± 1.13 |

## Generation Quality

| Dataset | Quality pass | Length ok | Dataset leak | Repeated ngram | Mean word count | Mean max Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 91.7 | 91.7 | 0.0 | 0.0 | 143.0 | 0.31 |
| PubMed | 66.7 | 66.7 | 0.0 | 0.0 | 195.2 | 0.32 |
| ogbn-arxiv | 57.5 | 76.0 | 0.0 | 0.0 | 8.2 | 0.58 |

## Interpretation

The synthetic-label-text condition supports a conservative conclusion:

- It can mitigate missing-sample effects in several Cora and PubMed settings.
- It consistently remains below true node text.
- On ogbn-arxiv, it is weaker than `GAT (LLM, Sparse)`, showing that short title-level synthetic priors are not strong enough to replace true node-specific text.
- The arXiv quality diagnostics show limited generation quality: short text, higher lexical overlap, and lower quality pass rate.

This ablation should be written as a few-shot or lightweight deployment diagnostic: class-conditioned generation can provide a partial semantic prior when true node text is unavailable, but the benefit is limited by LLM generation quality and should not be treated as a replacement for real text.

## Artifacts

- Cora summary: `logs/cora_synthetic_text_recovery_summary_20260523_152147.csv`
- Cora raw: `logs/cora_synthetic_text_recovery_raw_20260523_152147.csv`
- Cora quality: `logs/cora_synthetic_text_recovery_quality_20260523_152147.csv`
- PubMed summary: `logs/pubmed_synthetic_text_recovery_summary_20260523_154008.csv`
- PubMed raw: `logs/pubmed_synthetic_text_recovery_raw_20260523_154008.csv`
- PubMed quality: `logs/pubmed_synthetic_text_recovery_quality_20260523_154008.csv`
- arXiv summary: `logs/arxiv_synthetic_text_recovery_summary_20260523_172141.csv`
- arXiv raw: `logs/arxiv_synthetic_text_recovery_raw_20260523_172141.csv`
- arXiv quality: `logs/arxiv_synthetic_text_recovery_quality_20260523_172141.csv`
- Synthetic texts: `embeddings/*_synthetic_text_hf_models_Qwen3-4B-Instruct-2507_n12_length_matched.json`
- Synthetic embeddings: `embeddings/*_synthetic_text_hf_models_Qwen3-4B-Instruct-2507_n12_length_matched.pt`
