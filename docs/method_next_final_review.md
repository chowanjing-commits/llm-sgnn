# method-next Final Review Snapshot

Date: 2026-05-30

This note records the current review state of the `method-next` branch before
any push or merge decision. It is a branch-level checklist for the cold-start
extension and related paper updates.

## Branch State

- Branch: `method-next`
- Check the live remote tracking state with `git status --short --branch`.
- Main branch was not modified or merged.
- No push has been performed from this review pass.

## Scope Completed

- Added cold-start pseudo-label reliability controls:
  `--min-pseudo-label-support`, `--pseudo-label-agreement`, and
  `--pseudo-label-loss-weight`.
- Added `scripts/utils/screen_cold_start_reliability.py` to screen
  confidence/support gates before launching downstream GNN training.
- Integrated cold-start mode into the main single-dataset runner for Ours
  backbones.
- Separated graph admission from pseudo-label supervision:
  `selected_mask` controls admitted text-only nodes and recovered edges;
  `pseudo_train_mask` controls which admitted nodes enter supervised loss.
- Scrubbed hidden cold-start labels from the returned `pseudo_y` training target
  outside the actual training mask.
- Verified no-cold-start controls are sparse controls, not full-graph or hidden
  label leakage.
- Ran and recorded complete main-entry cold-start matrices for Cora, PubMed,
  WikiCS, and full ogbn-arxiv across Ours-GCN, Ours-GAT, and Ours-GraphSAGE.
- Updated `docs/cold_start_recovery.md` and `docs/paper_draft_zh.md` with the
  current interpretation: cold-start remains an appendix extension and protocol
  capability, not the paper's main contribution.
- Replaced related-work placeholders with verified source links in the paper
  drafts.
- Added table and repository hygiene verification scripts.

## Current Technical Position

The safest cold-start setting remains edge-only recovery: admit selected
text-only nodes and recover semantic edges, but assign zero supervised-loss
weight to pseudo labels. This removes the major harm from noisy pseudo-label
supervision and approximately matches the no-cold-start sparse control.

High-reliability pseudo-label supervision is supported as an optional policy.
It can improve selected settings, especially full ogbn-arxiv with GraphSAGE, but
it is not stable enough across datasets and backbones to promote cold-start to
the main contribution.

A stricter confidence-threshold screen improves pseudo-label diagnostic accuracy
but does not improve the downstream conclusion: very high thresholds leave too
few pseudo-supervised nodes on small datasets and reduce full-arXiv GraphSAGE
accuracy below the unthresholded high-reliability setting.

## Verification Commands

Use the project conda environment:

```powershell
conda run -n llm-sgnn python -m py_compile src\cold_start.py scripts\utils\run_cold_start_recovery.py scripts\utils\run_single_dataset_pilot.py scripts\utils\screen_cold_start_reliability.py scripts\utils\verify_cold_start_protocol.py scripts\utils\verify_no_cold_start_control.py scripts\utils\verify_selective_pseudo_label_filter.py scripts\utils\verify_pseudo_label_loss_weight.py scripts\utils\verify_cold_start_paper_tables.py scripts\utils\verify_repository_hygiene.py
conda run -n llm-sgnn python scripts\utils\verify_cold_start_protocol.py
conda run -n llm-sgnn python scripts\utils\verify_no_cold_start_control.py
conda run -n llm-sgnn python scripts\utils\verify_selective_pseudo_label_filter.py
conda run -n llm-sgnn python scripts\utils\verify_pseudo_label_loss_weight.py
conda run -n llm-sgnn python scripts\utils\verify_cold_start_paper_tables.py
conda run -n llm-sgnn python scripts\utils\verify_repository_hygiene.py
conda run -n llm-sgnn python scripts\utils\screen_cold_start_reliability.py --datasets cora --num-runs 1 --confidence-thresholds 0.0,0.7 --min-supports 0,2 --output-prefix cold_start_reliability_smoke
git diff --check
```

The verification scripts cover:

- hidden cold-start labels do not affect admission, edges, pseudo labels, or
  training membership;
- no-cold-start controls do not add cold-start nodes, labels, or edges;
- low-confidence admitted nodes can stay in the graph while being withheld from
  pseudo-supervised training;
- zero-weight pseudo labels do not affect supervised loss;
- the reliability screen can regenerate gate-level diagnostics without training;
- paper Tables D1-D8 match curated CSV summaries;
- tracked files exclude large local artifacts, unexpected logs, embeddings,
  checkpoints, models, and generated binary caches.

## Repository Hygiene

Tracked files under `logs/` are limited to the existing `.gitignore` allowlist
of small paper-result CSVs. Local experiment outputs, including cold-start full
matrix logs and smoke logs, remain ignored. The following directories remain
ignored and should not be committed:

- `data/`
- `embeddings/`
- `models/`
- `checkpoints/`
- non-allowlisted `logs/`

## Ahead Commit Themes

Use the following command for the live ahead list:

```powershell
git log --oneline origin/method-next..HEAD
```

The ahead commits cover formal cold-start controls, pseudo-label support and
agreement filters, pseudo-label loss weighting, edge-only and high-reliability
experiment records, main-entry metadata hardening, hidden-label scrubbing,
paper-table verification, repository hygiene verification, and this review
snapshot.

## Recommended Next Decision

The branch is suitable for remote backup with `git push origin method-next`.
Merging into `main` should wait until the user explicitly decides that the
cold-start appendix extension and paper narrative are ready to enter the main
paper baseline.
