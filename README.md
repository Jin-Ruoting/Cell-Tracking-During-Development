# Biohub Cell Tracking During Development

Reproducible public notebook for the Kaggle research competition
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

The verified E000 reference achieved a clean public score of `0.908` without
metric exploits on 2026-07-24. The experimental E016 dual-seed candidate also
scored `0.908`; it did not improve the leaderboard baseline. The current E025
notebook is published as
[Kaggle Kernel version 1](https://www.kaggle.com/code/buaaauto/biohub-e025-guarded-dual-seed-center-gaps)
and is awaiting its own public score. Pilkwang Kim's pinned public v40 run
scored `0.912`; that score is an attributed method reference, not an E025
result.

## Method

E025 starts from this repository's verified E016 `0.908` submission. It
applies the primary model and an independently trained temporal model to every
hidden dataset with global detection-blend alpha `0.475`, low-margin link
consensus, spatial D4 detection TTA, ILP disappearance weight `1.5`, and a
relaxed motion gate of `10.0 um`. These settings and the center-confirmed
synthetic-gap rule are adapted from
[Pilkwang Kim's public v40 method](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-two-seeds-logit-blend?scriptVersionId=337798568)
with explicit attribution.

The output is rejected if it contains dangling or nonconsecutive edges,
multiple parents, hub-like sources, or coordinates outside the image volume.
The primary, independent-seed, and DeepCenter weights are verified by SHA256;
the DeepCenter checkpoint must also report epoch `500`. Only newly synthetic
gap midpoints with endpoint span at least `8.5 um` are center-confirmed at
threshold `0.25`. Observed or shorter-gap midpoints bypass that gate. E025
retains a local per-source safe-division guard so repairs cannot create
nonbinary lineages. Searched divisions and label-dependent routes remain
disabled.

This repository describes E025 as an attributed engineering integration over
the verified `0.908` baseline, not as an independently invented `0.912`
method. Its own leaderboard score must be established by its Kaggle
submission.

The offline validator also supports an exact, externally pinned
postprocessing-parity check for Pilkwang Kim's public two-seed Notebook v40.
Supplying
`--expected-notebook-sha256` makes that mode fail closed on the Notebook
source, while the public-v40 DeepCenter checkpoint is always verified before
the original postprocessor is loaded. This mode evaluates externally supplied
baseline GEFF graphs; it is not an end-to-end inference-parity claim. It is
separate from E025, which retains the local binary-lineage guard.

See [NOTICE.md](NOTICE.md) before reusing the notebook.

## Public Files

- `kaggle/biohub_clean_baseline.ipynb`: complete executable notebook.
- `kaggle/kernel-metadata.json`: Kaggle kernel configuration.
- `kaggle/validate_e006_postprocess.py`: offline controlled-variant validator,
  including the externally pinned public-v40 parity mode.
- `kaggle/audit_deepcenter_rescue.py`: sparse-label detector-complement audit.
- `kaggle/run_dual_seed_control.py`: pinned independent-seed inference control
  with optional pre-ILP graph export and guarded edge-logit TTA.
- `kaggle/audit_hoct_rerank.py`: pinned HOCT edge-ranking compatibility audit.
- `kaggle/audit_e000_error_budget.py`: official-matcher error-budget audit.
- `NOTICE.md`: attribution and reuse notice.
