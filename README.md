# Biohub Cell Tracking During Development

Reproducible public notebook for the Kaggle research competition
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

The verified E000 reference achieved a clean public score of `0.908` without
metric exploits on 2026-07-24. The experimental E016 dual-seed candidate also
scored `0.908`; it did not improve the leaderboard baseline. The current E022
notebook is an offline-confirmed, post-hoc candidate and has no public score
yet.

## Method

The current E022 notebook applies the primary model and an independently
trained temporal model to both supported embryo domains. It uses calibrated
detection-blend alpha `0.65` for `44b6` and `0.475` for `6bba`, with the same
low-margin link consensus, spatial D4 detection TTA, ILP settings, and exact
E000 graph postprocessing. Unknown embryo prefixes fail closed.

The output is rejected if it contains dangling or nonconsecutive edges,
multiple parents, hub-like sources, or coordinates outside the image volume.
Both model weights are verified by SHA256 before inference. Searched divisions
and all label-dependent routes remain disabled.

On the frozen 64-video corpus, independently materialized E022 scored
`0.9026472386`. It passed strict binary-lineage checks and improved both embryo
groups and both embryo-balanced alternating halves over E000. Because the two
alphas were selected after observing embryo-group results, this evidence has
explicit post-hoc overfitting risk and is not presented as a public
leaderboard improvement.

The offline validator also supports an exact, externally pinned parity check
for Pilkwang Kim's public two-seed Notebook v40. Supplying
`--expected-notebook-sha256` makes that mode fail closed on the Notebook
source, while the public-v40 DeepCenter checkpoint is always verified before
the original postprocessor is loaded. The mode is separate from the E022
submission Notebook and does not change the public-score claims above.

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
