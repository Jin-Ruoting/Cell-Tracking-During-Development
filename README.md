# Biohub Cell Tracking During Development

Reproducible public notebook for the Kaggle research competition
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

The verified E000 reference achieved a clean public score of `0.908` and a
top-10% result without metric exploits on 2026-07-24.

## Method

The current E016 notebook preserves E000's detector, spatial D4 test-time
augmentation, ILP settings, graph postprocessing, and conservative short-track
rescue for every `44b6` video. For `6bba` videos only, it combines the primary
model with an independently trained temporal model through calibrated
detection blending and low-margin link consensus.

The output is rejected if it contains dangling or nonconsecutive edges,
multiple parents, hub-like sources, or coordinates outside the image volume.
Both model weights are verified by SHA256 before inference. Searched divisions
and all label-dependent routes remain disabled.

E016 improved a 26-video screen by `+0.0094`. On a disjoint 64-video
confirmation corpus, it improved the pinned official score from `0.8947` to
`0.8992` (`+0.0045`), stayed exactly equal to primary on all `44b6` rows, and
improved both prespecified `6bba` halves. These are offline validation results;
the E016 public leaderboard score has not yet been verified.

See [NOTICE.md](NOTICE.md) before reusing the notebook.

## Public Files

- `kaggle/biohub_clean_baseline.ipynb`: complete executable notebook.
- `kaggle/kernel-metadata.json`: Kaggle kernel configuration.
- `kaggle/validate_e006_postprocess.py`: offline controlled-variant validator.
- `kaggle/audit_deepcenter_rescue.py`: sparse-label detector-complement audit.
- `kaggle/run_dual_seed_control.py`: pinned independent-seed inference control.
- `kaggle/audit_hoct_rerank.py`: pinned HOCT edge-ranking compatibility audit.
- `kaggle/audit_e000_error_budget.py`: official-matcher error-budget audit.
- `NOTICE.md`: attribution and reuse notice.
