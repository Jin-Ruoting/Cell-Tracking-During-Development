# Biohub Cell Tracking During Development

Reproducible public notebook for the Kaggle research competition
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

The verified E000 reference achieved a clean public score of `0.908` and a
top-10% result without metric exploits on 2026-07-24.

## Method

The current E006 notebook preserves E000's detector, spatial D4
test-time augmentation, ILP settings, graph postprocessing, and conservative
short-track rescue. It then adds only prediction-supported division edges that
pass learned-edge, motion, and 3D geometry gates.

The output is rejected if it contains dangling or nonconsecutive edges,
multiple parents, hub-like sources, or coordinates outside the image volume.
The controlled validator confirmed that E006 preserves every E000 node and
edge apart from audited division additions.

On 26 label-disjoint clips from both embryos, however, E000 and the current
E006 rule both scored `0.8827` under the pinned official scorer. E006 is
therefore not promoted for submission. The validator also provides a fixed
sweep of rules selected on a separate calibration split.

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
