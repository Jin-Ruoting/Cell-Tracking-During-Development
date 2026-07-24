# Biohub Cell Tracking During Development

Reproducible public notebook for the Kaggle research competition
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

The project achieved a clean public score of `0.908` and a top-10% result
without metric exploits on 2026-07-24.

## Method

The E005 notebook combines:

- a 3D cell detector with four-view flip test-time augmentation;
- a learned adjacent-frame edge scorer;
- integer-linear-programming lineage reconstruction;
- conservative, prediction-supported direct-division recovery; and
- final coordinate and graph-topology validation.

On 26 label-disjoint clips from both embryos, the pinned official scorer
improved from `0.8966` to `0.9120` while preserving the adjusted edge term.

See [NOTICE.md](NOTICE.md) before reusing the notebook.

## Public Files

- `kaggle/biohub_clean_baseline.ipynb`: complete executable notebook.
- `kaggle/kernel-metadata.json`: Kaggle kernel configuration.
- `NOTICE.md`: attribution and reuse notice.
