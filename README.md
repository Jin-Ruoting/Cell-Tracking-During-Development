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
E006 remains an experimental candidate until its hidden output is shown to
preserve every E000 node and edge apart from audited division additions.

On 26 label-disjoint clips from both embryos, the searched-division rule was
directionally positive under the same D4 inference setting. This offline result
is not presented as a verified leaderboard score.

See [NOTICE.md](NOTICE.md) before reusing the notebook.

## Public Files

- `kaggle/biohub_clean_baseline.ipynb`: complete executable notebook.
- `kaggle/kernel-metadata.json`: Kaggle kernel configuration.
- `NOTICE.md`: attribution and reuse notice.
