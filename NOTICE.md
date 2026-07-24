# Attribution and Reuse Notice

`kaggle/biohub_clean_baseline.ipynb` is derived from the public Kaggle notebook:

- Title: `Clean Approach + Lightweight Local CV | No Hack`
- Author: Yusuke Togashi
- URL: https://www.kaggle.com/code/yusuketogashi/clean-approach-lightweight-local-cv-no-hack
- Upstream state observed: 2026-07-23
- Upstream notebook SHA256:
  `b754eaffca194e1b1ebbf5aa6471016996313eea1f18af4ff94316df749a2684`

The current E006 revision preserves the verified E000 detector, spatial D4
test-time augmentation, learned edge scorer, ILP lineage construction, and
graph postprocessing, then adds prediction-supported direct-division recovery
and strict graph validation. All project code required by the published kernel
is embedded in the notebook. Users remain responsible for complying with the
competition rules and the terms attached to upstream datasets and artifacts.

The official patched metric and its documentation are maintained separately at:

- https://github.com/royerlab/kaggle-cell-tracking-competition
- https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md

No metric exploit is intentionally included. In particular, this project does not
add artificial hubs, fake division forks, negative-time nodes, out-of-volume nodes,
or cross-dataset edges.
