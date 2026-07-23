# Attribution and Reuse Notice

`kaggle/biohub_clean_baseline.ipynb` is prepared from the public Kaggle notebook:

- Title: `Clean Approach + Lightweight Local CV | No Hack`
- Author: Yusuke Togashi
- URL: https://www.kaggle.com/code/yusuketogashi/clean-approach-lightweight-local-cv-no-hack
- Upstream state observed: 2026-07-23
- Upstream notebook SHA256:
  `b754eaffca194e1b1ebbf5aa6471016996313eea1f18af4ff94316df749a2684`

E000 preserved the pinned revision's tracking logic. The current E001 preparation
selects the A/B control described by that revision by changing only
`BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE` from `1` to `0`; it also makes attribution
and experiment text explicit, clears execution state, and normalizes notebook
metadata. Kaggle competition rules permit public notebook sharing and reuse, but
users remain responsible for complying with the competition rules and the terms
attached to upstream datasets and artifacts.

The official patched metric and its documentation are maintained separately at:

- https://github.com/royerlab/kaggle-cell-tracking-competition
- https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md

No metric exploit is intentionally included. In particular, this project does not
add artificial hubs, fake division forks, negative-time nodes, out-of-volume nodes,
or cross-dataset edges.
