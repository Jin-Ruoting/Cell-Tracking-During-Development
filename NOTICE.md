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

`kaggle/run_dual_seed_control.py` can reproduce a bounded independent-seed
control by verifying and applying the guarded inference patch from:

- Title: `Biohub Cell Tracking: Two Seeds Logit Blend`
- Author: Pilkwang Kim
- URL:
  https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-two-seeds-logit-blend
- Reference state observed: 2026-07-24
- Reference notebook SHA256:
  `70e0c300ceae3cd7ee2cf1650c4a5f74463543e3aae1b486ba5f729a76281656`

The reference notebook remains an external input and is not redistributed in
this repository. The control runs its verified patch only against a temporary
copy of the public support source and verifies both model-weight checksums.

The official patched metric and its documentation are maintained separately at:

- https://github.com/royerlab/kaggle-cell-tracking-competition
- https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md

No metric exploit is intentionally included. In particular, this project does not
add artificial hubs, fake division forks, negative-time nodes, out-of-volume nodes,
or cross-dataset edges.
