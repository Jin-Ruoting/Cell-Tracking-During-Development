# Attribution and Reuse Notice

`kaggle/biohub_clean_baseline.ipynb` is derived from the public Kaggle notebook:

- Title: `Clean Approach + Lightweight Local CV | No Hack`
- Author: Yusuke Togashi
- URL: https://www.kaggle.com/code/yusuketogashi/clean-approach-lightweight-local-cv-no-hack
- Upstream state observed: 2026-07-23
- Upstream notebook SHA256:
  `b754eaffca194e1b1ebbf5aa6471016996313eea1f18af4ff94316df749a2684`

The current E016 revision preserves the verified E000 detector, spatial D4
test-time augmentation, ILP lineage construction, and graph postprocessing.
It routes `44b6` videos through the primary model and `6bba` videos through a
calibrated independent-seed detector/link ensemble, followed by strict graph
validation. All project code required by the published kernel is embedded in
the notebook. Users remain responsible for complying with the competition
rules and the terms attached to upstream datasets and artifacts.

The E016 notebook and `kaggle/run_dual_seed_control.py` adapt the guarded
independent-seed inference patch from:

- Title: `Biohub Cell Tracking: Two Seeds Logit Blend`
- Author: Pilkwang Kim
- URL:
  https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-two-seeds-logit-blend
- Reference state observed: 2026-07-24
- Reference notebook SHA256:
  `70e0c300ceae3cd7ee2cf1650c4a5f74463543e3aae1b486ba5f729a76281656`

The original reference notebook is not redistributed in this repository.
Derived dual-seed patch logic is embedded in the E016 notebook and is limited
to the `6bba` route. The control applies the verified reference patch only to a
temporary copy of the public support source, and both paths verify the primary
and secondary model-weight checksums.

The optional edge-logit TTA control in `kaggle/run_dual_seed_control.py` is
informed by a second public Pilkwang Kim notebook:

- Title: `Biohub Cell Tracking: Blend Preprocessings`
- URL:
  https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-blend-preprocessings
- Reference state observed: 2026-07-27
- Reference notebook SHA256:
  `fd4d166ef72afc8db2e191df6e7dad661b18151f6faf9fa303e97531b6de892c`

That reference notebook is not redistributed. The runner verifies an external
copy by SHA256 and independently integrates raw edge-logit aggregation into
the existing dual-seed control. The option is disabled by default and is not
part of the published E016 result.

`kaggle/audit_hoct_rerank.py` is a read-only compatibility and edge-ranking
audit for the Higher-Order Cell Tracking Transformer (HOCT):

- Project: `royerlab/hoct`
- Copyright: 2026 Jordao Bragantini and the HOCT contributors
- URL: https://github.com/royerlab/hoct
- Source commit:
  `cabe8fd4bd1ccc3a18edc2b82b1e6501e396f357`
- License: MIT

The audit uses the feature order and normalization constants published by
HOCT and verifies the external `general_v0` TorchScript model against SHA256
`024c2e4606275c96667907abfc9e0c27487b543480caf99d9ebd1d267cef8e4a`.
Neither the HOCT source tree nor its model weight is redistributed here.

The official patched metric and its documentation are maintained separately at:

- https://github.com/royerlab/kaggle-cell-tracking-competition
- https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md

No metric exploit is intentionally included. In particular, this project does not
add artificial hubs, fake division forks, negative-time nodes, out-of-volume nodes,
or cross-dataset edges.
