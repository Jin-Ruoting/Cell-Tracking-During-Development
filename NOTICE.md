# Attribution and Reuse Notice

`kaggle/biohub_clean_baseline.ipynb` is derived from the public Kaggle notebook:

- Title: `Clean Approach + Lightweight Local CV | No Hack`
- Author: Yusuke Togashi
- URL: https://www.kaggle.com/code/yusuketogashi/clean-approach-lightweight-local-cv-no-hack
- Upstream state observed: 2026-07-23
- Upstream notebook SHA256:
  `b754eaffca194e1b1ebbf5aa6471016996313eea1f18af4ff94316df749a2684`

The current E025 revision starts from this repository's verified E016 public
submission `54972789` at `0.908`. It preserves the spatial D4 detector,
lineage construction, and graph-validation framework while applying a global
independent-seed detection alpha of `0.475`, ILP disappearance weight `1.5`,
relaxed motion gate `10.0 um`, and center confirmation of long synthetic-gap
midpoints. All E025-specific integration logic is embedded in the notebook;
upstream inference source and model artifacts are supplied by the attached
public datasets. Users remain responsible for complying with the competition
rules and the terms attached to upstream datasets and artifacts.

The E025 notebook and `kaggle/run_dual_seed_control.py` adapt the guarded
independent-seed inference patch from:

- Title: `Biohub Cell Tracking: Two Seeds Logit Blend`
- Author: Pilkwang Kim
- URL:
  https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-two-seeds-logit-blend
- Reference state observed: 2026-07-24
- Reference notebook SHA256:
  `70e0c300ceae3cd7ee2cf1650c4a5f74463543e3aae1b486ba5f729a76281656`

The original reference notebook is not redistributed in this repository.
Derived dual-seed patch logic is embedded in E025. The control applies the
verified reference patch only to a temporary copy of the public support
source, and both paths verify the primary and secondary model-weight
checksums.

The optional postprocessing-parity mode in
`kaggle/validate_e006_postprocess.py` targets a later state of the same public
Notebook:

- Version: `40`
- Kaggle run: `337798568`
- Pinned URL:
  https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-two-seeds-logit-blend?scriptVersionId=337798568
- Reference state observed: 2026-07-27
- Reference notebook SHA256:
  `e75c2c384294c8e151993b426e7739b16a4646f6cf9bb23dd448fc78cd230e50`

The v40 source is not redistributed. The validator requires the user-supplied
external Notebook and can fail closed against its SHA256. Its exact
postprocessing arm also uses the public DeepCenter epoch-500 checkpoint at
SHA256
`8164d1ffa07f87e0506027a0392edeab7939a32bd5e3f756377c0d72885cf127`;
neither that model nor its dataset is included here. The arm processes
externally supplied baseline GEFF graphs and is not an end-to-end inference
parity claim.

Pilkwang Kim's pinned v40 run `337798568`, submission `54969360`, publicly
scored `0.912`. E025 integrates its global blend and center-confirmed
synthetic-gap settings over this repository's verified `0.908` baseline. It
also adds fail-closed artifact checks and retains a local per-source
safe-division guard. E025 was evaluated through its own competition submission
`55023652`, which independently achieved a public score of `0.912`. The
matching displayed score does not transfer authorship of the upstream method;
E025 remains an attributed engineering integration rather than a claim of
independent invention.

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
part of the E025 notebook.

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
