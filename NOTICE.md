# Third-Party Components and Competition Resources

This repository references external Kaggle datasets and model artifacts through
the attached input slugs in `kaggle/kernel-metadata.json`. Those resources are
not redistributed here. Users remain responsible for complying with the
competition rules and the terms attached to each external resource.

## HOCT Compatibility Audit

`kaggle/audit_hoct_rerank.py` is a read-only compatibility and edge-ranking
audit for the Higher-Order Cell Tracking Transformer (HOCT):

- Project: `royerlab/hoct`
- Copyright: 2026 Jordao Bragantini and the HOCT contributors
- URL: https://github.com/royerlab/hoct
- Source commit: `cabe8fd4bd1ccc3a18edc2b82b1e6501e396f357`
- License: MIT

The audit uses the published feature order and normalization constants and
verifies the external `general_v0` TorchScript model against SHA256
`024c2e4606275c96667907abfc9e0c27487b543480caf99d9ebd1d267cef8e4a`.
Neither the HOCT source tree nor its model weights are redistributed here.

## Official Competition Metric

The official patched metric and its documentation are maintained separately:

- https://github.com/royerlab/kaggle-cell-tracking-competition
- https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md

No metric exploit is intentionally included. The project does not add
artificial hubs, fake division forks, negative-time nodes, out-of-volume
nodes, or cross-dataset edges.

## Additional Public Method Reference

`kaggle/run_geometric_reference.py` reproduces a frozen configuration from
[Aman Atar's Biohub Geometric Fusion](https://www.kaggle.com/code/amanatar/biohub-geometric-fusion),
which extends [Igor Zharov's Biohub Harmonic Fusion](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion).
The runner requires the separately acquired, checksum-pinned reference
Notebook and the original model artifacts. It does not redistribute them.
This reproduction is an external method comparison, not an originality or
leaderboard-score claim for this repository.

`kaggle/run_flow_relink_experiment.py` evaluates the neighborhood-flow
association function from
[Anvith Pothula's biohub x138](https://www.kaggle.com/code/anvithpothula/biohub-x138)
on the frozen E029 detector graph. Its externally acquired Notebook is
checksum-pinned. Only the named association function is extracted; the
coordinate-regression head and the remaining x138 changes are excluded.
