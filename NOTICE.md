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
