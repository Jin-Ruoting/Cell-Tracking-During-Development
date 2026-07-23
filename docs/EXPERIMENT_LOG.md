# Experiment Log

All times use Asia/Shanghai unless stated otherwise. Raw logs remain on the
server and are never committed.

## S001 - Initial Server and Competition Audit

- Time: 2026-07-23
- Server repository directory existed but was empty.
- Intended `Dataset/` and `logs/` directories were not yet present.
- Conda environment `Kaggle`: Python 3.11.15, Kaggle CLI 2.2.3.
- GPUs: two NVIDIA GeForce RTX 4090 cards, each with 24,564 MiB total memory.
- Screen state: one detached `Kaggle` session (`2615526.Kaggle`); no extra
  sessions were created or removed.
- Kaggle account had accepted the competition rules and had no submissions.
- The public clean reference notebook was pulled only into server `/tmp` for
  read-only inspection; it was not placed in the server repository.
- Reference run evidence: four example test videos, 9.26 minutes prediction,
  236,203 output rows (120,246 nodes and 115,957 edges), and fixed-eight
  official-spec lite CV score `0.87892959136423`.

Status: audit complete; no project source was edited on the server.

## E000 - Public Rescue-Candidate Reproduction

- Status: kernel version 2 complete; competition submission pending
- Candidate: `buaaauto/biohub-clean-baseline-no-metric-exploit`
- Source: pinned, attributed public no-exploit notebook revision with adaptive
  five-node short-track rescue enabled.
- Expected clean public baseline reference: approximately `0.908`; this is not
  yet our verified score and is the upstream control rather than an E000 result.

## S002 - Server Repository Bootstrap

- Time: 2026-07-23
- Cloned the public GitHub repository into the previously empty server project
  directory at local/GitHub commit `788f4083b6f34220004001af0cce4c1617cfa57b`.
- Created the ignored `Dataset/` and `logs/` runtime directories.
- Verified `main...origin/main` with no tracked or untracked repository changes.
- Source edits on server: none.

Status: server is ready for future `git pull --ff-only` synchronization.

## S003 - E000 Kaggle Notebook Push

- Time: 2026-07-23 16:11 CST
- Server HEAD before launch:
  `00fc0fc60c58b969dd9fb0594d2de24e2cb2a191`.
- Launched from the persistent `Kaggle` screen in window `biohub-e000`; the
  window and screen were left open.
- Kaggle kernel version 1 was accepted and started at:
  `buaaauto/biohub-clean-baseline-no-metric-exploit`.
- Initial Kaggle status: `RUNNING`.
- Raw server log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/e000_kernel_push_20260723_161126.log`.
- Kaggle normalized the requested slug from `biohub-clean-baseline-no-hack` to
  `biohub-clean-baseline-no-metric-exploit`; local metadata will use the
  canonical slug for future versions.
- Source edits on server: none.

## S004 - E000 Version 1 Failure Diagnosis

- Time: 2026-07-23 16:13 CST
- Terminal Kaggle status: `ERROR`.
- Failure occurred before the first model inference completed and no
  `submission.csv` was produced.
- Root cause: Kaggle assigned a Tesla P100 (`sm_60`), while the current Kaggle
  PyTorch image used by the notebook supports CUDA architectures `sm_70` and
  newer. The first UNet CUDA operation raised
  `torch.AcceleratorError: no kernel image is available for execution on the device`.
- This was an execution-environment mismatch, not a model or graph failure.
- Correction: pin the exact upstream runtime image and
  `NvidiaTeslaT4` machine shape in kernel metadata.
- Downloaded failure artifacts only to server `/tmp/biohub_e000_v1` for
  inspection; no output entered Git.
- Raw status log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/e000_kernel_status_20260723_161126.log`.
- Raw output-download log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/e000_kernel_output_20260723_161126.log`.
- Source edits on server: none.

## S005 - E000 Version 2 Successful Notebook Run

- Launch time: 2026-07-23 16:18 CST
- Terminal time: 2026-07-23 16:55 CST
- Launch commit:
  `e44703ab27e9a9699d35bd6ca16b1edcec31c59e`.
- Runtime correction: pinned `NvidiaTeslaT4` and the upstream Kaggle image.
- Kaggle status: `COMPLETE`.
- Example-test prediction: 4 videos in 9.93 minutes.
- Submission rows: 236,203 total, comprising 120,246 nodes and 115,957 edges.
- Submission SHA256:
  `512d3f5fbc927d2c0cd59ad4b4bf22f7259acea07ea5d4c7ed7da706e2b65add`.
- Short-track rescue: 12 components / 60 nodes in one dataset.
- Fixed-eight official-spec lite CV:
  - combined score / adjusted edge Jaccard: `0.87892959136423`;
  - edge TP / FP / FN: `3852 / 287 / 251`;
  - node recall: `0.9804152902312412`;
  - division Jaccard: `0.0`;
  - the pre-CV and post-CV submission hashes were identical.
- Raw kernel-push log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/e000_v2_kernel_push_20260723_161757.log`.
- Raw kernel-status log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/e000_v2_kernel_status_20260723.log`.
- Raw output-download log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/e000_v2_kernel_output_20260723_165600.log`.
- Source edits on server: none.

## S006 - E000 Competition Submission

- Submission time: 2026-07-23 17:01 CST.
- Submission ID: `54923913`.
- Kernel: `buaaauto/biohub-clean-baseline-no-metric-exploit`, version 2.
- Message: `E000 clean baseline no metric exploit, T4, e44703a`.
- Initial status: `PENDING`; public and private scores were not yet available.
- An event-driven watcher runs in the existing `Kaggle` screen and logs status
  changes without editing repository files.
- Raw submission command log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/e000_v2_submission_20260723_165900.log`.
- Raw submission-status log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/e000_submission_status_54923913.log`.

## S007 - Competition Dataset Download

- Start time: 2026-07-23 16:23 CST.
- Server HEAD:
  `e44703ab27e9a9699d35bd6ca16b1edcec31c59e`.
- Download size reported by Kaggle: 81.4 GiB.
- Destination: ignored server `Dataset/` directory.
- Execution: existing `Kaggle` screen, window `biohub-data`; the screen session
  remains detached and retained.
- Status at record time: running.
- Raw log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/dataset_download_20260723_162300.log`.
- Source edits on server: none.

## S008 - E001 Clean Baseline Control Preparation

- Preparation time: 2026-07-23 17:16 CST.
- Launch time: 2026-07-23 17:18 CST.
- Rationale: E000's fixed-8 score `0.87892959136423` was below the pinned public
  notebook's control reference `0.879219745066878` by `0.000290153702648`.
- Controlled change: set only
  `BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE=0`; retain the same detector threshold,
  ILP costs, motion gates, and all other graph settings.
- Pinned upstream notebook SHA256:
  `b754eaffca194e1b1ebbf5aa6471016996313eea1f18af4ff94316df749a2684`.
- The preparation script rejects source drift, strips all execution output, and
  updates experiment-facing text to describe the control arm.
- Launch commit:
  `c1191c3129051895cafc5ff9fca6c6a61818c3cf`.
- Server synchronization: clean `git pull --ff-only` fast-forward from
  `143fe76` to `c1191c3`.
- Kaggle kernel version 3 was accepted under the existing canonical kernel ID;
  one direct, non-interactive status check returned `RUNNING` without shell
  job-control warnings.
- Execution: retained `Kaggle` screen, window `biohub-e001`.
- An event-driven terminal-state watcher runs in window
  `biohub-e001-watch`; on completion it downloads outputs to server `/tmp` for
  validation and leaves repository files untouched.
- Raw kernel-push log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/e001_kernel_push_20260723_171900.log`.
- Raw status log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/e001_kernel_status_20260723.log`.
- Raw output-download log after terminal state:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/e001_kernel_output_20260723.log`.
- Status: kernel version 3 running.
- Source edits on server: none.

## S009 - Discussion Review During Asynchronous Scoring

- Time: 2026-07-23 17:25 CST.
- E000 submission `54923913` remained `PENDING`; no duplicate submission was
  created.
- Public participant reports describe scoring durations of roughly four to seven
  hours and raise a graph-connectivity explanation for some timeouts. These are
  operational observations, not official guarantees.
- A high-vote discussion reports 947 exact frozen transitions across 114 of 128
  `6bba` clips and none across 71 `44b6` clips. This remains unverified locally
  until the competition dataset is extracted.
- A temporal-affinity discussion motivates later motion-field and consensus
  pseudo-label experiments, but the lower-cost detector and freeze-aware controls
  remain ahead of it in the experiment order.
- Metric-exploit suggestions in the same discussions were explicitly excluded.
- Detailed source links and promotion gates:
  `docs/DISCUSSION_NOTES.md`.
- Source edits on server: none.

## S010 - Frozen-Frame Audit Tool Preparation

- Time: 2026-07-23 17:30 CST.
- Added a standard-library-only audit for exact consecutive-frame duplicates in
  the competition's Zarr v3 `tzyx` arrays.
- The audit validates four-dimensional shape, unit time chunks, default
  slash-separated chunk keys, and the presence of every compared chunk.
- It supports multiple spatial chunks, parallel clip processing, per-embryo
  summaries, and JSON output outside Git.
- Four synthetic-layout unit tests cover exact duplicates, multi-chunk frames,
  invalid time chunks, split discovery, and embryo summaries.
- Server validation time: 2026-07-23 17:33 CST.
- Server validation commit:
  `3220e4039c9a4e0818dc2df3847fac7b327672b0`.
- All four unit tests passed in the `Kaggle` Conda environment from retained
  screen window `biohub-audit-test`.
- Raw server test log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s010_frozen_audit_tests_20260723.log`.
- Status: locally and server validated; full-data execution waits for download
  and extraction completion.
- Source edits on server: none.

## S011 - Deferred Full-Data Frozen Audit

- Time: 2026-07-23 17:35 CST.
- Two initial attempts to embed the complete wait condition directly in
  `screen -X` exited during window creation. Neither attempt produced an audit
  log, ran the audit, or changed server source.
- Correction: add a versioned Bash runner that waits for the Kaggle download and
  unzip processes to finish, requires exactly 199 `.zarr` and 199 `.geff`
  training entries, verifies the active `Kaggle` Conda environment, and only then
  runs the committed audit.
- Generated JSON, progress output, and completion markers remain in ignored
  server `logs/`.
- Runner commit:
  `e049aa2a55b945f47eab9cd63fbf44d0fc643fcc`.
- Server launch time: 2026-07-23 17:39 CST.
- Execution: retained `Kaggle` screen, window `biohub-freeze-audit`.
- Initial gate state: waiting with zero extracted Zarr/GEFF pairs while the
  competition archive is still downloading.
- Progress and terminal output log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s011_frozen_frame_audit_20260723.log`.
- Terminal JSON path:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s011_frozen_frame_audit_20260723.json`.
- Status: deferred runner active.
- Source edits on server: none.

## S012 - Independent Submission Audit Preparation

- Time: 2026-07-23 17:42 CST.
- The E000 `run_stats.csv` was inspected from `/tmp/biohub_e000_v2`; it records
  per-dataset nodes, edges, divisions, and post-processing counts but not
  connected-component sizes.
- An initial read from the server's default shell returned
  `python: command not found`; rerunning with the exact `Kaggle` environment
  interpreter succeeded. No server file was edited.
- Added a dependency-free submission auditor for CSV schema, row IDs, sentinels,
  node and edge uniqueness, dangling edges, temporal adjacency, lineage degrees,
  optional Zarr coordinate bounds, and connected-component statistics.
- Added four unit tests covering valid connectivity, graph violations, coordinate
  bounds, and malformed headers.
- Server validation time: 2026-07-23 17:44 CST.
- Server validation commit:
  `556e5707181ecd33735fdce7293815d02501a481`.
- All four submission-audit unit tests passed in Conda environment `Kaggle`.
- Independent E000 result:
  - `236,203` rows, `120,246` nodes, `115,957` unique edges, four datasets;
  - exact SHA256 matched the earlier runtime record;
  - zero schema or graph violations;
  - maximum indegree `1` and maximum outdegree `2` in every dataset;
  - 4,289 total connected components;
  - largest component was 215 nodes;
  - per-dataset edge-to-node ratios ranged from `0.937756` to `0.970278`.
- Coordinate bounds were not rechecked in this run because the full competition
  data are still downloading; the notebook's own clean-graph audit already
  reported no coordinate violations.
- Raw server log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s012_e000_submission_audit_20260723.log`.
- JSON report:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s012_e000_submission_audit_20260723.json`.
- Status: locally and server validated; E000 artifact passed independent audit.
- Source edits on server: none.
