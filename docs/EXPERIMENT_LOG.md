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
- Download completion: before integrity verification at 2026-07-23 18:31 CST.
- Archive size: `87,393,127,165` bytes.
- `unzip -tq` result: no errors detected.
- Extraction completion: 2026-07-23 18:57 CST.
- Extracted inventory:
  - 199 training `.zarr` arrays;
  - 199 paired training `.geff` labels;
  - four test `.zarr` arrays;
  - `sample_submission.csv`.
- Disk usage after extraction: 164 GiB including the retained 82 GiB archive;
  extracted `train/` used 81 GiB and `test/` used 1.8 GiB.
- Raw log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/dataset_download_20260723_162300.log`.
- Raw integrity and extraction log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/dataset_unpack_20260723.log`.
- Status: complete and integrity checked.
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
- Gate satisfaction: 2026-07-23 18:57 CST at exactly 199 Zarr/GEFF pairs.
- Audit completion: 2026-07-23 18:58 CST.
- Execution commit:
  `0a0c5b31a140c9bb1ea3e0c48bb6a5b116d61111`.
- Compared all 19,701 adjacent transitions across 199 clips.
- Locally reproduced the discussion claim exactly:
  - `44b6`: 71 clips, zero affected clips, zero duplicate transitions;
  - `6bba`: 128 clips, 114 affected clips, 947 duplicate transitions;
  - total: 114 affected clips and 947 exact duplicate transitions.
- A first read-only summary command lost nested shell quotes and raised
  `NameError`; it wrote no files. The summary was then read successfully with
  `jq`.
- Status: complete; the frozen-frame hypothesis is now locally verified.
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

## S013 - E001 Terminal Validation

- Kernel terminal time: 2026-07-23 17:55 CST.
- Validation time: 2026-07-23 17:59 CST.
- Kaggle kernel version 3 status: `COMPLETE`.
- Output download directory: server `/tmp/biohub_e001_v3`.
- Submission rows: 236,095, comprising 120,186 nodes and 115,909 edges.
- Submission SHA256:
  `d85760b872c125ebf993b80d579c7065c3c36d8b980a273303b9ba01c3539487`.
- Fixed-eight official-spec lite CV:
  - score / adjusted edge Jaccard: `0.8789316334556234`;
  - edge TP / FP / FN: `3852 / 287 / 251`;
  - node recall: `0.9804152902312412`;
  - division Jaccard: `0.0`;
  - delta versus E000: `+0.0000020420913934`;
  - delta versus notebook-124 reference: `-0.0002881116112545268`;
  - the hidden-test submission hash was preserved exactly through CV.
- Independent graph audit:
  - zero violations;
  - maximum indegree `1`, maximum outdegree `2`;
  - 4,277 connected components;
  - largest component 215 nodes.
- Relative to E000, aggregate graph counts changed only in
  `44b6_0b24845f`: minus 60 nodes, 48 edges, and 12 five-node components.
  The other three hidden-test graph summaries were unchanged.
- Decision: hold E001 rather than submit while E000 remains `PENDING`. The local
  gain is negligible and participant reports indicate each leaderboard scoring
  job may consume several hours.
- Raw kernel-status log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/e001_kernel_status_20260723.log`.
- Raw output-download log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/e001_kernel_output_20260723.log`.
- Raw validation log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s013_e001_validation_20260723.log`.
- Independent audit JSON:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s013_e001_submission_audit_20260723.json`.
- Source edits on server: none.

## S015 - Public Support Pack Download

- Time: 2026-07-23 19:02 CST.
- Dataset:
  `pilkwang/biohub-tracking-support-pack-50ep-v1`, license `CC0-1.0`.
- Downloaded and extracted under ignored server `Dataset/support-pack/`.
- Size: 340 MiB across 87 files.
- The manifest declares primary UNet-transformer weight SHA256
  `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`.
- The pack includes public source, primary weights, and Kaggle Python 3.12
  offline wheels. The server `Kaggle` environment is Python 3.11, so compiled
  wheels are not installed blindly.
- Raw server log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s015_support_pack_download_20260723.log`.
- Source edits on server: none.

## S016 - Frozen-Transition Edge Analysis

- Preparation time: 2026-07-23 19:04 CST.
- Server execution time: 2026-07-23 19:26 CST.
- Server commit:
  `d70d9ebe579106e9a65f85ded2556d216aaf341b`.
- The fixed-eight CV set contains 31 locally verified frozen transitions across
  all four `6bba` clips and none across the four `44b6` clips.
- Added a dependency-free analysis that compares predicted physical edge
  displacement on frozen versus ordinary transitions and measures exact
  detection-position overlap between identical frames.
- Added three synthetic tests covering zero-motion frozen edges, cross-links
  despite identical detections, and clips without frozen transitions.
- All three focused tests passed locally and in Conda environment `Kaggle`.
- Across the 31 frozen transitions:
  - 8,421 predicted edges had mean displacement `0.916018` micrometres and
    median displacement `0.8125` micrometres;
  - 1,664 edges had exact zero displacement, a fraction of `0.197601`;
  - exact coordinate overlap was 1,664 of 8,460 source nodes (`0.196690`) and
    1,664 of 8,431 target nodes (`0.197367`).
- As a same-embryo control, ordinary `6bba` transitions had mean displacement
  `1.635166` micrometres and median displacement `1.675012` micrometres.
- Interpretation: the current graph already has lower motion on frozen
  transitions, but exact zero-motion correspondence covers only about one fifth
  of detections. E003 therefore remains a measured relinking experiment rather
  than an automatic promotion.
- Raw server log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s016_frozen_edge_analysis_20260723.log`.
- JSON report:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s016_frozen_edge_analysis_20260723.json`.
- Status: completed; no candidate submission produced.
- Source edits on server: none.

## S017 - Visible-Test Frozen-Frame Audit

- Time: 2026-07-23 19:48 CST.
- Server commit:
  `d70d9ebe579106e9a65f85ded2556d216aaf341b`.
- All four frozen-audit tests passed in Conda environment `Kaggle`.
- Compared all 396 adjacent transitions across the four visible test arrays.
- `44b6`: two clips, zero duplicate transitions.
- `6bba`: two clips, both affected, 20 duplicate transitions in total.
- The 20 pairs are detected from image bytes at runtime, not from a hard-coded
  schedule.
- The four visible test IDs also occur under `train/` with GEFF labels. Those
  paired labels are excluded from E003 parameter selection and are never copied
  into a submission. E003 calibration must use separate training holdouts.
- Raw server log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s017_test_frozen_frame_audit_20260723.log`.
- JSON report:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s017_test_frozen_frame_audit_20260723.json`.
- Status: completed; confirms that content-driven freeze handling has a visible
  test execution path without legitimizing label reuse.
- Source edits on server: none.

## S018 - E003 Offline Relinking Harness Preparation

- Preparation time: 2026-07-23 20:06 CST.
- Server execution: 2026-07-23 21:15-21:20 CST.
- Server commit:
  `8106952736afec8aef18d993e4858b806fa0dc2c`.
- Added a zero-motion linear-assignment post-processor that replaces only edges
  on transitions listed by a content-derived frozen-frame report.
- Added an independent patched-spec edge evaluator that reads single-chunk Zarr
  v3 GEFF arrays through the system `zstd` decoder. It does not require the
  Python 3.12-only compiled wheels in the public support pack.
- Candidate selection is restricted to fixed-eight training holdouts whose IDs
  are absent from the visible test directory:
  `44b6_341df25f`, `44b6_e57ff5c6`, `6bba_969618f6`, and
  `6bba_fc83837d`.
- The all-eight E001 run may be used once only to verify that the evaluator
  reproduces the recorded `3852 / 287 / 251` edge counts; it is not a tuning
  target.
- Added six focused tests for relinking, assignment gates, edge counting, and
  the node-count adjustment formula. Seventeen combined local tests passed.
- The first inline `screen` launch attempt at 20:08 CST created neither a
  window nor a log, so no evaluator or experiment ran. Rather than retrying the
  same fragile command, a versioned `scripts/run_e003_ablation.sh` runner now
  owns the baseline assertion, seven distance gates, graph audits, and summary
  generation.
- The versioned runner passed all 17 server tests.
- The independent evaluator exactly reproduced E001's all-eight edge counts
  `3852 / 287 / 251` and adjusted edge Jaccard
  `0.8789316334556233`. The recorded notebook value differs only by floating
  representation at the final decimal place.
- On the four disjoint tuning holdouts, unchanged E001 scored
  `0.8680921294395563` with edge counts `1834 / 135 / 142`.
- Zero-motion LAP results on those holdouts:

| Gate (um) | Adjusted edge Jaccard | Delta vs E001 |
|---:|---:|---:|
| 0.5 | 0.8352931691 | -0.0327989603 |
| 1.0 | 0.8454585763 | -0.0226335532 |
| 1.5 | 0.8522175697 | -0.0158745597 |
| 2.0 | 0.8633319821 | -0.0047601474 |
| 3.0 | 0.8676238368 | -0.0004682926 |
| 4.0 | 0.8680921294 | 0.0000000000 |
| 6.0 | 0.8680921294 | 0.0000000000 |

- Every candidate passed the independent graph audit. Smaller gates lost true
  edges; the 4 and 6 micrometre gates reproduced baseline edge counts exactly.
- Decision: reject this E003 zero-motion post-processing branch. It provides no
  disjoint-holdout gain and will not be submitted.
- Raw server log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s018_e003_ablation_20260723.log`.
- Sweep summary:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s018_e003_sweep_summary_20260723.json`.
- Status: completed with runner exit code `0`; E003 rejected.
- Source edits on server: none.

## S019 - E000 Terminal Score and Top-10% Verification

- Watcher terminal time: 2026-07-23 23:02:55 CST.
- Authoritative terminal submission query: 2026-07-24 01:00 CST.
- Submission ID: `54923913`.
- Description: `E000 clean baseline no metric exploit, T4, e44703a`.
- Kaggle status: `SubmissionStatus.COMPLETE`.
- Public score: `0.908`.
- Full leaderboard export time: `2026-07-23T17:00:39Z`.
- Team: `Steven #2`; member: `buaaauto`.
- Rank: 78 of 1,566 teams, approximately top 4.98%.
- Strict top-10% cutoff: rank 156; score at rank 156: `0.904`.
- Margin: 78 ranks better and `0.004` score above the displayed cutoff row.
- Result: the project objective is achieved.
- Raw terminal query and leaderboard summary log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s019_terminal_submission_and_leaderboard_20260723.log`.
- Full leaderboard archive:
  `/tmp/biohub_leaderboard_20260723/biohub-cell-tracking-during-development.zip`.
- Source edits on server: none.

## S020 - Final Repository Closure Audit

- Time: 2026-07-24 01:03 CST.
- Pre-closure commit:
  `23aaabddb8478d5578ed16d1253dede7cbd4bec3`.
- Local `HEAD`, local `origin/main`, live GitHub `main`, server `HEAD`, and
  server `origin/main` all resolved to the same commit.
- Local and server worktrees were clean with no staged or unstaged diff.
- The public repository contained 23 tracked files and zero tracked data,
  logs, submissions, model weights, archives, CSV outputs, Zarr arrays, or GEFF
  labels.
- Server `Dataset/`, `logs/`, and the S019 raw log were confirmed ignored.
- Seventeen focused tests, both Bash syntax checks, notebook-output stripping,
  whitespace validation, and the secret-pattern scan passed.
- `screen -ls` reported one detached session, `Kaggle`; it was retained.
- This closure record is the only change after the pre-closure audit. After
  push, it is fast-forwarded to the server and the same invariants are rechecked.

## S021 - Kernel Status and Goal Reopen

- Time: 2026-07-24 12:42 CST.
- Replaced non-TTY `ssh ... bash -ic` polling with `ssh -tt ... bash -ic`.
  This removes the misleading job-control warnings; they were shell warnings,
  not Kaggle failures.
- Kernel `buaaauto/biohub-clean-baseline-no-metric-exploit` had moved from
  `KernelWorkerStatus.RUNNING` to `KernelWorkerStatus.COMPLETE`.
- Submission `54923913` remained complete at Public Score `0.908`.
- Server and local repository remained clean at commit
  `5bf8c40ac9a7593556d541858b3a77c0ead2316a`.
- `screen -ls` showed exactly one detached session, `Kaggle`; it was retained.
- Raw log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s021_goal092_refresh_20260724.log`.
- Source edits on server: none.

## S022-S026 - Post-Rescore Public Route Audit

- Time: 2026-07-24 12:42-12:51 CST.
- Downloaded the full post-rescore leaderboard and listed 200 public kernels by
  votes, recency, and Kaggle score order.
- The 12:42 leaderboard top was `0.931`, followed by `0.929` and `0.923`;
  the large public cluster remained at `0.908`.
- Pulled ten public candidate notebooks into server `/tmp`, never into Git.
- Downloaded and audited the output of Pilkwang Kim's dual-seed confidence
  router. It ran successfully in 13m56s on two T4 GPUs and emitted 125,478
  nodes, 121,203 edges, and 351 division-like sources. Its public Kaggle page
  reports score `0.909`, so it is not a submission candidate.
- The public dual-seed CSV SHA256 is
  `1a363b983ca1dfd516dc1715c9d43b101574ae1a2d2971dd0aaf93567ee201df`.
  Its structure is otherwise valid, but the local bounds audit found one
  rounded `z=64` node for a depth-64 volume. This output was not submitted.
- Inspected `biohub-ct-mix-divaug`; its final cell explicitly adds
  negative-time, out-of-volume hub/fork structures and labels the operation a
  metric exploit. The path is rejected.
- Raw logs:
  `s022_goal092_leaderboard_kernels_20260724.log`,
  `s023_goal092_pull_public_kernels_20260724.log`,
  `s024_goal092_public_outputs_20260724.log`,
  `s025_goal092_public_candidate_audit_20260724.log`, and
  `s026_goal092_kernels_by_score_20260724.log`, all under server `logs/`.
- Source edits on server: none.

## S027-S031 - Official Scorer Runtime Audit

- Time: 2026-07-24 12:53-12:56 CST.
- The support pack contains the 400-epoch primary weight with expected SHA256
  `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771`
  and offline Python 3.12 wheels.
- The required server environment is Python 3.11.15 and initially lacks
  `tracksdata`, `geff`, `zarr`, `blosc2`, `pyscipopt`, `ilpy`, and `polars`.
- A pip dry-run confirmed compatible Python 3.11 wheels are available for the
  binary dependencies, including PySCIPOpt 6.2.1.
- Verified the official scorer `main` commit is
  `075fc5f5a52d11077f9dc2b074644618f26939e2`.
- The first full-leaderboard parser used API-style lowercase field names and
  failed with `KeyError: 'teamName'`; no experiment result was produced. The
  archive header was inspected, and the corrected parser used
  `Rank,TeamId,TeamName,LastSubmissionDate,Score,...`.
- Corrected exact snapshot: 1,579 teams, our rank 102 at `0.908`, strict
  top-10% cutoff rank 157 at `0.908`, and only three teams strictly above
  `0.92`.
- Raw logs `s027` through `s031` are under server `logs/`.
- Source edits on server: none.

## S032-S033 - E005 Baseline Launch and Runtime Pin Failure

- Time: 2026-07-24 12:59-13:03 CST.
- Fast-forwarded the clean server checkout from `5bf8c40` through `54c8c1c`
  and then to runner commit `2e476a7`; no server source was edited.
- Launched the versioned E005 baseline runner in retained `screen Kaggle`
  window `e005-metric`.
- All 20 server tests passed in 13.97 seconds.
- Dependency resolution stopped before installation because `zarr==3.2.1`
  requires Python 3.12, while the required `Kaggle` environment is Python
  3.11.15. Pip reported `zarr` 3.1.6 as the newest compatible release.
- No division analysis or official scoring ran, and no completion marker was
  created. The isolated runtime did not receive a partial package install.
- Corrective action: pin `zarr==3.1.6`, retain `numcodecs==0.15.1`, and rerun
  under a new run ID.
- Raw logs:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s032_goal092_pull_and_preflight_20260724.log`,
  `s033_e005_launch_20260724.log`, and
  `s033_e005_metric_baseline_20260724.log`.
- Source edits on server: none.

## S034 - E005b Official Runtime and Metric Completion

- Time: 2026-07-24 13:07-13:08 CST.
- Fast-forwarded the clean server checkout to `fdedb77`, which pins the
  isolated Python 3.11 runtime to `zarr==3.1.6`.
- Launched the corrected runner in retained `screen Kaggle` window
  `e005-metric2`; the failed `e005-metric` window was retained for provenance.
- All 20 focused project tests passed. The isolated runtime was built under
  ignored `Dataset/runtime-py311`, and the official scorer was pinned at
  `075fc5f5a52d11077f9dc2b074644618f26939e2`.
- Across all 199 annotated clips, the audit found 151 true divisions in 87
  clips: 26 in `44b6` and 125 in `6bba`. Median parent-to-child and sister
  distances were `5.745` and `10.570` micrometres, respectively.
- Official scoring of E001's final fixed-eight CSV returned `0.8789`:
  adjusted edge Jaccard `0.8789`, division Jaccard `0.0000`
  (`TP=0, FP=6, FN=7`), and node recall `0.9763`.
- Completion marker:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s033_e005_metric_baseline_20260724b.done`.
- Raw log and geometry report:
  `s033_e005_metric_baseline_20260724b.log` and
  `s033_ground_truth_divisions_20260724b.json`, both under server `logs/`.
- Source edits on server: none.

## S035-S038 - Inference Preflight and Artifact Inventory

- Time: 2026-07-24 13:09-13:10 CST.
- The first preflight expanded local `$PWD` before SSH execution, so the remote
  `PYTHONPATH` pointed at the wrong machine and imports failed. No inference or
  source edit occurred.
- The corrected preflight used absolute server paths. It verified the isolated
  runtime imports, enumerated the prediction CLI, and found the expected
  `edge_predictor_best.pth` checkpoint.
- The annotated corpus contains divisions in 21 of 71 `44b6` clips and 66 of
  128 `6bba` clips. The four visible-test IDs remain excluded from all tuning
  even though duplicate-named training labels exist.
- Inventoried E001's raw per-clip GEFF outputs separately from its final CSV so
  raw model graphs can be scored and augmented without changing the server
  checkout.
- Raw logs:
  `s035_e005_inference_preflight_20260724.log`,
  `s036_e005_inference_preflight_fixed_20260724.log`,
  `s037_e005_division_dataset_selection_20260724.log`, and
  `s038_e005_e001_artifact_inventory_20260724.log`, all under server `logs/`.
- Source edits on server: none.

## S039-S040 - Raw Graph Score and Division Recoverability

- Time: 2026-07-24 13:11-13:14 CST.
- Official scoring of the eight raw E001 ILP GEFFs returned `0.8951`:
  adjusted edge Jaccard `0.8951`, division Jaccard `0.0000`
  (`TP=0, FP=0, FN=7`), and node recall `0.9844`.
- Compared with the final fixed-eight CSV, notebook post-processing created six
  false division forks and reduced this offline official score. The public test
  distribution can differ, so this is diagnostic rather than a leaderboard
  estimate.
- The recoverability audit found all three matched nodes for six of seven true
  divisions. Three of seven could accept the missing direct fork without
  merging tracks; none was already a complete fork.
- By embryo, `44b6` had one detected and addable event out of one; `6bba` had
  five detected events and two addable events out of six.
- Restricting interpretation to the four non-visible tuning clips leaves four
  true divisions and two directly addable forks. This small oracle sample
  establishes non-zero headroom but is insufficient for threshold selection.
- A repository-root `pytest` invocation recursed into ignored dependency and
  scorer trees under `Dataset/`, collecting their third-party tests and
  reporting 46 missing optional-test-dependency errors. The project test suite
  itself did not regress; future server checks are explicitly scoped to
  `python -m pytest -q tests`.
- Reports:
  `s039_e005_raw_fixed8_official_score_20260724.log`,
  `s040_e005_raw_fixed8_recoverability_20260724.log`, and
  `s040_e005_raw_fixed8_recoverability_20260724.json`, all under server
  `logs/`.
- Source edits on server: none.

## S041 - E005 Balanced Dual-GPU Calibration Inference

- Time: 2026-07-24 13:26-13:39 CST.
- Fast-forwarded the clean server checkout to `394154a`; all 25 focused project
  tests passed.
- Selected 16 division-positive clips per embryo and matched each with one
  division-negative clip by annotated edge count. All four visible-test IDs
  were excluded before selection.
- The resulting 64 clips were split `32 + 32` across the two RTX 4090 GPUs.
  Annotated-edge workload proxies were `23,011` and `23,009`.
- Both temporary support-repository copies matched the guarded predictor SHA256
  and received the pre-ILP export patch. The project checkout itself was not
  modified.
- Both shards completed. Exactly 64 pre-ILP candidate graphs and 64 baseline
  ILP graphs were preserved under
  `/tmp/biohub_e005_divcal_20260724a/`.
- Completion marker:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s041_e005_division_calibration_inference_20260724a.done`.
- Raw logs:
  `s041_e005_division_calibration_inference_20260724a.log`,
  `s041_e005_gpu0_20260724a.log`, and
  `s041_e005_gpu1_20260724a.log`, all under server `logs/`.
- Source edits on server: none.

## S042 - Lower Division-Cost ILP Sweep Stopped

- Time: 2026-07-24 13:29-13:49 CST.
- Fast-forwarded the server to `632e6b4`; all 31 focused project tests passed.
  The runner waited for S041 and then started the `0.75` division-cost
  candidate.
- Only four of 64 clips completed in roughly nine minutes before scoring or
  advancing to the remaining four weights. Extrapolation put the five-weight
  serial solve near 12 hours, before official metric evaluation.
- The bottleneck was CPU-side ILP construction/solution on the dense pre-ILP
  graphs, not neural inference. The process was interrupted with `SIGINT`;
  no completion marker or experiment score was produced.
- The `e005-ilsweep` screen window and partial ignored `/tmp` output were
  retained. The sole detached `Kaggle` screen session remains active.
- Decision: stop this low-throughput path and use the same saved graphs for
  direct high-precision fork feature calibration.
- Raw log:
  `/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/s042_e005_division_weight_sweep_20260724a.log`.
- Source edits on server: none.
