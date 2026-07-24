# Repository Updates

## U001 - Initial Project Scaffold

- Time: 2026-07-23
- Initialized the local `main` branch and connected
  `git@github.com:Jin-Ruoting/cell-tracking-during-development.git`.
- Added repository hygiene rules before the first commit, including exclusions
  for Codex/agent state, competition data, artifacts, submissions, outputs, and
  logs.
- Added the attributed Kaggle baseline notebook preparation path and kernel
  metadata.
- Added experiment, leaderboard, and repository audit documents.
- Server source changes: none.

## U002 - First Push and Server Bootstrap Record

- Time: 2026-07-23
- Initial commit:
  `788f4083b6f34220004001af0cce4c1617cfa57b`.
- Pushed `main` to GitHub and updated the local remote to GitHub's canonical
  `git@github.com:Jin-Ruoting/Cell-Tracking-During-Development.git` URL.
- Bootstrapped the previously empty server checkout from that exact commit.
- Created only ignored runtime directories on the server; project source
  remained untouched there.

## U003 - Canonical Kaggle Kernel Metadata

- Time: 2026-07-23
- Updated the kernel ID to Kaggle's canonical
  `buaaauto/biohub-clean-baseline-no-metric-exploit` slug after version 1 was
  accepted.
- Removed unsupported free-form Kaggle tags while retaining the valid `gpu`
  keyword.
- Recorded the exact launch commit, screen window, log path, and initial
  `RUNNING` state for E000.

## U004 - Pin Compatible Kaggle Runtime

- Time: 2026-07-23
- Recorded E000 version 1's terminal `ERROR` and exact CUDA compatibility
  failure.
- Pinned the same Kaggle Docker image and `NvidiaTeslaT4` machine shape used by
  the successful upstream public notebook.
- No model, threshold, post-processing, or submission logic changed.

## U005 - Record E000 Success and Submission

- Time: 2026-07-23
- Recorded kernel version 2's successful T4 execution, exact output hash,
  graph counts, runtime, and fixed-eight official-spec CV metrics.
- Recorded competition submission `54923913` and its initial `PENDING` state.
- Recorded the 81.4 GiB competition-data download running in the retained
  `Kaggle` screen.
- Kept all raw logs, downloaded output artifacts, data, and submissions outside
  Git.

## U006 - Correct Strict Top-10% Cutoff

- Time: 2026-07-23
- Refreshed the live leaderboard at 1,549 teams.
- Corrected the qualifying rank from an upward-rounded 155 to the strict
  `floor(1,549 * 0.10) = 154`.
- Verified that the score at displayed rank 154 remained `0.902`.

## U007 - Prepare E001 No-Rescue Control

- Time: 2026-07-23 17:16 CST.
- Corrected the experiment description to distinguish E000's public
  short-track-rescue revision from its stated clean baseline control.
- Pinned the exact upstream notebook SHA256 in the generator and reuse notice.
- Prepared E001 by disabling only adaptive five-node rescue; detection, ILP,
  motion, and all other post-processing settings remain unchanged.
- Kept notebook outputs stripped and updated Kaggle metadata for the E001 run.
- Server source changes: none; execution requires a later `git pull --ff-only`.

## U008 - Record E001 Launch

- Time: 2026-07-23 17:18 CST.
- Committed and pushed E001 as `c1191c3`, then fast-forwarded the clean server
  checkout to that exact commit with `git pull --ff-only`.
- Recorded Kaggle kernel version 3 acceptance, initial `RUNNING` state, retained
  screen windows, and raw log paths.
- Replaced ad hoc foreground status checks with an event-driven watcher that
  downloads terminal output to server `/tmp`.
- Server source changes: none.

## U009 - Convert Discussion Evidence into Experiment Gates

- Time: 2026-07-23 17:25 CST.
- Added source-linked notes for scoring latency, graph connectivity, frozen
  frames, global jumps, temporal affinity, and pseudo-label proposals.
- Marked all participant claims as unverified until reproduced locally.
- Added concrete E002-E004 validation gates and explicitly excluded
  patched-metric exploit paths.
- Recorded the decision not to duplicate the still-pending E000 submission.
- Server source changes: none; synchronization requires `git pull --ff-only`.

## U010 - Add Exact Frozen-Frame Audit

- Time: 2026-07-23 17:30 CST.
- Added `scripts/audit_frozen_frames.py` for dependency-free comparison of
  consecutive Zarr v3 frame chunks.
- Added focused synthetic-layout unit tests under `tests/`.
- Linked the audit command to E003's discussion-derived validation gate.
- Kept generated JSON reports and server logs outside Git.
- Server source changes: none; execution requires a later `git pull --ff-only`.

## U011 - Record Frozen-Audit Server Validation

- Time: 2026-07-23 17:33 CST.
- Fast-forwarded the server to `3220e40` before execution.
- Ran all four frozen-audit unit tests in the `Kaggle` Conda environment through
  the retained `Kaggle` screen; all passed.
- Recorded the exact server commit, screen window, and ignored raw-log path.
- Server source changes: none.

## U012 - Add Deferred Frozen-Audit Runner

- Time: 2026-07-23 17:35 CST.
- Replaced fragile inline `screen` quoting with a versioned Bash runner.
- Added explicit gates for active downloads, active unzip jobs, 199 Zarr clips,
  199 GEFF labels, and the `Kaggle` Conda environment.
- Kept progress logs, JSON reports, and completion markers outside Git.
- Server source changes: none; execution requires a later `git pull --ff-only`.

## U013 - Record Deferred Audit Launch

- Time: 2026-07-23 17:39 CST.
- Fast-forwarded the clean server checkout to runner commit `e049aa2`.
- Passed server Bash syntax validation, activated Conda environment `Kaggle`,
  and launched the runner in retained screen window `biohub-freeze-audit`.
- Verified the initial log reports a waiting state with zero extracted training
  pairs; no premature audit execution occurred.
- Server source changes: none.

## U014 - Add Independent Submission Auditor

- Time: 2026-07-23 17:42 CST.
- Added a standard-library CSV and graph validator with component-size
  diagnostics motivated by the scorer-connectivity discussion.
- Added optional coordinate-bound validation against Zarr metadata.
- Added four focused unit tests and linked the auditor from the discussion gates.
- Kept generated audit reports outside Git.
- Server source changes: none; execution requires a later `git pull --ff-only`.

## U015 - Record E000 Independent Graph Audit

- Time: 2026-07-23 17:44 CST.
- Fast-forwarded the server to `556e570` before execution.
- Passed all four submission-audit tests in the `Kaggle` environment.
- Verified E000's exact hash, row counts, lineage degrees, 4,289 components, and
  zero independent schema or graph violations.
- Kept the detailed JSON report and raw output in ignored server `logs/`.
- Server source changes: none.

## U016 - Record E001 Completion and Hold Decision

- Time: 2026-07-23 17:59 CST.
- Recorded Kaggle kernel version 3 `COMPLETE`, exact output hash, fixed-eight
  metric, graph counts, and zero independent audit violations.
- Quantified E001's negligible `+0.0000020421` fixed-eight delta over E000.
- Added E001 to the held-candidate record rather than creating a duplicate
  leaderboard scoring job while E000 is pending.
- Kept all downloaded outputs and raw validation reports outside Git.
- Server source changes: none.

## U017 - Record Dataset and Frozen-Audit Completion

- Time: 2026-07-23 19:00 CST.
- Recorded the 87.39 GB archive's successful integrity test and complete
  extraction into 199 training Zarr/GEFF pairs plus four test arrays.
- Promoted the frozen-frame discussion claim from unverified to locally
  reproduced: 947 exact transitions across 114 `6bba` clips and none in `44b6`.
- Updated E003's rationale while retaining the no-schedule-memorization guard.
- Kept the archive, extracted data, JSON audit, markers, and raw logs outside
  Git.
- Server source changes: none.

## U018 - Add Frozen-Transition Edge Analysis

- Time: 2026-07-23 19:04 CST.
- Recorded the ignored CC0 public support-pack download without adding its
  weights, wheels, or source copy to Git.
- Added a standard-library analysis for fixed-eight detection overlap and edge
  displacement on the 31 verified frozen transitions.
- Added three focused synthetic tests and linked the analysis to E003's promotion
  gates.
- Server source changes: none; execution requires a later `git pull --ff-only`.

## U019 - Record Frozen-Edge and Visible-Test Audits

- Time: 2026-07-23 19:51 CST.
- Fast-forwarded the server to `d70d9eb` before both runs.
- Passed the focused server tests and recorded the 31-transition fixed-eight
  edge analysis.
- Recorded 20 content-detected frozen transitions across the two visible
  `6bba` test arrays and none in the two visible `44b6` arrays.
- Added an explicit leakage guard because all four visible test IDs also occur
  under `train/`: their GEFF labels cannot be used for tuning or submission
  construction.
- Kept JSON reports, completion markers, and raw logs outside Git.
- Server source changes: none.

## U020 - Add Clean E003 Offline Ablation Harness

- Time: 2026-07-23 20:06 CST.
- Added content-report-driven zero-motion LAP relinking with explicit distance
  gates and per-dataset change statistics.
- Added a standalone patched-spec edge evaluator for submission CSV and GEFF
  labels without installing incompatible support-pack wheels.
- Added six synthetic tests; seventeen combined repository tests pass locally.
- Documented the one-time evaluator reproduction check and the disjoint-holdout
  tuning boundary.
- Server source changes: none; execution requires a later `git pull --ff-only`.

## U021 - Replace Fragile E003 Inline Launch

- Time: 2026-07-23 21:14 CST.
- Recorded that the first inline `screen` command produced no window or log and
  therefore ran no experiment.
- Added a versioned Bash runner that stops unless the independent evaluator
  exactly reproduces E001's known all-eight edge counts and score.
- The runner evaluates seven zero-motion distance gates only on the four
  disjoint tuning holdouts, audits every candidate graph, and writes all
  generated artifacts under ignored server `logs/`.
- Server source changes: none; execution requires a later `git pull --ff-only`.

## U022 - Record E003 Reproduction and Rejection

- Time: 2026-07-23 21:25 CST.
- Ran the versioned E003 harness from clean server commit `8106952`; all 17
  tests passed and the runner exited `0`.
- Reproduced E001's all-eight edge counts and adjusted score exactly with the
  independent evaluator.
- Evaluated seven zero-motion gates on disjoint training holdouts. Five gates
  reduced the score and the two broadest gates tied E001.
- All seven candidate graphs passed independent validation.
- Rejected E003 without creating a Kaggle submission and kept every generated
  CSV, JSON report, completion marker, and raw log outside Git.
- Server source changes: none.

## U023 - Record Top-10% Goal Completion

- Time: 2026-07-24 01:04 CST.
- Recorded E000 submission `54923913` as `COMPLETE` with public score `0.908`.
- Downloaded the full leaderboard to server `/tmp` and counted 1,566 teams.
- Verified team `Steven #2` (`buaaauto`) at rank 78, approximately top 4.98%;
  the strict top-10% cutoff is rank 156 with score `0.904`.
- Updated the leaderboard, experiment record, project status, and held-candidate
  decisions without adding the leaderboard archive or server log to Git.
- Fast-forwarded the server to the goal-completion commit and completed the
  pre-closure clean-repository audit.
- Server source changes: none.

## U024 - Add Final Closure Evidence

- Time: 2026-07-24 01:03 CST.
- Recorded the matching local, GitHub, and server SHA evidence from commit
  `23aaabd`.
- Recorded clean worktrees, zero forbidden tracked artifacts, passing tests and
  syntax checks, stripped notebook outputs, and one retained `Kaggle` screen
  session.
- This documentation-only closure update is the final repository change. After
  push, the server is fast-forwarded once more and the same invariants are
  rechecked.
- Server source changes: none.

## U025 - Open the Strict 0.92 Goal and Pin Patched Scoring

- Time: 2026-07-24 13:00 CST.
- Reopened the project objective as a public score strictly greater than
  `0.92` and recorded the exact 12:42 leaderboard snapshot.
- Reviewed current Discussion posts, score-sorted public kernels, and ten
  public candidate notebooks. Rejected both an explicit division exploit and a
  legal dual-seed method whose verified public score is only `0.909`.
- Added a dependency-light annotated-division geometry analyzer with three
  focused tests.
- Added versioned server runners to build an isolated Python 3.11 dependency
  target under ignored `Dataset/` and execute official scorer commit
  `075fc5f5a52d11077f9dc2b074644618f26939e2`.
- Twenty focused tests, Bash syntax checks, Python compilation, and whitespace
  validation pass locally.
- Server source changes: none; execution requires `git pull --ff-only`.

## U026 - Add Robust E005 Baseline Runner

- Time: 2026-07-24 13:02 CST.
- Added one versioned runner for the E005 preflight instead of embedding a
  multi-stage experiment in a fragile `screen` command.
- The runner records the clean commit, runs all tests, builds the isolated
  Python 3.11 runtime, analyzes all annotated division geometry, and evaluates
  E001 with the pinned official patched scorer.
- Runtime artifacts, JSON reports, completion markers, converted GEFFs, and raw
  logs remain outside Git.
- Server source changes: none; execution requires `git pull --ff-only`.

## U027 - Pin the Python 3.11 Zarr Runtime

- Time: 2026-07-24 13:04 CST.
- Recorded the first E005 runner's dependency-resolution failure before any
  package installation or metric execution.
- Replaced the incompatible `zarr==3.2.1` pin with `zarr==3.1.6`, the newest
  release pip confirmed for Python 3.11.
- Kept the server environment, scorer commit, metric logic, and experiment
  inputs unchanged.
- Server source changes: none; execution requires `git pull --ff-only`.

## U028 - Add Division Recoverability Audit

- Time: 2026-07-24 13:14 CST.
- Added an oracle diagnostic that matches predicted and annotated nodes within
  the official 7 micrometre radius, then measures whether each real division's
  parent and two daughters are present and can form a merge-free direct fork.
- The audit is training-only and never exports ground-truth coordinates or
  choices into a test submission.
- Added three synthetic tests for an addable fork, a conflicting second parent,
  and a missing daughter detection.
- Server source changes: none; execution requires `git pull --ff-only`.

## U029 - Record E005 Completion and Scope Project Tests

- Time: 2026-07-24 13:17 CST.
- Recorded the successful isolated Python 3.11 runtime, pinned official scorer,
  all-training division geometry, and official scores for both E001's final CSV
  and its raw ILP GEFF graphs.
- Recorded the corrected inference preflight, training-only recoverability
  result, and the strict exclusion of the four visible-test labels from tuning.
- Distinguished 46 third-party collection errors under ignored `Dataset/`
  dependency trees from the passing project suite.
- Scoped the E005 runner to `python -m pytest -q tests` so later runtime and
  scorer installations cannot be collected as project tests.
- Generated logs, reports, runtimes, scorer clones, predictions, and GEFF files
  remain outside Git.
- Server source changes: none; execution requires `git pull --ff-only`.

## U030 - Add Leakage-Safe Division Calibration Inference

- Time: 2026-07-24 13:25 CST.
- Added deterministic selection of division-positive training clips and
  edge-count-matched division-negative clips, with explicit dataset exclusions
  and balanced workload shards.
- Added a minimal, hash-guarded patch to preserve candidate graphs before ILP
  selection in temporary support-repository copies.
- Added a two-GPU E005 calibration runner that keeps all copied repositories,
  split manifests, predictions, GEFF graphs, and raw logs outside Git.
- The four visible-test IDs are hard-excluded before calibration split
  construction.
- Server source changes: none; execution requires `git pull --ff-only`.

## U031 - Add Reusable ILP Division-Cost Sweep

- Time: 2026-07-24 13:32 CST.
- Added a GEFF-to-GEFF ILP resolver so one pre-ILP neural inference pass can
  evaluate multiple division penalties without recomputing detections or edge
  probabilities.
- Added a direct official-scorer wrapper for GEFF directories and a versioned
  runner covering the original control plus five lower division costs.
- Added focused tests for stable weight labels, deterministic GEFF discovery,
  and duplicate-dataset rejection.
- Candidate graphs, scorer logs, and completion markers remain outside Git.
- Server source changes: none; execution requires `git pull --ff-only`.

## U032 - Add Direct-Fork Candidate Feature Audit

- Time: 2026-07-24 13:49 CST.
- Added a training-only analyzer for every one-child parent and nearby root in
  the following frame.
- Features include the saved pre-ILP edge probability, parent/daughter and
  sister geometry, midpoint displacement, predecessor motion, branch
  continuation, and candidate rank.
- Oracle labels are used only to measure recoverability and select a general
  rule; they are never exported into test predictions.
- Added focused tests for true-fork labeling, wrong-root rejection, ranking,
  and summary counts.
- Server source changes: none; execution requires `git pull --ff-only`.

## U033 - Record Calibration Inference and Stop Serial ILP Sweep

- Time: 2026-07-24 13:49 CST.
- Recorded successful two-GPU inference on 64 leakage-safe calibration clips,
  including exact shard balance and artifact counts.
- Recorded that the lower-cost serial ILP sweep produced no score and was
  stopped after four clips because its projected runtime was approximately 12
  hours before metric evaluation.
- Preserved the completed pre-ILP and baseline graphs for the faster direct-fork
  calibration path.
- Server source changes: none.

## U034 - Add Versioned Direct-Fork Audit Runner

- Time: 2026-07-24 13:53 CST.
- Added a runner that validates the project, requires the successful S041
  completion marker, and extracts direct-fork candidates from all 64 paired
  baseline/pre-ILP graphs.
- The candidate report, raw log, and completion marker remain under ignored
  server `logs/`.
- Server source changes: none; execution requires `git pull --ff-only`.

## U035 - Add Conservative Direct-Fork Candidate Generator

- Time: 2026-07-24 14:00 CST.
- Added five explicit prediction-only fork rules and a GEFF augmenter that adds
  at most one root edge to each one-child parent.
- Every addition rechecks parent outdegree one and candidate indegree zero
  immediately before writing, preventing merges and hubs.
- The augmenter does not accept ground-truth paths or oracle labels; training
  labels remain confined to offline rule evaluation.
- Added focused tests for threshold semantics, missing-motion handling,
  per-parent ranking, and rejection of unknown rule fields.
- Generated candidate graphs and reports remain outside Git.
- Server source changes: none; execution requires `git pull --ff-only`.

## U036 - Add Official Direct-Fork Sweep Runner

- Time: 2026-07-24 14:02 CST.
- Added a versioned runner that generates all five direct-fork candidates and
  scores the unchanged baseline plus every rule with the pinned official
  scorer.
- The runner rechecks the four-ID exclusion manifest and refuses to overwrite
  an existing output root.
- Candidate GEFFs, rule reports, per-rule metric logs, and completion markers
  remain outside Git.
- Server source changes: none; execution requires `git pull --ff-only`.

## U037 - Enforce One-to-One Fork Candidate Assignment

- Time: 2026-07-24 14:04 CST.
- The first S044 generation attempt stopped before scoring when two parents
  selected the same root; the immediate indegree guard prevented a merge.
- Replaced independent per-parent selection with deterministic greedy
  one-to-one assignment over both parent and candidate IDs.
- Added a regression test proving that a shared candidate is assigned only to
  the higher-confidence parent.
- Server source changes: none; execution requires `git pull --ff-only`.

## U038 - Add Official Fork-Outcome Diagnostics

- Time: 2026-07-24 14:12 CST.
- Added a scorer-grounded diagnostic that joins every added edge to the exact
  predicted fork sets classified as true positive, false positive, or ignored
  by patched division scoring.
- The first diagnostic covers the broad pre-ILP rule and the geometry-only
  rule; narrower pre-ILP rules are subsets of the broad candidate set.
- Added tests for candidate specification parsing and embryo-level outcome
  summaries.
- Reports and logs remain outside Git.
- Server source changes: none; execution requires `git pull --ff-only`.

## U039 - Support Callable Tracksdata Edge Lists

- Time: 2026-07-24 14:14 CST.
- S045a stopped on its first clip because this tracksdata build exposes
  `edge_list` as a method rather than an iterable property; no report was
  written.
- Added one compatibility helper for both API forms plus parameterized
  regression coverage.
- Server source changes: none; execution requires `git pull --ff-only`.

## U040 - Add Cross-Embryo Official-Outcome Rule Search

- Time: 2026-07-24 14:22 CST.
- Added a dependency-light beam search over conjunctions of prediction-only
  fork features, using only patched-scorer TP/FP outcomes from the broad
  calibration candidate.
- The search reports both total division Jaccard and the minimum embryo-level
  Jaccard, and requires at least one retained true positive in each embryo.
- Fixed calibration denominators are 21 annotated divisions for `44b6` and 54
  for `6bba`.
- Added tests for missing-motion bounds, canonical thresholds, and the fixed
  denominator formula.
- Search reports and logs remain outside Git.
- Server source changes: none; execution requires `git pull --ff-only`.
