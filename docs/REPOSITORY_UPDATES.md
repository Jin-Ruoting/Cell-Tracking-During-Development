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
