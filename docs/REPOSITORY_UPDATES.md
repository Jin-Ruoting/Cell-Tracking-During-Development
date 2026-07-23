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
