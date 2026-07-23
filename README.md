# Biohub Cell Tracking During Development

Reproducible experiments for the Kaggle research competition
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

The immediate target is a clean top-10% public leaderboard result without metric
exploits. The repository keeps source code and experiment records only. Competition
data, model artifacts, notebook outputs, submissions, and server logs are excluded.

## Current Baseline

The first submission candidate is a reproducible fork of Yusuke Togashi's public
[Clean Approach + Lightweight Local CV | No Hack](https://www.kaggle.com/code/yusuketogashi/clean-approach-lightweight-local-cv-no-hack)
notebook, observed on 2026-07-23 after Kaggle completed the patched-metric rescore.
The upstream clean pipeline uses a 3D detector, learned edge scorer, ILP graph
construction, conservative graph repair, and an official-spec local metric check.

The fork preserves the tracking logic and adds explicit attribution plus repository
hygiene. See [NOTICE.md](NOTICE.md) before reusing the notebook.

## Repository Layout

- `kaggle/`: notebook and Kaggle kernel metadata.
- `scripts/`: local preparation and validation utilities.
- `docs/EXPERIMENT_PLAN.md`: staged path from baseline reproduction to stronger models.
- `docs/EXPERIMENT_LOG.md`: server runs and observed results.
- `docs/LEADERBOARD.md`: leaderboard snapshots and our submissions.
- `docs/REPOSITORY_UPDATES.md`: auditable repository changes.

## Strict Workflow

All source and documentation edits happen in the local repository, followed by
commit and push. The server copy is updated only with `git pull --ff-only`; it is
used for execution and result inspection, never for direct source edits.

Server logs live outside Git at:

```text
/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs/
```

Competition data live outside Git at:

```text
/data/zqjinruoting/Kaggle/Cell Tracking During Development/Dataset/
```

## Kaggle Submission

The competition is notebook-only. The committed kernel metadata targets a GPU
notebook with internet disabled and the required public support datasets attached.
The notebook must create `/kaggle/working/submission.csv`.
