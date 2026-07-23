# Biohub Cell Tracking During Development

Reproducible experiments for the Kaggle research competition
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

The immediate target is a clean top-10% public leaderboard result without metric
exploits. The repository keeps source code and experiment records only. Competition
data, model artifacts, notebook outputs, submissions, and server logs are excluded.

## Current Experiments

The first submission candidate is a reproducible fork of Yusuke Togashi's public
[Clean Approach + Lightweight Local CV | No Hack](https://www.kaggle.com/code/yusuketogashi/clean-approach-lightweight-local-cv-no-hack)
notebook, observed on 2026-07-23 after Kaggle completed the patched-metric rescore.
The upstream clean pipeline uses a 3D detector, learned edge scorer, ILP graph
construction, conservative graph repair, and an official-spec local metric check.

E000 reproduced that public revision's adaptive five-node short-track rescue.
E001 is its controlled baseline arm: it disables only the rescue switch because
E000's fixed-8 score was `0.000290154` below the public control reference. The
source snapshot is SHA256-pinned, outputs are stripped, and the transformation is
reproducible through `scripts/prepare_public_baseline.py`.

The repository also reproduces the discussion-reported frozen frames directly
from image content. A clean zero-motion relinking sweep was evaluated on
training holdouts disjoint from the visible test IDs; it produced no gain and
was rejected rather than submitted.

See [NOTICE.md](NOTICE.md) before reusing the notebook.

## Repository Layout

- `kaggle/`: notebook and Kaggle kernel metadata.
- `scripts/`: local preparation and validation utilities.
- `docs/EXPERIMENT_PLAN.md`: staged path from baseline reproduction to stronger models.
- `docs/EXPERIMENT_LOG.md`: server runs and observed results.
- `docs/DISCUSSION_NOTES.md`: discussion claims translated into validation gates.
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
