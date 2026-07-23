# Experiment Plan

## Objective

Reach the top 10% of the Biohub public leaderboard with a legitimate,
reproducible pipeline, then protect against public/private leaderboard shift by
validating across both training embryos.

## Competition Facts

- Task: detect cell centroids in 3D time-lapse volumes, link them across time,
  and reconstruct division-aware lineage graphs.
- Data: Zarr v3 arrays with typical shape `(100, 64, 256, 256)` and physical
  voxel scale `(z, y, x) = (1.625, 0.40625, 0.40625)` micrometres.
- Training set: 199 clips from two embryos (`44b6` and `6bba`); hidden test
  embryos are disjoint.
- Score: `adjusted_edge_jaccard + 0.1 * division_jaccard`.
- Submission: Kaggle Notebook, GPU or CPU runtime at most 12 hours, internet
  disabled, output named `submission.csv`.
- Public leaderboard: approximately 29% of hidden test data; final ranking uses
  the remaining 71%, so embryo-robust CV matters.

## E000: Reproduce Clean Public Baseline

Run the attributed no-exploit public pipeline unchanged apart from notebook
metadata. It combines:

1. 3D learned cell detection.
2. Learned adjacent-frame edge scoring.
3. ILP graph selection.
4. Conservative gap repair, short-track filtering, smoothing, and safe division
   handling.
5. Schema guards and fixed-eight official-spec local CV.

Promotion rule: submit only after the notebook succeeds, creates a valid
`submission.csv`, and preserves physically valid node coordinates and graph
edges.

## Next Experiments

| ID | Change | Reason | Main guard |
|---|---|---|---|
| E001 | Reproduce the clean baseline | Establish score and runtime | Exact notebook/version record |
| E002 | Disable five-node rescue | Isolate rescue contribution | Same detector and ILP settings |
| E003 | Tune detection threshold | Detection was the strongest public lever | Both embryos, fixed scorer |
| E004 | Freeze-aware relinking | `6bba` contains repeated frames | Detect duplicates from images, no schedule memorization |
| E005 | Motion/intensity LAP cost | Reduce dense-region track swaps | Component and edge FP controls |
| E006 | Conservative division calibration | Recover the weighted division term | Patched scorer only |
| E007 | Cross-embryo pseudo-label ensemble | Improve sparse supervision | No hidden-test probing |

## Evidence Policy

Public notebook claims, local CV, Kaggle runtime completion, and leaderboard
scores are recorded separately. A notebook that merely runs is not reported as
a validated leaderboard improvement.
