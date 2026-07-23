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

## E000: Reproduce Public Rescue Candidate

Run the pinned, attributed no-exploit public revision unchanged apart from
notebook metadata. That revision enables a conservative five-node short-track
rescue on top of its stated `0.908` baseline. The pipeline combines:

1. 3D learned cell detection.
2. Learned adjacent-frame edge scoring.
3. ILP graph selection.
4. Conservative gap repair, short-track filtering, smoothing, and safe division
   handling.
5. Schema guards and fixed-eight official-spec local CV.

Promotion rule: use it as the first leaderboard anchor only after the notebook
succeeds, creates a valid `submission.csv`, and preserves physically valid node
coordinates and graph edges.

## Next Experiments

| ID | Change | Reason | Main guard |
|---|---|---|---|
| E001 | Disable five-node rescue | Run the public revision's A/B control | Same detector and ILP settings |
| E002 | Tune detection threshold | Detection was the strongest public lever | Both embryos, fixed scorer |
| E003 | Freeze-aware relinking | Local audit confirmed 947 repeated transitions in `6bba` | Detect duplicates from images, no schedule memorization |
| E004 | Motion/intensity LAP cost | Reduce dense-region track swaps | Component and edge FP controls |
| E005 | Conservative division calibration | Recover the weighted division term | Patched scorer only |
| E006 | Cross-embryo pseudo-label ensemble | Improve sparse supervision | No hidden-test probing |
| E007 | Detector ensemble or retraining | Improve embryo robustness | Embryo-disjoint promotion gate |

## Evidence Policy

Public notebook claims, local CV, Kaggle runtime completion, and leaderboard
scores are recorded separately. A notebook that merely runs is not reported as
a validated leaderboard improvement.

Public discussion observations and their required local checks are tracked in
`docs/DISCUSSION_NOTES.md`. They remain hypotheses until reproduced on the
downloaded training data.

The four visible test IDs also exist in the training directory with GEFF labels.
Those labels are not used for tuning, candidate selection, or direct graph
construction. Experiments that touch visible-test images may use only
label-independent content checks; promotion evidence must come from separate
training holdouts.
