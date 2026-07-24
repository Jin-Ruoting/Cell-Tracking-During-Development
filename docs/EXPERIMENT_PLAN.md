# Experiment Plan

## Objective

Reach a Biohub public leaderboard score strictly greater than `0.92` with a
legitimate, reproducible pipeline, then protect against public/private
leaderboard shift by validating across both training embryos.

The earlier top-10% goal was achieved on 2026-07-24. The current 12:42 CST
snapshot has 1,579 teams: E000 scores `0.908` at rank 102, while only scores
`0.931`, `0.929`, and `0.923` are strictly above `0.92`. A promoted candidate
must therefore target at least the displayed score `0.921`, not `0.920`.

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
| E003 | Freeze-aware relinking | Zero-motion LAP sweep rejected: no disjoint-holdout gain | Detect duplicates from images, no schedule memorization |
| E004 | Motion/intensity LAP cost | Reduce dense-region track swaps | Component and edge FP controls |
| E005 | Supervised conservative division recovery | Recover at least `+0.013` without sacrificing the edge term | Current patched scorer, embryo-disjoint promotion |
| E006 | Cross-embryo pseudo-label ensemble | Improve sparse supervision | No hidden-test probing |
| E007 | Detector ensemble or retraining | Improve embryo robustness | Embryo-disjoint promotion gate |

E005 is now the priority. The strongest inspected public dual-seed router scores
only `0.909`, and the patched discussion frontier reports edge-only approaches
plateauing around `0.90`-`0.91`. Since the division term is weighted by `0.1`,
a division Jaccard near `0.13` can in principle close the current gap if the
adjusted edge score is preserved. This is a hypothesis until the official
patched scorer confirms it on holdouts from both embryos.

The first E005 calibration pass selects up to 16 division-positive clips per
embryo and an equal number of division-negative clips matched by annotated edge
count. It excludes all four visible-test IDs, balances the selected clips
across two GPUs, and preserves both pre-ILP candidate graphs and baseline ILP
graphs. The pre-ILP export permits multiple division-cost and conservative-fork
sweeps from one neural inference pass. Promotion still requires validation on
clips that were not used to select the rule.

The first ILP sweep holds detector, candidate edges, appearance cost, and
disappearance cost fixed while varying only division cost through
`0.75, 0.5, 0.25, 0.0, -0.25`; the original `1.0` graph is the control. Every
candidate is scored directly from GEFF with the pinned patched scorer.

The serial lower-cost ILP path was stopped after its projected runtime reached
approximately 12 hours before scoring. The replacement sweep adds at most one
edge from a one-child parent to a root at the next frame. Five explicit rules
cover broad pre-ILP support, balanced geometry, high edge confidence, strict
daughter symmetry, and a geometry-only control. Every rule preserves indegree
one and outdegree at most two by construction.

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
