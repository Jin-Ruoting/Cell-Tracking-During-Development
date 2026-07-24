# Discussion-Derived Experiment Notes

These notes convert public Kaggle discussions into testable experiments. A forum
observation is not treated as a verified fact until it is reproduced on the
downloaded training data with the patched official-spec scorer.

## Scoring Latency and Graph Structure

Sources:

- [Submission Stuck, topic 726526](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/726526)
- [Scoring timeouts: evidence it is graph connectivity, not size, topic 724917](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/724917)

Participant reports describe leaderboard evaluation taking roughly four to seven
hours. Another participant hypothesizes that a small number of long connected
components can be slower to score than the same number of nodes split across many
short components.

Evidence status: participant-reported, not official and not locally reproduced.

Operational decision:

- Do not duplicate a submission merely because it remains `PENDING` for minutes.
- Use the retained event watcher and act only on a terminal Kaggle status.
- For every candidate, record node count, edge count, component count, largest
  component, and edge-to-node ratio before submission with
  `scripts/audit_submission.py`.

## Frozen Frames and Global Jumps

Source:

- [Beware of jumps in ground truth track, topic 724283](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/724283)

The thread reports exact consecutive-frame duplicates concentrated in embryo
`6bba`: 947 duplicate pairs across 114 of 128 clips, with no exact duplicates
reported in the 71 `44b6` clips. It also reports occasional global spatial jumps.

Evidence status: locally reproduced on 2026-07-23 with exact encoded-chunk
comparison. Across all 19,701 adjacent transitions, the audit found:

- `44b6`: 71 clips, 0 affected clips, 0 duplicate transitions;
- `6bba`: 128 clips, 114 affected clips, 947 duplicate transitions;
- total: 199 clips, 114 affected clips, 947 duplicate transitions.

The same content-driven audit compared all 396 adjacent transitions in the four
visible test arrays. It found no duplicate transition in either `44b6` clip and
20 duplicates across the two `6bba` clips. The visible test IDs also occur under
`train/`; their GEFF labels are explicitly excluded from parameter selection and
must never be copied into a submission.

The committed `scripts/audit_frozen_frames.py` performs an exact encoded-chunk
comparison without loading or decompressing the full 3D volumes. It validates the
dataset's Zarr v3 chunk layout before comparing every consecutive transition.
Byte equality is a conclusive duplicate witness; the audit also records the
comparison method so that decoded-array confirmation can be added if the dataset
uses non-deterministic encoding.

E003 promotion gates:

1. Detect frozen transitions from image content at inference time; never encode a
   memorized frame schedule.
2. Use `scripts/analyze_frozen_edges.py` to measure detection overlap and edge
   displacement on frozen versus ordinary fixed-eight transitions.
3. Tune only on training holdouts whose IDs are absent from the visible test
   directory.
4. Use `scripts/relink_frozen_transitions.py` for a zero-motion LAP ablation and
   `scripts/evaluate_edge_predictions.py` for the patched-spec edge term.
5. First reproduce E001's fixed-eight edge counts exactly, then compare unchanged
   E001 tracking against freeze-aware linking on disjoint holdouts.
6. Reject the change if gains are confined to one embryo, edge false positives
   rise materially, or graph validation fails.

Result: the evaluator reproduced E001 exactly. Seven zero-motion LAP gates from
0.5 to 6.0 micrometres were then tested on four training holdouts whose IDs are
absent from the visible test directory. Gates through 3.0 micrometres reduced the
adjusted edge score, while 4.0 and 6.0 merely tied E001. All graphs were valid.
This E003 branch is rejected and is not a submission candidate.

## Detection and Temporal Affinity

Source:

- [Temporal Affinity Fields for 3D Cell Lineage Reconstruction, topic 723655](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/723655)

The proposed direction is to predict local motion or temporal affinity rather
than only pairwise links. The discussion also suggests consensus pseudo-labels
from multiple short-horizon trackers because annotated links are sparse, and
notes that unusually long predicted links are often wrong.

Evidence status: research hypothesis.

Experiment order:

1. E002: tune the existing detector threshold with fixed tracking and scorer.
2. E003: add image-detected freeze awareness to the current linker.
3. E004: add motion/intensity costs and inspect the longest links.
4. E006/E007: consider consensus pseudo-labels or temporal affinity only after
   the lower-cost controls have been evaluated across both embryos.

## Patched Division Frontier

Sources:

- [Division Metric exploit and patch, topic 727154](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/727154)
- [COMPLETED: Rescore Underway, topic 728324](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/728324)
- [Score locally and why a clean prediction can exceed 1.0, topic 728300](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/728300)
- [Post-patch 0.91+ frontier, topic 728551](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/728551)
- [Official metric specification](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/075fc5f5a52d11077f9dc2b074644618f26939e2/metrics.md)

Kaggle staff reported the rescore complete. The patched official metric requires
a local, directed parent-to-two-daughter topology within the division window;
weakly connected hubs and remote fake forks no longer qualify. The score remains
`adjusted_edge_jaccard + 0.1 * division_jaccard`.

A participant reports that common post-patch threshold, TTA, gap-closing, and
ILP variations plateau around adjusted edge Jaccard `0.90`-`0.91`, while the
leading public scores reach `0.923`-`0.931`. The same thread asks whether that
gap comes from a genuinely stronger edge model or division Jaccard around
`0.15`-`0.20`; it does not provide verified decomposition.

Local evidence:

- Kaggle's score-sorted public kernels contain many pre-patch exploit notebooks,
  so ordering alone is not promotion evidence.
- Pilkwang Kim's legitimate dual-seed confidence router was inspected and
  independently graph-audited. Its Kaggle page reports Public Score `0.909`,
  so a second temporal seed alone does not meet the current target.
- Xiaolei Lian's `mix-divaug` notebook contains an explicitly labelled
  negative-time, out-of-volume hub/fork exploit. It is excluded.
- The project pins the current official scorer commit
  `075fc5f5a52d11077f9dc2b074644618f26939e2` before evaluating E005.

E005 promotion gates:

1. Derive candidate rules or learned features only from training clips.
2. Use holdouts from both `44b6` and `6bba`, excluding all visible-test IDs.
3. Score edge and division TP/FP/FN with the current official patched code.
4. Require a projected combined gain of at least `+0.013`, valid coordinates,
   and no material adjusted-edge regression before spending a submission.
5. Never add negative-time nodes, out-of-volume nodes, artificial graph hubs,
   remote forks, or graph structures intended only to manipulate the metric.

## Excluded Paths

Discussion branches that rely on duplicated trajectories, artificial graph
connectivity, fake divisions, invalid coordinates, or other patched-metric
behavior are intentionally excluded.
