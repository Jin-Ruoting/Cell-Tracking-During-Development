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

## Excluded Paths

Discussion branches that rely on duplicated trajectories, artificial graph
connectivity, fake divisions, invalid coordinates, or other patched-metric
behavior are intentionally excluded.
