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

Evidence status: participant-reported; local verification is pending completion
of the competition-data download.

The committed `scripts/audit_frozen_frames.py` performs an exact encoded-chunk
comparison without loading or decompressing the full 3D volumes. It validates the
dataset's Zarr v3 chunk layout before comparing every consecutive transition.
Byte equality is a conclusive duplicate witness; the audit also records the
comparison method so that decoded-array confirmation can be added if the dataset
uses non-deterministic encoding.

E003 promotion gates:

1. Recompute exact consecutive-frame equality for all 199 training clips with
   `scripts/audit_frozen_frames.py`.
2. Report results separately for `44b6` and `6bba`.
3. Detect frozen transitions from image content at inference time; never encode a
   memorized frame schedule.
4. Compare unchanged E001 tracking against freeze-aware linking on both embryos.
5. Reject the change if gains are confined to one embryo or edge false positives
   rise materially.

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
