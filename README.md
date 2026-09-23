# Biohub Cell Tracking During Development

Reproducible, no-exploit inference and graph-reconstruction workflow for the
Kaggle research competition
[Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

This repository contains clean single-seed and dual-seed tracking pipelines,
plus attributed external-method comparisons. The executable E029 method has
achieved a verified public score of **`0.947`**.

## Verified Results

| Experiment | Public submission | Public score | Interpretation |
|---|---:|---:|---|
| E000 clean single-seed baseline | `54923913` | `0.908` | Verified single-seed baseline |
| E016 embryo-aware dual-seed router | `54972789` | `0.908` | Stable, but no leaderboard gain |
| E025 guarded global dual-seed method | `55023652` | `0.912` | Verified `+0.004` over E000/E016 |
| E029 frozen geometric reference | `56462196` | **`0.947`** | Verified `+0.035` over E025 |

The E025 score belongs to this repository's submitted Kernel version 1:
[biohub-e025-guarded-dual-seed-center-gaps](https://www.kaggle.com/code/buaaauto/biohub-e025-guarded-dual-seed-center-gaps?scriptVersionId=338254608).

The E029 score belongs to Kernel version 1 of
[biohub-e029-frozen-geometric-reference](https://www.kaggle.com/code/buaaauto/biohub-e029-frozen-geometric-reference).
The full public leaderboard snapshot at `2026-09-23 03:47 UTC` places the team
at **1,001 of 3,820 teams (top 26.20%)**. The top-10% cutoff is rank 382;
many teams share the displayed score `0.947`, so the target remains unmet.

## Historical Baseline Leaderboard Snapshot

![Kaggle public leaderboard snapshot showing Ruoting at rank 135 with a public score of 0.912](assets/kaggle-public-leaderboard-2026-07-28.png)

*Captured on 2026-07-28. The leaderboard is dynamic; this screenshot records
the displayed rank and score at capture time.*

## Shared Tracking Pipeline

The methods use pretrained inference artifacts rather than training
models inside the submission Notebook:

```text
3D+t microscopy
  -> TemporalUNet3D detection field and features
  -> spatial D4 test-time augmentation
  -> cell-center point extraction
  -> node-transformer adjacent-frame association scores
  -> constrained ILP lineage construction
  -> motion, gap, fragment, smoothing, and division repair
  -> topology audit
  -> submission.csv
```

The graph formulation is central to the approach. Learned detections and edge
scores provide proposals, while the ILP and postprocessor enforce temporal and
biological consistency across the complete lineage.

## Method A: Clean Single-Seed Baseline (`0.908`)

The clean baseline combines single-seed detection, learned adjacent-frame
association, constrained lineage optimization, and conservative graph repair.

### Detection and association

- A TemporalUNet3D processes two-frame windows and produces a volumetric
  center field plus features for association.
- Detection logits are averaged over the eight spatial symmetries of the
  `x-y` plane (D4 TTA).
- Center points are extracted at detection threshold `0.96875`.
- A node transformer scores candidate links between cells in consecutive
  frames using image features, physical coordinates, and relative position.

### Lineage optimization

The initial graph is selected by a constrained ILP with edge weight `-1.0`,
appearance cost `0.0`, disappearance cost `1.575`, and division cost `1.0`.
Edges must advance exactly one frame, each node may have at most one parent,
and binary division permits at most two children.

### Graph repair

- Motion-aware bipartite reassignment uses tight and relaxed gates of
  `6.0/9.5 um`.
- One-missing-frame gaps may be repaired by reusing an observed point or
  inserting and refining a synthetic midpoint.
- Components shorter than six nodes are normally removed, while components
  containing a division are retained.
- The public revision tested a conservative rescue for exactly five-node
  components with mean edge probability at least `0.90`, mean displacement at
  most `2.75 um`, and a maximum budget of 60 nodes.
- Linear track interiors are smoothed without changing graph topology, and
  tightly capped geometric repairs may add a second child for a plausible
  division.

The rescue branch recovered 12 components and 60 nodes on the hidden test, but
its fixed-eight local validation score was slightly below its reference.
Accordingly, `0.908` is treated as evidence for the complete clean tracking
pipeline, not as evidence that short-track rescue alone improved the metric.

## Method B: Guarded Dual-Seed Method (`0.912`)

E025 extends the clean baseline with globally calibrated dual-seed detection,
low-margin link consensus, and center-confirmed gap repair, while retaining
fail-closed artifact checks and a binary safe-division guard.

### Shared dual-seed detections

The primary and independently seeded temporal models both run spatial D4 TTA.
For each frame, the secondary detection field is aligned to the primary
field's mean and standard deviation before blending:

```text
shared_detection = 0.525 * primary + 0.475 * aligned_secondary
```

One shared point set is extracted from this fused field. Both node transformers
therefore score the same physical cells rather than producing two incompatible
coordinate sets.

### Low-margin link consensus

Secondary edge evidence is used only when the primary association is
uncertain:

- maximum primary top-two margin: `0.35`;
- maximum secondary edge weight: `0.15`;
- candidate edge threshold after calibration: `0.48`;
- secondary evidence is applied only when both models select the same best
  parent;
- disagreement leaves the primary edge score unchanged.

This preserves confident primary links while using independent evidence on
ambiguous associations.

### Global lineage and center-confirmed gaps

- ILP disappearance cost is reduced from `1.575` to `1.5`.
- Motion reassignment uses `6.0/10.0 um` tight and relaxed gates.
- DeepCenterUNet3D is not a second global detector. It only confirms newly
  synthetic one-frame gap midpoints whose endpoint span is at least `8.5 um`.
- The center threshold is `0.25`, the required checkpoint epoch is `500`, and
  observed or shorter-gap points bypass the gate.
- DeepCenter does not change accepted transformer edges or division
  decisions, and its safe-division veto is disabled.

In the scored E025 run, DeepCenter checked 263 long synthetic midpoint
proposals and rejected all 263. It therefore acted as a precision gate against
unsupported gap filling rather than adding new detections.

## E025 Execution Evidence

Kernel version 1 completed inference in `9.12` minutes and produced:

| Artifact statistic | Value |
|---|---:|
| Test datasets | `4` |
| Node rows | `119,039` |
| Edge rows | `114,863` |
| Total submission rows | `233,902` |
| Submission SHA256 | `78598f236bee33d2228096f4a4c19286e9a53cd49f2bcdf9eca2c652283fea3d` |

The downloaded output independently passed sequential-row, schema,
coordinate, dangling-edge, duplicate-edge, temporal-edge, maximum-indegree-one,
and maximum-outdegree-two checks. The primary, independent-seed, and
DeepCenter model files were verified from their materialized bytes before
inference.

The `+0.004` public improvement is the result of a combined configuration:
global dual-seed detection blending, low-margin link consensus, ILP and motion
calibration, and center-gated gap repair. Because these changes were submitted
together, the leaderboard result does not establish the isolated causal
contribution of any one component.

## Method C: Frozen External Reference (`0.947`)

E029 reproduces [Aman Atar's Geometric Fusion](https://www.kaggle.com/code/amanatar/biohub-geometric-fusion),
which extends [Igor Zharov's Harmonic Fusion](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion),
using the original pretrained artifacts by Pilkwang Kim. The external Notebook
is acquired separately and verified by SHA256; its training-label validator
and parameter sweep are excluded from this submission implementation.

The frozen method combines a secondary detection weight of `0.80`, aligned
eight-view association features for both seeds, secondary feature weight
`0.75`, and forward/reverse harmonic association with reverse weight `0.15`.
DeepCenter uses the verified epoch-2 checkpoint and confirms geometric division
proposals. The author's seven published postprocessing overrides are fixed.
An independent export check handles one-voxel upper-bound rounding and rejects
larger excursions or invalid lineage topology.

The pinned official scorer produced these development results:

| Evaluation set | E025 control | E029 | Difference |
|---|---:|---:|---:|
| Fixed 64 movies | `0.901433` | `0.909044` | `+0.007611` |
| 59 movies outside the author's parameter-selection set | `0.901729` | `0.908376` | `+0.006647` |

Both embryo groups and both alternating halves improved in overall score.
The full-corpus adjusted-edge component decreased by `0.001032`, while division
Jaccard increased from `0.029851` to `0.116279`. E029 therefore failed the
original all-component promotion rule. It was selected for an exploratory
public-score test under an explicitly recorded overall-score policy, with
that failed component retained in the evidence. These development comparisons
use public training movies and fixed pretrained checkpoints; they are not
training-disjoint model validation.

Private Kernel version 1 completed both GPU shards and produced 241,311 rows
(122,764 nodes and 118,547 edges) across four test movies. Downloaded output
passed an independent coordinate/topology audit and exact reexport check.
Its SHA256 is
`4254dbc610cb53b262c8fbd854b6696a3bf01461a18a61aa0eaf6b0fc38c015d`.
The independently generated output matches the reference author's selected
output byte for byte. The reference output was downloaded for this comparison
only after E029's runtime output had been audited and submitted.

## Development Candidates After E029

E031 changes only motion association, using the neighborhood-flow function
from [Anvith Pothula's x138](https://www.kaggle.com/code/anvithpothula/biohub-x138).
The original E029 raw detections are reused and the complete control CSV must
reproduce byte for byte before comparison. The x138 coordinate head and other
graph changes are excluded.

| Candidate | E029 control | Candidate score | Decision |
|---|---:|---:|---|
| E031 neighborhood flow, 64 movies | 0.909044 | 0.920862 | All frozen development gates passed |
| E031 excluding 5 author-selection movies | 0.908376 | 0.921565 | Both embryo groups also improved |
| E032 cross-fitted coordinate calibration, 64 movies | 0.909044 | 0.909374 | Rejected: small pooled gain and group regressions |
| E033 high-confidence detection readmission, 64 movies | 0.909044 | 0.908002 | Rejected: regressions in both embryos and halves |
| E034 observed-peak gap filling, 64 movies | 0.909044 | 0.909083 | Rejected: negligible gain, 4 affected wins / 49 losses |
| E035 per-track motion EMA, 64 movies | 0.909044 | 0.910636 | All frozen development gates passed; packaging checks pending |

E031 improves both embryo groups, both alternating halves, and the adjusted
edge term. Its 64 paired movies contain 37 wins, 26 losses, and one tie.
Private Kernel version 1 completed and its 241,653-row output passed an
independent configuration, coordinate/topology, and exact-reexport audit.
Formal submission `56483615` was accepted on `2026-09-23 05:16 UTC` and is
awaiting scoring; **no E031 public score is established yet**.

E032 preserves all E029 nodes and edges and cross-fits a bounded image-feature
ridge regressor on two disjoint sets of 32 movies. Its gain falls below the
fixed 0.001 threshold; one embryo and one half regress. It is not selected for
deployment. All comparisons remain adaptive public-training development
evidence; cross-fitting the new regressor does not make the pretrained
detector or earlier method selection training-disjoint.

The [2026-09-23 method review](docs/METHOD_REVIEW_20260923.md) records
source-verified candidates for observed-peak recovery, a complementary point
detector, and HOCT consensus filtering, with reproduction gaps and proposed
validation gates. The [parallel E033/E034/E035 protocol](docs/PARALLEL_EXPERIMENTS_20260923.md)
freezes separate detection-readmission, observed-gap and motion-EMA comparisons
while E031 scoring is pending. Weak-peak capture reproduced the original E029
pre-ILP detection-coordinate hashes on all 64 movies. E033 and E034 failed
the frozen advancement gates. E035 completed with positive embryo/half gains
and 34 affected wins, 18 losses and eight ties; four graphs were unchanged.
It advances separately to private Kernel packaging and actual-output checks.
No E035 public score is established.

## Reproducibility

- No hidden-test labels, metric exploits, artificial hubs, negative-time
  nodes, out-of-volume nodes, or cross-dataset edges are used.
- Model checkpoints and external Notebook states are checksum-pinned.
- The submission Notebook fails closed when required artifacts or topology
  contracts do not match.
- Searched divisions and label-dependent dataset routing are disabled in E025.
- The embedded candidate-status cell records the pre-submission state of the
  executed Notebook; the terminal score is the verified Kaggle result reported
  above.

## Public Files

- `kaggle/biohub_clean_baseline.ipynb`: complete executable E025 Notebook.
- `kaggle/kernel-metadata.json`: accepted Kaggle Kernel configuration.
- `kaggle/validate_e006_postprocess.py`: controlled postprocessing validator,
  including the pinned dual-seed parity mode.
- `kaggle/audit_deepcenter_rescue.py`: sparse-label detector-complement audit.
- `kaggle/run_dual_seed_control.py`: pinned independent-seed inference control.
- `kaggle/audit_hoct_rerank.py`: HOCT edge-ranking compatibility audit.
- `kaggle/audit_e000_error_budget.py`: official-matcher error-budget audit.
- `kaggle/run_geometric_reference.py`: frozen, attributed external-method
  comparison against E025 using the official scorer.
- `kaggle/build_geometric_submission.py`: private Notebook packaging gated
  on a complete 64-movie comparison, with explicit recording of any exploratory
  component tradeoff; packaging is not score evidence.
- `kaggle/fuse_division_evidence.py`: prediction-only division correspondence
  on fixed base detections, with a checksum-pinned evaluation launcher.
- `kaggle/watch_competition_submission.py`: read-only terminal-score and full
  leaderboard verification for an existing submission; it never submits.
- `kaggle/run_flow_relink_experiment.py` and `kaggle/build_flow_submission.py`:
  frozen E031 comparison and packaging, with exact-control and promotion checks.
- `kaggle/audit_flow_kernel_output.py`: independent downloaded-output,
  configuration, topology, and exact-reexport checks before an E031 submission.
- `kaggle/run_coordinate_calibration_experiment.py`: the two-fold E032
  coordinate-only comparison with fixed topology and displacement bounds.
- `NOTICE.md`: third-party component and competition-resource notice.
