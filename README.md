# Biohub — Cell Tracking During Development

This repository presents the final inference workflow for the [Biohub Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) competition. The task is to localize cells in three-dimensional time-lapse microscopy and reconstruct their trajectories and division lineages during embryonic development.

## Method

The workflow combines two pretrained temporal 3D U-Net detectors through model fusion and spatial test-time augmentation. A coordinate refinement head adjusts the detected cell centers, after which a node Transformer estimates association probabilities between adjacent frames. Forward and reverse association predictions are fused to form candidate temporal links.

Integer linear programming (ILP) selects an initial lineage graph with at most one parent and two children per node. Subsequent processing applies motion-based reassociation, detection readmission, gap completion, division repair, and trajectory smoothing. The final output is checked for coordinate validity and temporal graph consistency before CSV export.

The final **E048** configuration uses a detection threshold of **0.965**, a readmission threshold of **0.94**, and an ILP division cost of **1.2**.

## Results

Team **Ruoting** achieved a final private leaderboard score of **0.91863**, ranking **509 of 3,947 teams (top 12.90%)**.

| Configuration | Submission ID | Public score | Private score |
|---|---:|---:|---:|
| E048 — final leaderboard result | 56673115 | 0.95508 | **0.91863** |
| E047 — best public score | 56673114 | **0.95605** | 0.91792 |

The final rank is determined by the private leaderboard. The best public score belongs to a separate configuration. These results were verified against official Kaggle records on **14 October 2026**; submission identifiers and artifact checksums are recorded in [results/leaderboard.json](results/leaderboard.json).

## Repository structure

```text
.
├── README.md
├── configs/
│   └── kernel-metadata.json     # Kaggle inputs and execution environment
├── data/
│   └── README.md                # Required data and pretrained model inputs
├── notebooks/
│   └── e048_inference.ipynb     # Final inference and output validation
└── results/
    └── leaderboard.json         # Verified scores and submission identifiers
```

## Reproduction

1. Create a Kaggle Notebook and import [notebooks/e048_inference.ipynb](notebooks/e048_inference.ipynb).
2. Attach the competition data and the four inputs listed in [data/README.md](data/README.md). The [runtime configuration](configs/kernel-metadata.json) records the input identifiers and original Docker image.
3. Select **GPU T4 ×2**, disable Internet access, and execute the cells sequentially. Model weights, tracking dependencies, and offline wheels are loaded from the attached inputs.

The Notebook writes `submission.csv` to `/kaggle/working/` and validation records to `/kaggle/working/logs/e048-public953/`. Its nine code cells are identical to those in the submitted E048 version 1. The repository contains the inference workflow; competition data and pretrained weights are obtained through the specified Kaggle inputs.
