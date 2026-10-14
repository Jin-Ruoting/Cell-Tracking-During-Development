# Data and model inputs

Attach the [competition dataset](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data) and the following datasets to the Kaggle Notebook. The input identifiers are also specified in [configs/kernel-metadata.json](../configs/kernel-metadata.json).

| Kaggle input | Role |
|---|---|
| [pilkwang/biohub-tracking-support-pack-50ep-v1](https://www.kaggle.com/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1) | Primary tracking model, supporting source code, and offline dependency wheels |
| [pilkwang/biohub-temporal-unet3d-seed314159-v1](https://www.kaggle.com/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1) | Independently seeded temporal detector and association model |
| [pilkwang/biohub-deepcenter-unet3d-center-prior-v1](https://www.kaggle.com/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1) | Auxiliary cell-center confidence model for graph repair |
| [anvithpothula/biohub-v1284-head-s075](https://www.kaggle.com/datasets/anvithpothula/biohub-v1284-head-s075) | Coordinate refinement head |

The Notebook discovers these inputs under `/kaggle/input/` and checks the required source and model hashes before inference. It reads the competition `test/` volumes and writes generated files under `/kaggle/working/`.

Use a fresh session with the recorded runtime configuration. Local copies of datasets, model weights, and generated predictions are excluded from version control.
