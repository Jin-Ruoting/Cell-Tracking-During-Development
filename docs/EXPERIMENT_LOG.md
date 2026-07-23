# Experiment Log

All times use Asia/Shanghai unless stated otherwise. Raw logs remain on the
server and are never committed.

## S001 - Initial Server and Competition Audit

- Time: 2026-07-23
- Server repository directory existed but was empty.
- Intended `Dataset/` and `logs/` directories were not yet present.
- Conda environment `Kaggle`: Python 3.11.15, Kaggle CLI 2.2.3.
- GPUs: two NVIDIA GeForce RTX 4090 cards, each with 24,564 MiB total memory.
- Screen state: one detached `Kaggle` session (`2615526.Kaggle`); no extra
  sessions were created or removed.
- Kaggle account had accepted the competition rules and had no submissions.
- The public clean reference notebook was pulled only into server `/tmp` for
  read-only inspection; it was not placed in the server repository.
- Reference run evidence: four example test videos, 9.26 minutes prediction,
  236,203 output rows (120,246 nodes and 115,957 edges), and fixed-eight
  official-spec lite CV score `0.87892959136423`.

Status: audit complete; no project source was edited on the server.

## E000 - Clean Baseline Reproduction

- Status: pending
- Candidate: `buaaauto/biohub-clean-baseline-no-hack`
- Source: attributed public no-exploit notebook.
- Expected clean public baseline reference: approximately `0.908`; this is not
  yet our verified score.

## S002 - Server Repository Bootstrap

- Time: 2026-07-23
- Cloned the public GitHub repository into the previously empty server project
  directory at local/GitHub commit `788f4083b6f34220004001af0cce4c1617cfa57b`.
- Created the ignored `Dataset/` and `logs/` runtime directories.
- Verified `main...origin/main` with no tracked or untracked repository changes.
- Source edits on server: none.

Status: server is ready for future `git pull --ff-only` synchronization.
