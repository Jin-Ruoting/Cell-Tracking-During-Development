#!/usr/bin/env bash
set -euo pipefail
run_name=${1:?Provide a unique run name}
case "$run_name" in *[!a-zA-Z0-9_-]*|'') exit 64 ;; esac
test ! -e "logs/$run_name"
mkdir -p logs
date -Is > "logs/$run_name.started"
set +e
python -u kaggle/fuse_division_evidence.py \
  --base-csv logs/s157_e026_retention_ab_20260731v1/post_blend/e025_exact.csv \
  --reference-csv logs/s165_e029_geometric_full_20260922v2/submission.csv \
  --data-dir Dataset --runtime-dir Dataset/runtime-py311 \
  --scorer-dir Dataset/official-scorer-075fc5f \
  --output-dir "logs/$run_name" > "logs/$run_name.log" 2>&1
status=$?
printf '%s\n' "$status" > "logs/$run_name.done"
exit "$status"
