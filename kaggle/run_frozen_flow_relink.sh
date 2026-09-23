#!/usr/bin/env bash
set -euo pipefail
mode=${1:?Usage: run_frozen_flow_relink.sh smoke|full RUN_NAME}
run_name=${2:?Provide a unique run name}
case "$mode" in smoke|full) ;; *) exit 64 ;; esac
case "$run_name" in *[!a-zA-Z0-9_-]*|'') exit 64 ;; esac
root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-python}
cd "$root"
mkdir -p logs
test ! -e "logs/$run_name"
date -Is > "logs/$run_name.started"
set +e
"$python_bin" -u kaggle/run_flow_relink_experiment.py \
  --reference-notebook logs/s163_frontier_20260922/geometric/biohub-geometric-fusion.ipynb \
  --flow-reference logs/s169_frontier_20260923v1/x138/biohub-x138.ipynb \
  --data-dir Dataset --runtime-dir Dataset/runtime-py311 \
  --scorer-dir Dataset/official-scorer-075fc5f \
  --control-dir logs/s165_e029_geometric_full_20260922v2/geffs \
  --control-csv logs/s165_e029_geometric_full_20260922v2/submission.csv \
  --raw-run-dir logs/s165_e029_geometric_full_20260922v1 \
  --output-dir "logs/$run_name" --mode "$mode" > "logs/$run_name.log" 2>&1
status=$?
printf '%s\n' "$status" > "logs/$run_name.done"
exit "$status"
