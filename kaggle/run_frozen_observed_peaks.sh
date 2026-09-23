#!/usr/bin/env bash
set -euo pipefail
arm=${1:?Usage: run_frozen_observed_peaks.sh readmit|gapfill smoke|full CACHE_RUN RUN_NAME}
mode=${2:?Provide smoke or full}
cache_run=${3:?Provide completed cache run name}
run_name=${4:?Provide a new output run name}
case "$arm" in readmit|gapfill) ;; *) exit 64 ;; esac
case "$mode" in smoke|full) ;; *) exit 64 ;; esac
for name in "$cache_run" "$run_name"; do
  case "$name" in *[!a-zA-Z0-9_-]*|'') exit 64 ;; esac
done
root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-python}
cd "$root"
test ! -e "logs/$run_name"
test "$(cat "logs/$cache_run.done")" = 0
date -Is > "logs/$run_name.started"
set +e
"$python_bin" -u kaggle/run_observed_peak_experiment.py \
  --reference-notebook logs/s163_frontier_20260922/geometric/biohub-geometric-fusion.ipynb \
  --peak-reference logs/s169_frontier_20260923v1/x138/biohub-x138.ipynb \
  --data-dir Dataset --runtime-dir Dataset/runtime-py311 \
  --scorer-dir Dataset/official-scorer-075fc5f \
  --control-dir logs/s165_e029_geometric_full_20260922v2/geffs \
  --control-csv logs/s165_e029_geometric_full_20260922v2/submission.csv \
  --replayed-control-csv logs/s171_e031_flow_full_20260923v1/control/submission.csv \
  --raw-run-dir logs/s165_e029_geometric_full_20260922v1 \
  --cache-dir "logs/$cache_run" --arm "$arm" --mode "$mode" \
  --output-dir "logs/$run_name" > "logs/$run_name.log" 2>&1
status=$?
printf '%s\n' "$status" > "logs/$run_name.done"
exit "$status"
