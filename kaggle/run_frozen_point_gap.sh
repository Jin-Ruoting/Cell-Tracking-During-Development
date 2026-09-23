#!/usr/bin/env bash
set -euo pipefail
mode=${1:?Usage: run_frozen_point_gap.sh smoke|full NEW_RUN_NAME}
run_name=${2:?Provide a new run name}
case "$mode" in smoke|full) ;; *) exit 64 ;; esac
case "$run_name" in *[!a-zA-Z0-9_-]*|'') exit 64 ;; esac
root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-python}
cd "$root"
test ! -e "logs/$run_name"
test ! -e "logs/$run_name.log"
test "$(cat logs/s203_point_coverage_full_20260923v1.done)" = 0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4
export PYTHONPATH="$root/Dataset/runtime-py311${PYTHONPATH:+:$PYTHONPATH}"
date -Is > "logs/$run_name.started"
finish() { status=$?; printf '%s\n' "$status" > "logs/$run_name.done"; }
trap finish EXIT
timeout 1800 "$python_bin" -u kaggle/run_point_gap_experiment.py \
  --mode "$mode" --data-dir Dataset --runtime-dir Dataset/runtime-py311 \
  --scorer-dir Dataset/official-scorer-075fc5f \
  --control-dir logs/s165_e029_geometric_full_20260922v2/geffs \
  --control-csv logs/s165_e029_geometric_full_20260922v2/submission.csv \
  --replayed-control-csv logs/s171_e031_flow_full_20260923v1/control/submission.csv \
  --cache-dir logs/s203_point_coverage_full_20260923v1/shard0 logs/s203_point_coverage_full_20260923v1/shard1 \
  --output-dir "logs/$run_name" > "logs/$run_name.log" 2>&1
