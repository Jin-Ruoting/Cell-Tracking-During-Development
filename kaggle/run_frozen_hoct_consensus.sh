#!/usr/bin/env bash
set -euo pipefail
mode=${1:?Usage: run_frozen_hoct_consensus.sh smoke|full RUN_NAME}
run_name=${2:?Provide a new run name}
case "$mode" in smoke|full) ;; *) exit 64 ;; esac
case "$run_name" in *[!a-zA-Z0-9_-]*|'') exit 64 ;; esac
root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-python}
cd "$root"
test ! -e "logs/$run_name"
test ! -e "logs/$run_name.log"
export PYTHONPATH="$root/logs/s193_hoct_sized_runtime_20260923v1/runtime:$root/Dataset/runtime-py311:$root/Dataset/official-scorer-075fc5f/src:$root/Dataset/official-scorer-075fc5f/scripts"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4
finish() {
  status=$?
  printf '%s\n' "$status" > "logs/$run_name.done"
}
trap finish EXIT
"$python_bin" -u kaggle/run_hoct_consensus_experiment.py run \
  --data-dir Dataset --runtime-dir Dataset/runtime-py311 \
  --scorer-dir Dataset/official-scorer-075fc5f \
  --control-dir logs/s165_e029_geometric_full_20260922v2/geffs \
  --control-csv logs/s165_e029_geometric_full_20260922v2/submission.csv \
  --replayed-control-csv logs/s171_e031_flow_full_20260923v1/control/submission.csv \
  --mode "$mode" --output-dir "logs/$run_name" > "logs/$run_name.log" 2>&1
