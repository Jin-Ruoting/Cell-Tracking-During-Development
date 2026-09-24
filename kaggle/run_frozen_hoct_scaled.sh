#!/usr/bin/env bash
set -euo pipefail
mode=${1:?Usage: run_frozen_hoct_scaled.sh smoke|full RUN_NAME [SMOKE_DIR]}
run_name=${2:?Provide a new run name}
case "$mode" in smoke|full) ;; *) exit 64 ;; esac
case "$run_name" in *[!a-zA-Z0-9_-]*|'') exit 64 ;; esac
root=${BIOHUB_ROOT:-$(pwd)}
cd "$root"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(git rev-parse '@{upstream}')"
test ! -e "logs/$run_name"
test ! -e "logs/$run_name.log"
extra=()
if [ "$mode" = full ]; then extra=(--smoke-dir "${3:?Full mode requires completed smoke directory}"); fi
export PYTHONPATH="$root/logs/s193_hoct_sized_runtime_20260923v1/runtime:$root/Dataset/runtime-py311:$root/Dataset/official-scorer-075fc5f/src:$root/Dataset/official-scorer-075fc5f/scripts"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
finish() {
  status=$?
  printf '%s\n' "$status" > "logs/$run_name.done"
  if [ "$status" = 124 ]; then
    mkdir -p "logs/$run_name"
    printf '# E041 timed out\n\nNo completed comparison or promotion.\n' > "logs/$run_name/run_summary.md"
  fi
}
trap finish EXIT
budget=3600
if [ "$mode" = full ]; then budget=14400; fi
{
python -m unittest discover -s tests -p test_hoct_compatibility.py -v
timeout "$budget" python -u kaggle/run_hoct_scaled_consensus.py run \
  --data-dir Dataset --runtime-dir Dataset/runtime-py311 \
  --scorer-dir Dataset/official-scorer-075fc5f \
  --control-dir logs/s165_e029_geometric_full_20260922v2/geffs \
  --control-csv logs/s165_e029_geometric_full_20260922v2/submission.csv \
  --replayed-control-csv logs/s171_e031_flow_full_20260923v1/control/submission.csv \
  --mode "$mode" --output-dir "logs/$run_name" "${extra[@]}"
} > "logs/$run_name.log" 2>&1
