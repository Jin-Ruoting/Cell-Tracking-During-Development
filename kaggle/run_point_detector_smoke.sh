#!/usr/bin/env bash
set -euo pipefail
run_name=${1:?Usage: run_point_detector_smoke.sh NEW_RUN_NAME}
case "$run_name" in *[!a-zA-Z0-9_-]*|'') exit 64 ;; esac
root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-python}
cd "$root"
test ! -e "logs/$run_name"
test ! -e "logs/$run_name.log"
mkdir "logs/$run_name"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4
finish() {
  status=$?
  printf '%s\n' "$status" > "logs/$run_name.done"
}
trap finish EXIT
{
  timeout 1200 "$python_bin" -u kaggle/capture_point_detector_peaks.py \
    --mode smoke --device cpu --data-dir Dataset \
    --control-dir logs/s165_e029_geometric_full_20260922v2/geffs \
    --reference-dir logs/s199_point_reference_20260923v1 \
    --runtime-dir Dataset/runtime-py311 \
    --tracking-repo logs/s165_e029_geometric_full_20260922v1/tracking_repo \
    --output-dir "logs/$run_name/cache"
  timeout 300 "$python_bin" -u kaggle/audit_point_detector_coverage.py \
    --mode smoke --data-dir Dataset --runtime-dir Dataset/runtime-py311 \
    --scorer-dir Dataset/official-scorer-075fc5f \
    --control-dir logs/s165_e029_geometric_full_20260922v2/geffs \
    --cache-dir "logs/$run_name/cache" --output-dir "logs/$run_name/coverage"
} > "logs/$run_name.log" 2>&1
