#!/usr/bin/env bash
set -euo pipefail
run_name=${1:?Usage: run_hoct_compatibility.sh RUN_NAME GPU}
gpu=${2:?Specify an idle GPU index}
case "$run_name" in *[!a-zA-Z0-9_-]*|'') exit 64 ;; esac
root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-python}
cd "$root"
test ! -e "logs/$run_name"
test ! -e "logs/$run_name.log"
export PYTHONPATH="$root/logs/s193_hoct_sized_runtime_20260923v1/runtime:$root/Dataset/runtime-py311"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4
finish() {
  status=$?
  mkdir -p "logs/$run_name"
  printf '%s\n' "$status" > "logs/$run_name/exit_code.txt"
  if [ "$status" = 124 ]; then
    printf '# HOCT probe timeout\n\nReached the 900-second budget; no successful compatibility or quality claim.\n' > "logs/$run_name/run_summary.md"
  fi
}
trap finish EXIT
timeout 900 "$python_bin" -u kaggle/probe_hoct_consensus.py \
  --data-dir "$root/Dataset" \
  --control-csv "$root/logs/s165_e029_geometric_full_20260922v2/submission.csv" \
  --output-dir "$root/logs/$run_name" > "logs/$run_name.log" 2>&1
