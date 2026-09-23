#!/usr/bin/env bash
set -euo pipefail
mode=${1:?Usage: run_frozen_detection_peaks.sh smoke|full RUN_NAME}
run_name=${2:?Provide a unique run name}
case "$mode" in smoke|full) ;; *) exit 64 ;; esac
case "$run_name" in *[!a-zA-Z0-9_-]*|'') exit 64 ;; esac
root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-python}
cd "$root"
test ! -e "logs/$run_name"
mkdir -p "logs/$run_name"
date -Is > "logs/$run_name.started"
workers=()
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$shard "$python_bin" -u kaggle/capture_frozen_detection_peaks.py \
    --raw-run-dir logs/s165_e029_geometric_full_20260922v1 \
    --data-dir Dataset --runtime-dir Dataset/runtime-py311 \
    --control-dir logs/s165_e029_geometric_full_20260922v2/geffs \
    --mode "$mode" --shard "$shard" --output-dir "logs/$run_name/shard$shard" \
    > "logs/$run_name/shard$shard.log" 2>&1 &
  workers+=("$!")
done
status=0
for pid in "${workers[@]}"; do
  if wait "$pid"; then :; else status=1; fi
done
printf '%s\n' "$status" > "logs/$run_name.done"
exit "$status"
