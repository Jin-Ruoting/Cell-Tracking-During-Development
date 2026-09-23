#!/usr/bin/env bash
set -euo pipefail
run_name=${1:?Usage: run_point_detector_full.sh NEW_RUN_NAME}
case "$run_name" in *[!a-zA-Z0-9_-]*|'') exit 64 ;; esac
root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-python}
cd "$root"
test ! -e "logs/$run_name"
test ! -e "logs/$run_name.log"
"$python_bin" -c 'import json; from pathlib import Path; r=Path("logs/s202_point_real_smoke_20260923v1"); assert Path(str(r)+".done").read_text().strip()=="0"; assert json.loads((r/"cache/run_manifest.json").read_text())["complete"] is True; assert (r/"coverage/coverage.json").is_file()'
mkdir "logs/$run_name"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 POLARS_MAX_THREADS=4
finish() {
  status=$?
  printf '%s\n' "$status" > "logs/$run_name.done"
}
trap finish EXIT
{
  workers=()
  for shard in 0 1; do
    CUDA_VISIBLE_DEVICES=$shard timeout 1800 "$python_bin" -u kaggle/capture_point_detector_peaks.py \
      --mode full --device cuda --shard "$shard" --data-dir Dataset \
      --control-dir logs/s165_e029_geometric_full_20260922v2/geffs \
      --reference-dir logs/s199_point_reference_20260923v1 \
      --runtime-dir Dataset/runtime-py311 \
      --tracking-repo logs/s165_e029_geometric_full_20260922v1/tracking_repo \
      --output-dir "logs/$run_name/shard$shard" > "logs/$run_name/shard$shard.log" 2>&1 &
    workers+=("$!")
  done
  status=0
  for pid in "${workers[@]}"; do
    if wait "$pid"; then :; else status=1; fi
  done
  if [ "$status" -ne 0 ]; then
    printf '# V5 full diagnostic failed\n\nAt least one prediction shard did not complete; no partial-corpus audit.\n' > "logs/$run_name/run_summary.md"
    exit "$status"
  fi
  timeout 900 "$python_bin" -u kaggle/audit_point_detector_coverage.py \
    --mode full --data-dir Dataset --runtime-dir Dataset/runtime-py311 \
    --scorer-dir Dataset/official-scorer-075fc5f \
    --control-dir logs/s165_e029_geometric_full_20260922v2/geffs \
    --cache-dir "logs/$run_name/shard0" "logs/$run_name/shard1" \
    --e029-cache-dir logs/s179_peak_capture_full_20260923v1 \
    --raw-run-dir logs/s165_e029_geometric_full_20260922v1 \
    --output-dir "logs/$run_name/coverage"
  printf '# V5 full diagnostic complete\n\nAll 64 movies predicted before labels were read. See coverage/run_summary.md for geometric opportunity only; no tracking score or submission.\n' > "logs/$run_name/run_summary.md"
} > "logs/$run_name.log" 2>&1
