#!/usr/bin/env bash
set -euo pipefail

mode=${1:?Usage: run_frozen_geometric_reference.sh smoke|full RUN_NAME}
run_name=${2:?Provide a unique run name}
case "$mode" in smoke|full) ;; *) exit 64 ;; esac
case "$run_name" in *[!a-zA-Z0-9_-]*|'') exit 64 ;; esac
extra=()
if [[ -n "${3:-}" || -n "${4:-}" ]]; then
  source_run=${3:?Provide the completed source run name}
  source_sha=${4:?Provide its submission SHA256}
  case "$source_run" in *[!a-zA-Z0-9_-]*|'') exit 64 ;; esac
  [[ "$source_sha" =~ ^[0-9a-f]{64}$ ]] || exit 64
  extra=(--completed-run-dir "logs/$source_run" --source-submission-sha256 "$source_sha")
fi
root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-python}
cd "$root"
mkdir -p logs
test ! -e "logs/$run_name"
date -Is > "logs/$run_name.started"
set +e
"$python_bin" -u kaggle/run_geometric_reference.py \
  --reference-notebook logs/s163_frontier_20260922/geometric/biohub-geometric-fusion.ipynb \
  --data-dir Dataset \
  --control-dir logs/s157_e026_retention_ab_20260731v1/post_blend/score_e025_exact/geffs \
  --runtime-dir Dataset/runtime-py311 \
  --scorer-dir Dataset/official-scorer-075fc5f \
  --output-dir "logs/$run_name" \
  --mode "$mode" "${extra[@]}" > "logs/$run_name.log" 2>&1
status=$?
printf '%s\n' "$status" > "logs/$run_name.done"
exit "$status"
