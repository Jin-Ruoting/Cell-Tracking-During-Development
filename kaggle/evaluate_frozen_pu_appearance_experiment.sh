#!/usr/bin/env bash
set -u

root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-/home/zqjinruoting/anaconda3/envs/Kaggle/bin/python}
run_name=s162_e028_pu_appearance_20260731v1
run_done="$root/logs/$run_name.done"

cd "$root" || exit 1
if [[ ! -f "$run_done" ]]; then
  printf 'E028 run is not complete: %s\n' "$run_done" >&2
  exit 1
fi
if [[ "$(<"$run_done")" != 0 ]]; then
  printf 'E028 run failed; refusing to evaluate\n' >&2
  exit 1
fi
date -Is > "$root/logs/$run_name.evaluation.started"

command=(
  "$python_bin"
  kaggle/evaluate_pu_appearance_stability.py
  --control-dir
  logs/s157_e026_retention_ab_20260731v1/post_blend/score_e025_exact/geffs
  --candidate-dir
  "logs/$run_name/filtered_geffs"
  --control-validation-report
  logs/s157_e026_retention_ab_20260731v1/post_blend/validation_report.json
  --control-inference-manifest
  logs/s157_e026_retention_ab_20260731v1/raw/run_manifest.json
  --filter-manifest
  "logs/$run_name/filter_manifest.json"
  --gt-dir
  Dataset/train
  --runtime-dir
  Dataset/runtime-py311
  --scorer-dir
  Dataset/official-scorer-075fc5f
  --expected-count
  64
  --expected-names-sha256
  276c09d16cddaf2e865896ce147161a7beb5a62142bf47c1f1bd7648f7643e7f
  --minimum-pooled-delta
  0.002
  --output-json
  "logs/$run_name/stability.json"
  --output-csv
  "logs/$run_name/per_movie.csv"
)

"${command[@]}" > "$root/logs/$run_name.evaluation.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/logs/$run_name.evaluation.done"
exit "$status"
