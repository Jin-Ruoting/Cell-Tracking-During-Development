#!/usr/bin/env bash
set -u

root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-/home/zqjinruoting/anaconda3/envs/Kaggle/bin/python}
run_name=s162_e028_pu_appearance_20260731v1

cd "$root" || exit 1
date -Is > "$root/logs/$run_name.started"

command=(
  "$python_bin"
  kaggle/run_pu_appearance_filter.py
  --notebook
  kaggle/biohub_clean_baseline.ipynb
  --baseline-dir
  logs/s157_e026_retention_ab_20260731v1/post_blend/score_e025_exact/geffs
  --image-dir
  Dataset/train
  --ground-truth-dir
  Dataset/train
  --runtime-dir
  Dataset/runtime-py311
  --support-src
  Dataset/support-pack/repo/src
  --scorer-dir
  Dataset/official-scorer-075fc5f
  --output-dir
  "logs/$run_name"
  --expected-count
  64
  --expected-names-sha256
  276c09d16cddaf2e865896ce147161a7beb5a62142bf47c1f1bd7648f7643e7f
  --device
  cpu
)

"${command[@]}" > "$root/logs/$run_name.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/logs/$run_name.done"
exit "$status"
