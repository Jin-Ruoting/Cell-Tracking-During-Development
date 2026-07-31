#!/usr/bin/env bash
set -eu

experiment=${1:?Usage: evaluate_frozen_edge_tta_experiment.sh e023|e027}
root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-/home/zqjinruoting/anaconda3/envs/Kaggle/bin/python}
names_sha256=276c09d16cddaf2e865896ce147161a7beb5a62142bf47c1f1bd7648f7643e7f

case "$experiment" in
  e023)
    run_name=s159_e023_legacy_d4_64_20260731v1
    control_dir=logs/s161_e023_control_e000_revalidation_20260731v1/score_e000_safe/geffs
    candidate_dir=logs/s159_e023_legacy_d4_64_20260731v1/post_e000/score_e000_safe/geffs
    control_report=logs/s161_e023_control_e000_revalidation_20260731v1/validation_report.json
    candidate_report=logs/s159_e023_legacy_d4_64_20260731v1/post_e000/validation_report.json
    control_manifest=logs/s139_e020_alpha0475_inference_20260727v1/run_manifest.json
    ;;
  e027)
    run_name=s160_e027_selective_parent_64_20260731v1
    control_dir=logs/s157_e026_retention_ab_20260731v1/post_blend/score_e025_exact/geffs
    candidate_dir=logs/s160_e027_selective_parent_64_20260731v1/post_e025/score_e025_exact/geffs
    control_report=logs/s157_e026_retention_ab_20260731v1/post_blend/validation_report.json
    candidate_report=logs/s160_e027_selective_parent_64_20260731v1/post_e025/validation_report.json
    control_manifest=logs/s157_e026_retention_ab_20260731v1/raw/run_manifest.json
    ;;
  *)
    printf 'Unknown experiment: %s\n' "$experiment" >&2
    exit 64
    ;;
esac

cd "$root" || exit 1
completion_done="logs/$run_name/completion.done"
while [[ ! -f "$completion_done" ]]; do
  sleep 30
done
completion_status=$(<"$completion_done")
if [[ "$completion_status" != 0 ]]; then
  printf '%s completion failed with status %s\n' \
    "$experiment" "$completion_status" >&2
  exit 1
fi

"$python_bin" kaggle/evaluate_edge_tta_stability.py \
  --experiment "$experiment" \
  --control-dir "$control_dir" \
  --candidate-dir "$candidate_dir" \
  --control-validation-report "$control_report" \
  --candidate-validation-report "$candidate_report" \
  --control-inference-manifest "$control_manifest" \
  --candidate-inference-manifest "logs/$run_name/run_manifest.json" \
  --gt-dir Dataset/train \
  --runtime-dir Dataset/runtime-py311 \
  --scorer-dir Dataset/official-scorer-075fc5f \
  --expected-count 64 \
  --expected-names-sha256 "$names_sha256" \
  --minimum-pooled-delta 0.002 \
  --output-json "logs/$run_name/stability.json" \
  --output-csv "logs/$run_name/per_movie.csv"
