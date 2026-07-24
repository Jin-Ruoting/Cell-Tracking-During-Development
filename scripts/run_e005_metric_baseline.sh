#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${BIOHUB_E005_RUN_ID:-20260724}"
DATA_ROOT="${BIOHUB_DATA_ROOT:-/data/zqjinruoting/Kaggle/Cell Tracking During Development/Dataset}"
LOG_DIR="${BIOHUB_LOG_DIR:-/data/zqjinruoting/Kaggle/Cell Tracking During Development/logs}"
PREDICTIONS="${BIOHUB_E001_CV_PREDICTIONS:-/tmp/biohub_e001_v3/127_official_spec_smoke_cv_predictions.csv}"
GROUND_TRUTH_DIR="${BIOHUB_GROUND_TRUTH_DIR:-${DATA_ROOT}/train}"
WORK_DIR="${BIOHUB_E005_WORK_DIR:-/tmp/biohub_e005_official_e001_${RUN_ID}}"
LOG_PATH="${LOG_DIR}/s033_e005_metric_baseline_${RUN_ID}.log"
DIVISION_REPORT="${LOG_DIR}/s033_ground_truth_divisions_${RUN_ID}.json"
DONE_PATH="${LOG_DIR}/s033_e005_metric_baseline_${RUN_ID}.done"

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_PATH") 2>&1

printf 'run_id=%s\n' "$RUN_ID"
printf 'started_at=%s\n' "$(date -Ins)"
printf 'repo=%s\n' "$REPO_DIR"
printf 'predictions=%s\n' "$PREDICTIONS"
printf 'ground_truth=%s\n' "$GROUND_TRUTH_DIR"
printf 'work_dir=%s\n' "$WORK_DIR"

cd "$REPO_DIR"
git status --short --branch
git rev-parse HEAD

python -m pytest -q tests
bash scripts/bootstrap_server_runtime.sh
python scripts/analyze_ground_truth_divisions.py \
  "$GROUND_TRUTH_DIR" \
  --output "$DIVISION_REPORT"
bash scripts/run_official_scorer.sh \
  "$PREDICTIONS" \
  "$GROUND_TRUTH_DIR" \
  "$WORK_DIR"

printf 'completed_at=%s\n' "$(date -Ins)"
printf 'ok\n' > "$DONE_PATH"
