#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${BIOHUB_E005_CANDIDATE_RUN_ID:-20260724}"
INFERENCE_RUN_ID="${BIOHUB_E005_INFERENCE_RUN_ID:-20260724a}"
DATA_ROOT="${BIOHUB_DATA_ROOT:-${REPO_DIR}/Dataset}"
LOG_DIR="${BIOHUB_LOG_DIR:-${REPO_DIR}/logs}"
RUNTIME="${BIOHUB_RUNTIME:-${DATA_ROOT}/runtime-py311}"
INFERENCE_DIR="${BIOHUB_E005_INFERENCE_DIR:-/tmp/biohub_e005_divcal_${INFERENCE_RUN_ID}}"
INFERENCE_DONE="${LOG_DIR}/s041_e005_division_calibration_inference_${INFERENCE_RUN_ID}.done"
LOG_PATH="${LOG_DIR}/s043_e005_division_candidate_audit_${RUN_ID}.log"
REPORT_PATH="${LOG_DIR}/s043_e005_division_candidates_${RUN_ID}.json"
DONE_PATH="${LOG_DIR}/s043_e005_division_candidate_audit_${RUN_ID}.done"

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_PATH") 2>&1

printf 'run_id=%s\n' "$RUN_ID"
printf 'started_at=%s\n' "$(date -Ins)"
printf 'repo=%s\n' "$REPO_DIR"
printf 'inference_dir=%s\n' "$INFERENCE_DIR"

cd "$REPO_DIR"
git status --short --branch
git rev-parse HEAD
python -m pytest -q tests

test -f "$INFERENCE_DONE"
test -d "$INFERENCE_DIR/baseline"
test -d "$INFERENCE_DIR/preilp"
test -d "$DATA_ROOT/train"
test -d "$RUNTIME"

export PYTHONPATH="$RUNTIME${PYTHONPATH:+:$PYTHONPATH}"
python scripts/analyze_division_candidates.py \
  "$INFERENCE_DIR/baseline" \
  "$INFERENCE_DIR/preilp" \
  "$DATA_ROOT/train" \
  --output "$REPORT_PATH"

printf 'completed_at=%s\n' "$(date -Ins)"
printf 'ok\n' > "$DONE_PATH"
