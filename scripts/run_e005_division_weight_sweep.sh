#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${BIOHUB_E005_SWEEP_RUN_ID:-20260724}"
INFERENCE_RUN_ID="${BIOHUB_E005_INFERENCE_RUN_ID:-20260724a}"
DATA_ROOT="${BIOHUB_DATA_ROOT:-${REPO_DIR}/Dataset}"
LOG_DIR="${BIOHUB_LOG_DIR:-${REPO_DIR}/logs}"
RUNTIME="${BIOHUB_RUNTIME:-${DATA_ROOT}/runtime-py311}"
SUPPORT_REPO="${BIOHUB_SUPPORT_REPO:-${DATA_ROOT}/support-pack/repo}"
INFERENCE_DIR="${BIOHUB_E005_INFERENCE_DIR:-/tmp/biohub_e005_divcal_${INFERENCE_RUN_ID}}"
OUTPUT_ROOT="${BIOHUB_E005_SWEEP_DIR:-/tmp/biohub_e005_division_weight_sweep_${RUN_ID}}"
INFERENCE_DONE="${LOG_DIR}/s041_e005_division_calibration_inference_${INFERENCE_RUN_ID}.done"
LOG_PATH="${LOG_DIR}/s042_e005_division_weight_sweep_${RUN_ID}.log"
DONE_PATH="${LOG_DIR}/s042_e005_division_weight_sweep_${RUN_ID}.done"

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_PATH") 2>&1

printf 'run_id=%s\n' "$RUN_ID"
printf 'started_at=%s\n' "$(date -Ins)"
printf 'repo=%s\n' "$REPO_DIR"
printf 'inference_dir=%s\n' "$INFERENCE_DIR"
printf 'output_root=%s\n' "$OUTPUT_ROOT"

cd "$REPO_DIR"
git status --short --branch
git rev-parse HEAD
python -m pytest -q tests

for _ in $(seq 1 720); do
  if [[ -f "$INFERENCE_DONE" ]]; then
    break
  fi
  sleep 30
done
test -f "$INFERENCE_DONE"
test -d "$INFERENCE_DIR/preilp"
test -d "$INFERENCE_DIR/baseline"
test -f "$INFERENCE_DIR/splits/manifest.json"
test "$(jq '.excluded | length' "$INFERENCE_DIR/splits/manifest.json")" -eq 4
case "$OUTPUT_ROOT" in
  /tmp/biohub_e005_division_weight_sweep_*) ;;
  *) printf 'refusing unsafe output root: %s\n' "$OUTPUT_ROOT" >&2; exit 2 ;;
esac
test ! -e "$OUTPUT_ROOT"

export PYTHONPATH="$RUNTIME:$SUPPORT_REPO/src"
python scripts/solve_ilp_division_weights.py \
  "$INFERENCE_DIR/preilp" \
  "$OUTPUT_ROOT" \
  --division-weight 0.75 \
  --division-weight 0.5 \
  --division-weight 0.25 \
  --division-weight 0.0 \
  --division-weight -0.25

score_candidate() {
  local label="$1"
  local prediction_dir="$2"
  printf '\n===== %s =====\n' "$label"
  bash scripts/run_official_geff_scorer.sh \
    "$prediction_dir" \
    "$DATA_ROOT/train" 2>&1 | tee "$LOG_DIR/s042_e005_score_${label}_${RUN_ID}.log"
}

score_candidate baseline "$INFERENCE_DIR/baseline"
score_candidate division_0p75 "$OUTPUT_ROOT/division_0p75"
score_candidate division_0p5 "$OUTPUT_ROOT/division_0p5"
score_candidate division_0p25 "$OUTPUT_ROOT/division_0p25"
score_candidate division_0 "$OUTPUT_ROOT/division_0"
score_candidate division_m0p25 "$OUTPUT_ROOT/division_m0p25"

printf 'completed_at=%s\n' "$(date -Ins)"
printf 'ok\n' > "$DONE_PATH"
