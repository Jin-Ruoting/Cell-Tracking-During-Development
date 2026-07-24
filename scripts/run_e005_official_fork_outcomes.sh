#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${BIOHUB_E005_OUTCOME_RUN_ID:-20260724}"
INFERENCE_RUN_ID="${BIOHUB_E005_INFERENCE_RUN_ID:-20260724a}"
FORK_RUN_ID="${BIOHUB_E005_FORK_RUN_ID:-20260724b}"
DATA_ROOT="${BIOHUB_DATA_ROOT:-${REPO_DIR}/Dataset}"
LOG_DIR="${BIOHUB_LOG_DIR:-${REPO_DIR}/logs}"
RUNTIME="${BIOHUB_RUNTIME:-${DATA_ROOT}/runtime-py311}"
SCORER="${BIOHUB_OFFICIAL_SCORER_DIR:-${DATA_ROOT}/official-scorer-075fc5f}"
INFERENCE_DIR="${BIOHUB_E005_INFERENCE_DIR:-/tmp/biohub_e005_divcal_${INFERENCE_RUN_ID}}"
FORK_DIR="${BIOHUB_E005_FORK_DIR:-/tmp/biohub_e005_direct_fork_sweep_${FORK_RUN_ID}}"
FORK_DONE="${LOG_DIR}/s044_e005_direct_fork_sweep_${FORK_RUN_ID}.done"
LOG_PATH="${LOG_DIR}/s045_e005_official_fork_outcomes_${RUN_ID}.log"
REPORT_PATH="${LOG_DIR}/s045_e005_official_fork_outcomes_${RUN_ID}.json"
DONE_PATH="${LOG_DIR}/s045_e005_official_fork_outcomes_${RUN_ID}.done"

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_PATH") 2>&1

printf 'run_id=%s\n' "$RUN_ID"
printf 'started_at=%s\n' "$(date -Ins)"
printf 'repo=%s\n' "$REPO_DIR"
printf 'inference_dir=%s\n' "$INFERENCE_DIR"
printf 'fork_dir=%s\n' "$FORK_DIR"

cd "$REPO_DIR"
git status --short --branch
git rev-parse HEAD
python -m pytest -q tests

test -f "$FORK_DONE"
test -d "$INFERENCE_DIR/baseline"
test -d "$INFERENCE_DIR/preilp"
test -d "$FORK_DIR/preedge_broad"
test -d "$FORK_DIR/geometry_symmetric"
test -d "$RUNTIME"
test -d "$SCORER/src"

export PYTHONPATH="$RUNTIME:$SCORER/src"
python scripts/analyze_official_fork_outcomes.py \
  "$INFERENCE_DIR/baseline" \
  "$INFERENCE_DIR/preilp" \
  "$DATA_ROOT/train" \
  --candidate "preedge_broad=$FORK_DIR/preedge_broad" \
  --candidate "geometry_symmetric=$FORK_DIR/geometry_symmetric" \
  --output "$REPORT_PATH"

printf 'completed_at=%s\n' "$(date -Ins)"
printf 'ok\n' > "$DONE_PATH"
