#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${BIOHUB_E005_EMBRYO_RUN_ID:-20260724}"
INFERENCE_RUN_ID="${BIOHUB_E005_INFERENCE_RUN_ID:-20260724v1}"
FORK_RUN_ID="${BIOHUB_E005_FORK_RUN_ID:-20260724v1}"
RULE="${BIOHUB_E005_RULE:-search_overall_rounded}"
DATA_ROOT="${BIOHUB_DATA_ROOT:-${REPO_DIR}/Dataset}"
LOG_DIR="${BIOHUB_LOG_DIR:-${REPO_DIR}/logs}"
INFERENCE_DIR="${BIOHUB_E005_INFERENCE_DIR:-/tmp/biohub_e005_divcal_${INFERENCE_RUN_ID}}"
FORK_DIR="${BIOHUB_E005_FORK_DIR:-/tmp/biohub_e005_direct_fork_sweep_${FORK_RUN_ID}}"
OUTPUT_ROOT="${BIOHUB_E005_EMBRYO_DIR:-/tmp/biohub_e005_embryo_scores_${RUN_ID}}"
FORK_DONE="${LOG_DIR}/s044_e005_direct_fork_sweep_${FORK_RUN_ID}.done"
LOG_PATH="${LOG_DIR}/s049_e005_embryo_validation_scores_${RUN_ID}.log"
DONE_PATH="${LOG_DIR}/s049_e005_embryo_validation_scores_${RUN_ID}.done"

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_PATH") 2>&1

printf 'run_id=%s\n' "$RUN_ID"
printf 'started_at=%s\n' "$(date -Ins)"
printf 'repo=%s\n' "$REPO_DIR"
printf 'rule=%s\n' "$RULE"
printf 'output_root=%s\n' "$OUTPUT_ROOT"

cd "$REPO_DIR"
git status --short --branch
git rev-parse HEAD
python -m pytest -q tests

test -f "$FORK_DONE"
test -d "$INFERENCE_DIR/baseline"
test -d "$FORK_DIR/$RULE"
case "$OUTPUT_ROOT" in
  /tmp/biohub_e005_embryo_scores_*) ;;
  *) printf 'refusing unsafe output root: %s\n' "$OUTPUT_ROOT" >&2; exit 2 ;;
esac
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

link_subset() {
  local source_dir="$1"
  local embryo="$2"
  local target_dir="$3"
  local paths=("$source_dir/${embryo}_"*.geff)
  test -e "${paths[0]}"
  mkdir -p "$target_dir"
  local path
  for path in "${paths[@]}"; do
    ln -s "$path" "$target_dir/$(basename "$path")"
  done
}

for embryo in 44b6 6bba; do
  link_subset \
    "$INFERENCE_DIR/baseline" \
    "$embryo" \
    "$OUTPUT_ROOT/baseline_$embryo"
  link_subset \
    "$FORK_DIR/$RULE" \
    "$embryo" \
    "$OUTPUT_ROOT/${RULE}_$embryo"

  printf '\n===== baseline_%s =====\n' "$embryo"
  bash scripts/run_official_geff_scorer.sh \
    "$OUTPUT_ROOT/baseline_$embryo" \
    "$DATA_ROOT/train"
  printf '\n===== %s_%s =====\n' "$RULE" "$embryo"
  bash scripts/run_official_geff_scorer.sh \
    "$OUTPUT_ROOT/${RULE}_$embryo" \
    "$DATA_ROOT/train"
done

printf 'completed_at=%s\n' "$(date -Ins)"
printf 'ok\n' > "$DONE_PATH"
