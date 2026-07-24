#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${BIOHUB_E005_SEARCH_RUN_ID:-20260724}"
OUTCOME_RUN_ID="${BIOHUB_E005_OUTCOME_RUN_ID:-20260724b}"
LOG_DIR="${BIOHUB_LOG_DIR:-${REPO_DIR}/logs}"
OUTCOME_REPORT="${LOG_DIR}/s045_e005_official_fork_outcomes_${OUTCOME_RUN_ID}.json"
OUTCOME_DONE="${LOG_DIR}/s045_e005_official_fork_outcomes_${OUTCOME_RUN_ID}.done"
LOG_PATH="${LOG_DIR}/s046_e005_official_rule_search_${RUN_ID}.log"
REPORT_PATH="${LOG_DIR}/s046_e005_official_rule_search_${RUN_ID}.json"
DONE_PATH="${LOG_DIR}/s046_e005_official_rule_search_${RUN_ID}.done"

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_PATH") 2>&1

printf 'run_id=%s\n' "$RUN_ID"
printf 'started_at=%s\n' "$(date -Ins)"
printf 'repo=%s\n' "$REPO_DIR"
printf 'outcome_report=%s\n' "$OUTCOME_REPORT"

cd "$REPO_DIR"
git status --short --branch
git rev-parse HEAD
python -m pytest -q tests

test -f "$OUTCOME_DONE"
test -f "$OUTCOME_REPORT"
python scripts/search_official_fork_rules.py \
  "$OUTCOME_REPORT" \
  --rule preedge_broad \
  --division-total 44b6=21 \
  --division-total 6bba=54 \
  --max-depth 4 \
  --beam-width 500 \
  --result-count 20 \
  --output "$REPORT_PATH"

printf 'completed_at=%s\n' "$(date -Ins)"
printf 'ok\n' > "$DONE_PATH"
