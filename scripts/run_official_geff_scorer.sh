#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  printf 'Usage: %s PREDICTIONS_DIR GROUND_TRUTH_DIR\n' "$0" >&2
  exit 2
fi

PREDICTIONS_DIR="$1"
GROUND_TRUTH_DIR="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${BIOHUB_DATA_ROOT:-${REPO_DIR}/Dataset}"
RUNTIME_DIR="${BIOHUB_RUNTIME_DIR:-${DATA_ROOT}/runtime-py311}"
SCORER_DIR="${BIOHUB_OFFICIAL_SCORER_DIR:-${DATA_ROOT}/official-scorer-075fc5f}"
SCORER_COMMIT="075fc5f5a52d11077f9dc2b074644618f26939e2"

test -d "$PREDICTIONS_DIR"
test -d "$GROUND_TRUTH_DIR"
test -d "$RUNTIME_DIR"
test -d "$SCORER_DIR/.git"
test "$(git -C "$SCORER_DIR" rev-parse HEAD)" = "$SCORER_COMMIT"

export PYTHONPATH="$RUNTIME_DIR:$SCORER_DIR/src:$SCORER_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
python "$SCORER_DIR/scripts/evaluate.py" \
  --pred-dir "$PREDICTIONS_DIR" \
  --gt-dir "$GROUND_TRUTH_DIR"
