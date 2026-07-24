#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  printf 'Usage: %s PREDICTIONS.csv GROUND_TRUTH_DIR WORK_DIR\n' "$0" >&2
  exit 2
fi

PREDICTIONS="$1"
GROUND_TRUTH_DIR="$2"
WORK_DIR="$3"

DATA_ROOT="${BIOHUB_DATA_ROOT:-/data/zqjinruoting/Kaggle/Cell Tracking During Development/Dataset}"
RUNTIME_DIR="${BIOHUB_RUNTIME_DIR:-${DATA_ROOT}/runtime-py311}"
SCORER_DIR="${BIOHUB_OFFICIAL_SCORER_DIR:-${DATA_ROOT}/official-scorer-075fc5f}"
SCORER_REPOSITORY="https://github.com/royerlab/kaggle-cell-tracking-competition.git"
SCORER_COMMIT="075fc5f5a52d11077f9dc2b074644618f26939e2"

if [[ ! -f "$PREDICTIONS" ]]; then
  printf 'Missing predictions CSV: %s\n' "$PREDICTIONS" >&2
  exit 1
fi
if [[ ! -d "$GROUND_TRUTH_DIR" ]]; then
  printf 'Missing ground-truth directory: %s\n' "$GROUND_TRUTH_DIR" >&2
  exit 1
fi
if [[ ! -d "$RUNTIME_DIR" ]]; then
  printf 'Missing runtime directory: %s\n' "$RUNTIME_DIR" >&2
  exit 1
fi

if [[ ! -d "$SCORER_DIR/.git" ]]; then
  git clone --no-checkout "$SCORER_REPOSITORY" "$SCORER_DIR"
fi
git -C "$SCORER_DIR" fetch origin "$SCORER_COMMIT" --depth=1
git -C "$SCORER_DIR" checkout --detach "$SCORER_COMMIT"

ACTUAL_COMMIT="$(git -C "$SCORER_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$SCORER_COMMIT" ]]; then
  printf 'Official scorer commit mismatch: expected %s, got %s\n' \
    "$SCORER_COMMIT" "$ACTUAL_COMMIT" >&2
  exit 1
fi

if [[ -e "$WORK_DIR" ]]; then
  printf 'Refusing to overwrite existing work directory: %s\n' "$WORK_DIR" >&2
  exit 1
fi
mkdir -p "$WORK_DIR/geffs"

export PYTHONPATH="$RUNTIME_DIR:$SCORER_DIR/src:$SCORER_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"

python "$SCORER_DIR/scripts/csv_to_geffs.py" \
  --csv "$PREDICTIONS" \
  --out-dir "$WORK_DIR/geffs"

python "$SCORER_DIR/scripts/evaluate.py" \
  --pred-dir "$WORK_DIR/geffs" \
  --gt-dir "$GROUND_TRUTH_DIR"
