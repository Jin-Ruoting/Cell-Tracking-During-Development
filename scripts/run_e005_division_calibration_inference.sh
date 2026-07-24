#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${BIOHUB_E005_CALIBRATION_RUN_ID:-20260724}"
DATA_ROOT="${BIOHUB_DATA_ROOT:-${REPO_DIR}/Dataset}"
LOG_DIR="${BIOHUB_LOG_DIR:-${REPO_DIR}/logs}"
DIVISION_REPORT="${BIOHUB_DIVISION_REPORT:-${LOG_DIR}/s033_ground_truth_divisions_20260724b.json}"
SUPPORT_REPO="${BIOHUB_SUPPORT_REPO:-${DATA_ROOT}/support-pack/repo}"
RUNTIME="${BIOHUB_RUNTIME:-${DATA_ROOT}/runtime-py311}"
WEIGHTS="${BIOHUB_WEIGHTS:-${DATA_ROOT}/support-pack/weights/unet_transformer/split_0/edge_predictor_best.pth}"
WORK_DIR="${BIOHUB_E005_CALIBRATION_WORK_DIR:-/tmp/biohub_e005_divcal_${RUN_ID}}"
POSITIVE_LIMIT="${BIOHUB_E005_CALIBRATION_POSITIVE_LIMIT:-16}"
EXCLUDE_MANIFEST="${BIOHUB_E005_EXCLUDE_MANIFEST:-}"
ILP_DISAPPEARANCE_WEIGHT="${BIOHUB_E005_ILP_DISAPPEARANCE_WEIGHT:-1.5}"
SPATIAL_D4_TTA="${BIOHUB_E005_SPATIAL_D4_TTA:-0}"
LOG_PATH="${LOG_DIR}/s041_e005_division_calibration_inference_${RUN_ID}.log"
DONE_PATH="${LOG_DIR}/s041_e005_division_calibration_inference_${RUN_ID}.done"

VISIBLE_TEST_IDS=(
  44b6_0113de3b
  44b6_0b24845f
  6bba_05b6850b
  6bba_05db0fb1
)

mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_PATH") 2>&1

printf 'run_id=%s\n' "$RUN_ID"
printf 'started_at=%s\n' "$(date -Ins)"
printf 'repo=%s\n' "$REPO_DIR"
printf 'work_dir=%s\n' "$WORK_DIR"
printf 'positive_limit_per_embryo=%s\n' "$POSITIVE_LIMIT"
printf 'exclude_manifest=%s\n' "${EXCLUDE_MANIFEST:-none}"
printf 'ilp_disappearance_weight=%s\n' "$ILP_DISAPPEARANCE_WEIGHT"
printf 'spatial_d4_tta=%s\n' "$SPATIAL_D4_TTA"

cd "$REPO_DIR"
git status --short --branch
git rev-parse HEAD
python -m pytest -q tests

test -d "$RUNTIME"
test -d "$SUPPORT_REPO"
test -f "$WEIGHTS"
test -f "$DIVISION_REPORT"
test "$(sha256sum "$SUPPORT_REPO/scripts/predict_unet_transformer.py" | cut -d' ' -f1)" = \
  "c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9"
case "$WORK_DIR" in
  /tmp/biohub_e005_divcal_*) ;;
  *) printf 'refusing unsafe work directory: %s\n' "$WORK_DIR" >&2; exit 2 ;;
esac

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR/splits" "$WORK_DIR/preilp" "$WORK_DIR/baseline"
cp -a "$SUPPORT_REPO" "$WORK_DIR/repo0"
cp -a "$SUPPORT_REPO" "$WORK_DIR/repo1"
patch -d "$WORK_DIR/repo0" -p1 < "$REPO_DIR/patches/export_preilp_graphs.patch"
patch -d "$WORK_DIR/repo1" -p1 < "$REPO_DIR/patches/export_preilp_graphs.patch"
if [[ "$SPATIAL_D4_TTA" != "0" ]]; then
  patch -d "$WORK_DIR/repo0" -p1 < "$REPO_DIR/patches/spatial_d4_tta.patch"
  patch -d "$WORK_DIR/repo1" -p1 < "$REPO_DIR/patches/spatial_d4_tta.patch"
fi

split_args=()
for dataset in "${VISIBLE_TEST_IDS[@]}"; do
  split_args+=(--exclude "$dataset")
done
if [[ -n "$EXCLUDE_MANIFEST" ]]; then
  test -f "$EXCLUDE_MANIFEST"
  while IFS= read -r dataset; do
    split_args+=(--exclude "$dataset")
  done < <(jq -r '.selected[]' "$EXCLUDE_MANIFEST")
fi
python scripts/prepare_division_calibration_splits.py \
  "$DIVISION_REPORT" \
  "$WORK_DIR/splits" \
  "${split_args[@]}" \
  --positive-limit-per-embryo "$POSITIVE_LIMIT" \
  --shards 2

run_shard() {
  local gpu="$1"
  local shard="$2"
  local copy="$WORK_DIR/repo${shard}"
  local shard_log="$LOG_DIR/s041_e005_gpu${gpu}_${RUN_ID}.log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$RUNTIME:$copy/src"
    export BIOHUB_PREILP_DIR="$WORK_DIR/preilp/shard_${shard}"
    export USER="e005_gpu${gpu}"
    cd "$copy"
    python scripts/predict_unet_transformer.py \
      --data-dir "$DATA_ROOT/train" \
      --splits "$WORK_DIR/splits/shard_${shard}.json" \
      --split 0 \
      --weights "$WEIGHTS" \
      --method unet_transformer \
      --det-threshold 0.96875 \
      --use-ilp \
      --ilp-edge-weight -1.0 \
      --ilp-appearance-weight 0.0 \
      --ilp-disappearance-weight "$ILP_DISAPPEARANCE_WEIGHT" \
      --ilp-division-weight 1.0
    cp -a \
      "$copy/predictions/e005_gpu${gpu}/unet_transformer/split_0/." \
      "$WORK_DIR/baseline/"
  ) > "$shard_log" 2>&1
}

run_shard 0 0 &
pid0=$!
run_shard 1 1 &
pid1=$!
shard_status=0
wait "$pid0" || shard_status=1
wait "$pid1" || shard_status=1
test "$shard_status" -eq 0

expected_count="$(jq '.selected | length' "$WORK_DIR/splits/manifest.json")"
test "$(find "$WORK_DIR/preilp" -mindepth 2 -maxdepth 2 -name '*.geff' | wc -l)" -eq "$expected_count"
test "$(find "$WORK_DIR/baseline" -mindepth 1 -maxdepth 1 -name '*.geff' | wc -l)" -eq "$expected_count"

printf 'completed_at=%s\n' "$(date -Ins)"
printf 'expected_datasets=%s\n' "$expected_count"
printf 'ok\n' > "$DONE_PATH"
