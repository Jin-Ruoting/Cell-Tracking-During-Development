#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${BIOHUB_DATASET_DIR:-${PROJECT_DIR}/Dataset}"
LOG_DIR="${BIOHUB_LOG_DIR:-${PROJECT_DIR}/logs}"
POLL_SECONDS="${BIOHUB_POLL_SECONDS:-300}"
EXPECTED_PAIRS="${BIOHUB_EXPECTED_TRAIN_PAIRS:-199}"
WORKERS="${BIOHUB_AUDIT_WORKERS:-8}"
STAMP="${BIOHUB_AUDIT_STAMP:-20260723}"

LOG_PATH="${LOG_DIR}/s011_frozen_frame_audit_${STAMP}.log"
JSON_PATH="${LOG_DIR}/s011_frozen_frame_audit_${STAMP}.json"
DONE_PATH="${LOG_DIR}/s011_frozen_frame_audit_${STAMP}.done"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

count_entries() {
    local suffix="$1"
    if [[ ! -d "${DATASET_DIR}/train" ]]; then
        printf '0\n'
        return 0
    fi
    find "${DATASET_DIR}/train" -maxdepth 1 -name "*.${suffix}" -print \
        2>/dev/null | wc -l | tr -d ' '
}

download_running() {
    pgrep -f \
        "[k]aggle competitions download -c biohub-cell-tracking-during-development" \
        >/dev/null
}

unpack_running() {
    pgrep -f "[u]nzip .*biohub-cell-tracking-during-development" >/dev/null
}

dataset_ready() {
    local zarr_count
    local geff_count
    zarr_count="$(count_entries zarr)"
    geff_count="$(count_entries geff)"

    ! download_running \
        && ! unpack_running \
        && [[ "${zarr_count}" -eq "${EXPECTED_PAIRS}" ]] \
        && [[ "${geff_count}" -eq "${EXPECTED_PAIRS}" ]]
}

echo "started_at=$(date --iso-8601=seconds)"
echo "project_dir=${PROJECT_DIR}"
echo "dataset_dir=${DATASET_DIR}"
echo "expected_pairs=${EXPECTED_PAIRS}"
echo "poll_seconds=${POLL_SECONDS}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "Kaggle" ]]; then
    echo "ERROR: expected active Conda environment Kaggle" >&2
    exit 2
fi

while ! dataset_ready; do
    echo "waiting_at=$(date --iso-8601=seconds) zarr=$(count_entries zarr) geff=$(count_entries geff)"
    sleep "${POLL_SECONDS}"
done

echo "ready_at=$(date --iso-8601=seconds)"
echo "commit=$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
echo "zarr_count=$(count_entries zarr)"
echo "geff_count=$(count_entries geff)"

python "${PROJECT_DIR}/scripts/audit_frozen_frames.py" \
    "${DATASET_DIR}" \
    --workers "${WORKERS}" \
    --output "${JSON_PATH}"

date --iso-8601=seconds > "${DONE_PATH}"
echo "completed_at=$(cat "${DONE_PATH}")"
