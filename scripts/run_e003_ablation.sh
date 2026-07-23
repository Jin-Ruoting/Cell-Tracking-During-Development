#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${BIOHUB_REPO_ROOT:-/data/zqjinruoting/Kaggle/Cell Tracking During Development}"
DATA_ROOT="${BIOHUB_DATA_ROOT:-${REPO_ROOT}/Dataset}"
LOG_DIR="${BIOHUB_LOG_DIR:-${REPO_ROOT}/logs}"
PREDICTIONS="${BIOHUB_E001_CV_PREDICTIONS:-/tmp/biohub_e001_v3/127_official_spec_smoke_cv_predictions.csv}"
FROZEN_REPORT="${BIOHUB_FROZEN_REPORT:-${LOG_DIR}/s011_frozen_frame_audit_20260723.json}"
RUN_ID="${BIOHUB_E003_RUN_ID:-20260723}"

TUNING_DATASETS="44b6_341df25f,44b6_e57ff5c6,6bba_969618f6,6bba_fc83837d"
GATES=(0.5 1.0 1.5 2.0 3.0 4.0 6.0)

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"

printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
printf 'commit=%s\n' "$(git rev-parse HEAD)"
printf 'python=%s\n' "$(command -v python)"
printf 'predictions=%s\n' "${PREDICTIONS}"
printf 'frozen_report=%s\n' "${FROZEN_REPORT}"
printf 'tuning_datasets=%s\n' "${TUNING_DATASETS}"

test -f "${PREDICTIONS}"
test -f "${FROZEN_REPORT}"
test -d "${DATA_ROOT}/train"

python -m unittest -v \
  tests.test_relink_frozen_transitions \
  tests.test_evaluate_edge_predictions \
  tests.test_analyze_frozen_edges \
  tests.test_audit_submission \
  tests.test_audit_frozen_frames

BASELINE_ALL="${LOG_DIR}/s018_e001_edge_eval_all8_${RUN_ID}.json"
BASELINE_TUNING="${LOG_DIR}/s018_e001_edge_eval_tuning_${RUN_ID}.json"
SWEEP_SUMMARY="${LOG_DIR}/s018_e003_sweep_summary_${RUN_ID}.json"
DONE_MARKER="${LOG_DIR}/s018_e003_ablation_${RUN_ID}.done"

python scripts/evaluate_edge_predictions.py \
  "${PREDICTIONS}" \
  "${DATA_ROOT}/train" \
  --output "${BASELINE_ALL}"

python - "${BASELINE_ALL}" <<'PY'
import json
import math
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
summary = report["summary"]
expected = {
    "edge_tp": 3852,
    "edge_fp": 287,
    "edge_fn": 251,
}
observed = {key: int(summary[key]) for key in expected}
if observed != expected:
    raise SystemExit(
        f"baseline reproduction failed: observed={observed}, expected={expected}"
    )
expected_score = 0.8789316334556234
score = float(summary["adjusted_edge_jaccard"])
if not math.isclose(score, expected_score, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit(
        f"baseline score reproduction failed: {score} != {expected_score}"
    )
print("BASELINE_REPRODUCTION_OK", observed, score)
PY

python scripts/evaluate_edge_predictions.py \
  "${PREDICTIONS}" \
  "${DATA_ROOT}/train" \
  --datasets "${TUNING_DATASETS}" \
  --output "${BASELINE_TUNING}"

for gate in "${GATES[@]}"; do
  tag="${gate/./p}"
  candidate="${LOG_DIR}/s018_e003_gate_${tag}_${RUN_ID}.csv"
  relink_report="${LOG_DIR}/s018_e003_gate_${tag}_relink_${RUN_ID}.json"
  audit_report="${LOG_DIR}/s018_e003_gate_${tag}_audit_${RUN_ID}.json"
  eval_report="${LOG_DIR}/s018_e003_gate_${tag}_eval_${RUN_ID}.json"

  python scripts/relink_frozen_transitions.py \
    "${PREDICTIONS}" \
    "${FROZEN_REPORT}" \
    "${candidate}" \
    --max-distance-um "${gate}" \
    --report "${relink_report}"

  python scripts/audit_submission.py \
    "${candidate}" \
    --report "${audit_report}"

  python scripts/evaluate_edge_predictions.py \
    "${candidate}" \
    "${DATA_ROOT}/train" \
    --datasets "${TUNING_DATASETS}" \
    --output "${eval_report}"
done

python - "${LOG_DIR}" "${RUN_ID}" "${BASELINE_TUNING}" "${SWEEP_SUMMARY}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

log_dir = Path(sys.argv[1])
run_id = sys.argv[2]
baseline_path = Path(sys.argv[3])
output_path = Path(sys.argv[4])
rows = []
for path in sorted(log_dir.glob(f"s018_e003_gate_*_eval_{run_id}.json")):
    tag = path.name.split("_gate_", 1)[1].split("_eval_", 1)[0]
    evaluation = json.loads(path.read_text())
    relink_path = log_dir / f"s018_e003_gate_{tag}_relink_{run_id}.json"
    audit_path = log_dir / f"s018_e003_gate_{tag}_audit_{run_id}.json"
    relink = json.loads(relink_path.read_text())
    audit = json.loads(audit_path.read_text())
    rows.append(
        {
            "gate_um": float(tag.replace("p", ".")),
            "adjusted_edge_jaccard": evaluation["summary"][
                "adjusted_edge_jaccard"
            ],
            "edge_tp": evaluation["summary"]["edge_tp"],
            "edge_fp": evaluation["summary"]["edge_fp"],
            "edge_fn": evaluation["summary"]["edge_fn"],
            "removed_edges": relink["totals"]["removed_edges"],
            "matched_edges": relink["totals"]["matched_edges"],
            "unmatched_sources": relink["totals"]["unmatched_sources"],
            "unmatched_targets": relink["totals"]["unmatched_targets"],
            "graph_valid": audit["valid"],
        }
    )

baseline = json.loads(baseline_path.read_text())["summary"]
for row in rows:
    row["delta_vs_baseline"] = (
        float(row["adjusted_edge_jaccard"])
        - float(baseline["adjusted_edge_jaccard"])
    )

report = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "selection_boundary": "training holdouts absent from visible test directory",
    "baseline": baseline,
    "candidates": rows,
}
output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY

printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
printf 'completed\n' > "${DONE_MARKER}"
