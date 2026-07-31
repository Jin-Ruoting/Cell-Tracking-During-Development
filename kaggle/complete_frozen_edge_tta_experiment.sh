#!/usr/bin/env bash
set -eu

experiment=${1:?Usage: complete_frozen_edge_tta_experiment.sh e023|e027}
root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-/home/zqjinruoting/anaconda3/envs/Kaggle/bin/python}

case "$experiment" in
  e023)
    run_name=s159_e023_legacy_d4_64_20260731v1
    control_preilp=logs/s139_e020_alpha0475_inference_20260727v1/blend/preilp
    postprocess_arm=e023_candidate
    ;;
  e027)
    run_name=s160_e027_selective_parent_64_20260731v1
    control_preilp=logs/s157_e026_retention_ab_20260731v1/raw/blend/preilp
    postprocess_arm=e027_candidate
    ;;
  *)
    printf 'Unknown experiment: %s\n' "$experiment" >&2
    exit 64
    ;;
esac

cd "$root" || exit 1
inference_done="logs/$run_name.done"
while [[ ! -f "$inference_done" ]]; do
  sleep 30
done
inference_status=$(<"$inference_done")
if [[ "$inference_status" != 0 ]]; then
  printf '%s inference failed with status %s\n' \
    "$experiment" "$inference_status" >&2
  exit 1
fi

candidate_preilp="logs/$run_name/blend/preilp"
candidate_final="logs/$run_name/blend/support_repo/predictions/dual_seed_blend/blend/split_0"
"$python_bin" kaggle/audit_edge_tta_semantics.py \
  --control-preilp-dir "$control_preilp" \
  --candidate-preilp-dir "$candidate_preilp" \
  --candidate-final-dir "$candidate_final" \
  --runtime-dir Dataset/runtime-py311 \
  --scorer-dir Dataset/official-scorer-075fc5f \
  --expected-count 64 \
  --output-json "logs/$run_name/semantic_audit_full.json"

bash kaggle/run_frozen_edge_tta_postprocess.sh "$postprocess_arm"
