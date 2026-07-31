#!/usr/bin/env bash
set -u

experiment=${1:?Usage: run_frozen_edge_tta_experiment.sh e023|e027}
root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-/home/zqjinruoting/anaconda3/envs/Kaggle/bin/python}

case "$experiment" in
  e023)
    run_name=s159_e023_legacy_d4_64_20260731v1
    gpu=0
    disappearance_weight=1.575
    edge_tta_mode=pilkwang_legacy_d4
    edge_tta_weight=0.125
    edge_tta_application=global
    ;;
  e027)
    run_name=s160_e027_selective_parent_64_20260731v1
    gpu=1
    disappearance_weight=1.5
    edge_tta_mode=corrected_d4
    edge_tta_weight=0.5
    edge_tta_application=ambiguous_parent_consensus
    ;;
  *)
    printf 'Unknown experiment: %s\n' "$experiment" >&2
    exit 64
    ;;
esac

cd "$root" || exit 1
date -Is > "$root/logs/$run_name.started"

command=(
  "$python_bin"
  kaggle/run_dual_seed_control.py
  --reference-notebook
  /tmp/biohub_public_two_seed_audit_20260724/biohub-cell-tracking-two-seeds-logit-blend.ipynb
  --support-repo
  Dataset/support-pack/repo
  --primary-weights
  Dataset/support-pack/weights/unet_transformer/split_0/edge_predictor_best.pth
  --secondary-weights
  Dataset/secondary-seed-v1/weights/unet_transformer/split_0/edge_predictor_best.pth
  --baseline-dir
  /tmp/biohub_e005_divcal_20260724a/baseline
  --data-dir
  Dataset/train
  --runtime-dir
  Dataset/runtime-py311
  --output-dir
  "logs/$run_name"
  --variants
  blend
  --gpus
  "$gpu"
  --det-threshold
  0.96875
  --ilp-disappearance-weight
  "$disappearance_weight"
  --blend-edge-weight
  0.15
  --blend-detection-weight
  0.475
  --blend-link-mode
  low_margin_consensus
  --blend-mix-temperature
  1.0
  --blend-low-margin-max
  0.35
  --blend-edge-threshold
  0.48
  --edge-tta-reference-notebook
  logs/s137_public_clean_mechanism_audit_20260727/blend_preprocessings/biohub-cell-tracking-blend-preprocessings.ipynb
  --edge-tta-mode
  "$edge_tta_mode"
  --edge-tta-original-weight
  "$edge_tta_weight"
  --edge-tta-application
  "$edge_tta_application"
  --edge-tta-ambiguous-margin-max
  0.35
  --export-preilp
  --expected-reference-sha256
  70e0c300ceae3cd7ee2cf1650c4a5f74463543e3aae1b486ba5f729a76281656
)

"${command[@]}" > "$root/logs/$run_name.log" 2>&1
status=$?
printf '%s\n' "$status" > "$root/logs/$run_name.done"
exit "$status"
