#!/usr/bin/env bash
set -u

arm=${1:?Usage: run_frozen_edge_tta_postprocess.sh e023_control|e023_candidate|e027_candidate}
root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-/home/zqjinruoting/anaconda3/envs/Kaggle/bin/python}

case "$arm" in
  e023_control)
    baseline=logs/s139_e020_alpha0475_inference_20260727v1/blend/support_repo/predictions/dual_seed_blend/blend/split_0
    output=logs/s161_e023_control_e000_revalidation_20260731v1
    mode=--e000-only
    ;;
  e023_candidate)
    baseline=logs/s159_e023_legacy_d4_64_20260731v1/blend/support_repo/predictions/dual_seed_blend/blend/split_0
    output=logs/s159_e023_legacy_d4_64_20260731v1/post_e000
    mode=--e000-only
    ;;
  e027_candidate)
    baseline=logs/s160_e027_selective_parent_64_20260731v1/blend/support_repo/predictions/dual_seed_blend/blend/split_0
    output=logs/s160_e027_selective_parent_64_20260731v1/post_e025
    mode=--e025-exact
    ;;
  *)
    printf 'Unknown arm: %s\n' "$arm" >&2
    exit 64
    ;;
esac

cd "$root" || exit 1
command=(
  "$python_bin"
  kaggle/validate_e006_postprocess.py
  --notebook
  kaggle/biohub_clean_baseline.ipynb
  --baseline-dir
  "$baseline"
  --image-dir
  Dataset/train
  --ground-truth-dir
  Dataset/train
  --runtime-dir
  Dataset/runtime-py311
  --support-src
  Dataset/support-pack/repo/src
  --scorer-dir
  Dataset/official-scorer-075fc5f
  --output-dir
  "$output"
  "$mode"
)
if [[ "$mode" == --e025-exact ]]; then
  command+=(
    --deepcenter-checkpoint
    Dataset/deepcenter-v1-full/weights/full_frame_center/checkpoint_last.pt
  )
fi

"${command[@]}"
