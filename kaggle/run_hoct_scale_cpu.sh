#!/usr/bin/env bash
# Run after git pull --ff-only in conda Kaggle; return to the existing screen.
set -euo pipefail
run_name=${1:?Usage: run_hoct_scale_cpu.sh RUN_NAME}
case "$run_name" in *[!a-zA-Z0-9_-]*|'') exit 64 ;; esac
root=${BIOHUB_ROOT:-$(pwd)}
cd "$root"
test ! -e "logs/$run_name"
test ! -e "logs/$run_name.log"
finish() {
  status=$?
  mkdir -p "logs/$run_name"
  printf '%s\n' "$status" > "logs/$run_name/exit_code.txt"
  if [ "$status" = 124 ]; then
    printf '# HOCT audit timeout\n\nReached the 840-second CPU budget; no successful full audit or quality claim.\n' > "logs/$run_name/run_summary.md"
  fi
}
trap finish EXIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
timeout 840 python -u kaggle/check_hoct_scale.py server --run-name "$run_name" > "logs/$run_name.log" 2>&1
