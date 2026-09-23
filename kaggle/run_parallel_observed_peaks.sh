#!/usr/bin/env bash
# Wait for the shared capture receipt, then evaluate both fixed arms concurrently.
set -euo pipefail
cache_run=${1:?Usage: run_parallel_observed_peaks.sh CACHE_RUN READMIT_RUN GAP_RUN QUEUE_RUN}
readmit_run=${2:?Provide a new readmission run}
gap_run=${3:?Provide a new gap-fill run}
queue_run=${4:?Provide a new queue receipt name}
for name in "$cache_run" "$readmit_run" "$gap_run" "$queue_run"; do
  case "$name" in *[!a-zA-Z0-9_-]*|'') exit 64 ;; esac
done
root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-python}
cd "$root"
test ! -e "logs/$queue_run"
test ! -e "logs/$readmit_run"
test ! -e "logs/$gap_run"
mkdir -p "logs/$queue_run"
printf '# Parallel observed-peak comparison\n\nWaiting for completed shared detector cache. No parameter selection or submission is performed by this queue.\n' > "logs/$queue_run/run_summary.md"
finish() {
  status=$?
  printf '%s\n' "$status" > "logs/$queue_run/exit_code.txt"
  if [ "$status" != 0 ]; then
    printf '\nExecution failed with status %s; inspect the individual run summaries.\n' "$status" >> "logs/$queue_run/run_summary.md"
  fi
}
trap finish EXIT
deadline=$((SECONDS + 3600))
while [ ! -f "logs/$cache_run.done" ]; do
  if [ "$SECONDS" -ge "$deadline" ]; then exit 124; fi
  sleep 10
done
test "$(cat "logs/$cache_run.done")" = 0
printf '# Parallel observed-peak comparison\n\nFull detector cache completed. E033 and E034 are evaluating on separate GPUs with the same fixed control.\n' > "logs/$queue_run/run_summary.md"
CUDA_VISIBLE_DEVICES=0 BIOHUB_PYTHON="$python_bin" timeout 7200 bash kaggle/run_frozen_observed_peaks.sh \
  readmit full "$cache_run" "$readmit_run" > "logs/$queue_run/readmit_launch.log" 2>&1 &
readmit_pid=$!
CUDA_VISIBLE_DEVICES=1 BIOHUB_PYTHON="$python_bin" timeout 7200 bash kaggle/run_frozen_observed_peaks.sh \
  gapfill full "$cache_run" "$gap_run" > "logs/$queue_run/gap_launch.log" 2>&1 &
gap_pid=$!
status=0
if wait "$readmit_pid"; then :; else status=1; fi
if wait "$gap_pid"; then :; else status=1; fi
if [ "$status" != 0 ]; then exit "$status"; fi
"$python_bin" - "logs/$queue_run" "logs/$readmit_run" "logs/$gap_run" <<'PY'
import json
from pathlib import Path
import sys
out, *runs = map(Path, sys.argv[1:])
lines = ["# Parallel observed-peak comparison", "", "| Experiment | Control | Candidate | Difference | Gates passed |",
         "|---|---:|---:|---:|---|"]
for run in runs:
    result = json.loads((run / "stability.json").read_text())
    group = result["groups"]["all"]
    lines.append(f"| {result['experiment']} | {group['control']['score']:.9f} | {group['candidate']['score']:.9f} | "
                 f"{group['delta']['score']:+.9f} | {result['promotion_passed']} |")
lines += ["", "Completed development comparisons only; no automatic combination, Kernel upload or competition submission.", ""]
(out / "run_summary.md").write_text("\n".join(lines))
PY
