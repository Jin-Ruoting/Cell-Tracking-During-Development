#!/usr/bin/env bash
# Prepare an isolated reference runtime without upgrading the base environment.
set -euo pipefail
run_name=${1:?Usage: prepare_hoct_consensus_runtime.sh RUN_NAME}
case "$run_name" in *[!a-zA-Z0-9_-]*|'') exit 64 ;; esac
root=${BIOHUB_ROOT:-$(pwd)}
python_bin=${BIOHUB_PYTHON:-python}
cd "$root"
out="$root/logs/$run_name"
test ! -e "$out"
mkdir -p "$out/wheels"
finish() {
  status=$?
  printf '%s\n' "$status" > "$out/exit_code.txt"
  if [ "$status" != 0 ]; then
    printf '# HOCT runtime preparation failed\n\nExit status: %s. No tracking score or promotion established. See install.log.\n' "$status" > "$out/run_summary.md"
  fi
}
trap finish EXIT
timeout 180 "$python_bin" -m pip download --no-deps --timeout 20 --retries 1 \
  --dest "$out/wheels" hoct==0.2.0 spatial-graph==0.1.1 pooch==1.9.0 > "$out/install.log" 2>&1
"$python_bin" - "$out" <<'PY'
import hashlib
from pathlib import Path
import sys
path = Path(sys.argv[1]) / "wheels/hoct-0.2.0-py3-none-any.whl"
if hashlib.sha256(path.read_bytes()).hexdigest() != "c6194e81a05d272913dd0945ede2484d9a7d89f8d751c4eb1e30c43031e9093c":
    raise ValueError("HOCT 0.2.0 wheel differs from its official PyPI SHA256")
PY
"$python_bin" -m pip install --no-index --no-deps --find-links "$out/wheels" \
  --target "$out/runtime" hoct==0.2.0 spatial-graph==0.1.1 pooch==1.9.0 >> "$out/install.log" 2>&1
PYTHONPATH="$out/runtime:$root/Dataset/runtime-py311" "$python_bin" - "$out" "$root" <<'PY' >> "$out/install.log" 2>&1
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import hoct
from hoct import load_model, predict
out, root = map(Path, sys.argv[1:])
weight = root / "Dataset/hoct-general-v0/general_v0.pt"
digest = hashlib.sha256(weight.read_bytes()).hexdigest()
if digest != "024c2e4606275c96667907abfc9e0c27487b543480caf99d9ebd1d267cef8e4a":
    raise ValueError("Frozen HOCT v0 checkpoint changed")
model = load_model(weight, device="cpu")
receipt = {
    "hoct_version": importlib.metadata.version("hoct"),
    "predict_signature": str(inspect.signature(predict)),
    "checkpoint_sha256": digest,
    "parameters": sum(p.numel() for p in model.parameters()),
    "wheels": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "wheels").glob("*.whl")},
    "base_environment_modified": False, "inference_run": False,
}
(out / "runtime_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
(out / "run_summary.md").write_text(
    "# HOCT consensus runtime prepared\n\nPinned HOCT 0.2.0 and existing general_v0 load successfully "
    "in an isolated run directory. No tracking inference, quality evaluation or submission has run.\n")
print(json.dumps(receipt, indent=2))
PY
