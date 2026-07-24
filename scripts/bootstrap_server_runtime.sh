#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${BIOHUB_DATA_ROOT:-/data/zqjinruoting/Kaggle/Cell Tracking During Development/Dataset}"
SUPPORT_ROOT="${BIOHUB_SUPPORT_ROOT:-${DATA_ROOT}/support-pack}"
RUNTIME_DIR="${BIOHUB_RUNTIME_DIR:-${DATA_ROOT}/runtime-py311}"
WHEELS_DIR="${SUPPORT_ROOT}/wheels"

TRACKSDATA_WHEEL="${WHEELS_DIR}/tracksdata-0.1.0rc6.dev3+g980c2d30a-py3-none-any.whl"
GEFF_WHEEL="${WHEELS_DIR}/geff-1.2.0.1.1-py3-none-any.whl"
GEFF_SPEC_WHEEL="${WHEELS_DIR}/geff_spec-1.1.1-py3-none-any.whl"
ILPY_WHEEL="${WHEELS_DIR}/ilpy-0.6.0-py3-none-any.whl"

for path in \
  "$TRACKSDATA_WHEEL" \
  "$GEFF_WHEEL" \
  "$GEFF_SPEC_WHEEL" \
  "$ILPY_WHEEL"; do
  if [[ ! -f "$path" ]]; then
    printf 'Missing required support wheel: %s\n' "$path" >&2
    exit 1
  fi
done

mkdir -p "$RUNTIME_DIR"

python -m pip install \
  --target "$RUNTIME_DIR" \
  --upgrade \
  "$TRACKSDATA_WHEEL" \
  "$GEFF_WHEEL" \
  "$GEFF_SPEC_WHEEL" \
  "$ILPY_WHEEL" \
  "numcodecs==0.15.1" \
  "zarr==3.2.1" \
  "pyscipopt==6.2.1"

PYTHONPATH="$RUNTIME_DIR${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import importlib
import json
import sys

modules = [
    "tracksdata",
    "geff",
    "zarr",
    "blosc2",
    "pyscipopt",
    "ilpy",
    "polars",
]
versions = {}
for name in modules:
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "unknown")

print(
    json.dumps(
        {
            "python": sys.version,
            "modules": versions,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
