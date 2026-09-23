#!/usr/bin/env python3
"""Acquire the public author's fixed-size reference files without executing them."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import zipfile

BASE = "https://www.kaggle.com/api/v1/datasets/download/hengck23/hengck23-cell-point-detector-demo/"
FILES = {"model_v5.py": 7885, "model_v12.py": 21346, "loss_and_metric_v12.py": 126,
         "00000030.pth": 69455847, "00000008.pth": 43076221}
V5_SHA256 = "68e13d3424721ee43e6d247f97d9f533dbeb735b46c038ec3c636035ee3a4e23"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def acquire(out, name, size):
    pending = out / (name + ".part")
    destination = out / name
    if pending.exists() or destination.exists():
        raise FileExistsError("Do not overwrite an earlier reference download")
    seconds = 30 if size < 1000000 else min(1200, math.ceil(size / 65536) + 60)
    command = ["curl", "-4", "--fail", "--silent", "--show-error", "--location",
               "--proto", "=https", "--proto-redir", "=https", "--connect-timeout", "5",
               "--max-time", str(seconds), "--max-filesize", str(math.ceil(size * 1.1) + 65536),
               "--speed-limit", "16384", "--speed-time", "60", "--output", str(pending),
               "--write-out", "%{http_code} %{size_download} %{time_total}\n", BASE + name]
    result = subprocess.run(command, capture_output=True, text=True, timeout=seconds + 10)
    (out / (name + ".download.log")).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{name}: bounded download failed, exit {result.returncode}")
    payload = pending.read_bytes()
    transport_sha = sha(payload)
    wrapped = len(payload) != size
    if wrapped:
        # PyTorch checkpoints can themselves be ZIPs: only unwrap when the
        # transport size differs and there is exactly the expected named file.
        if not zipfile.is_zipfile(pending):
            raise ValueError(f"{name}: unexpected content size")
        with zipfile.ZipFile(pending) as archive:
            if archive.namelist() != [name] or archive.getinfo(name).file_size != size:
                raise ValueError(f"{name}: unexpected download archive")
            payload = archive.read(name)
    if len(payload) != size:
        raise ValueError(f"{name}: public file-size contract changed")
    digest = sha(payload)
    if name == "model_v5.py" and digest != V5_SHA256:
        raise ValueError("Previously reviewed detector source changed")
    if name.endswith(".py"):
        ast.parse(payload.decode("utf-8"), filename=name)
    if wrapped:
        with destination.open("xb") as handle:
            handle.write(payload)
        pending.rename(out / (name + ".transport.zip"))
    else:
        pending.rename(destination)
    return {"name": name, "bytes": size, "sha256": digest, "public_url": BASE + name,
            "transport_sha256": transport_sha, "wrapped": wrapped, "timeout_seconds": seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--files", nargs="+", choices=list(FILES), default=list(FILES))
    args = parser.parse_args()
    if len(set(args.files)) != len(args.files):
        parser.error("Each public filename must occur only once")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"dataset": "hengck23/hengck23-cell-point-detector-demo", "artifacts": [],
              "requested_files": args.files,
              "started_at": datetime.now(timezone.utc).isoformat(),
              "model_executed": False, "weights_loaded": False, "gpu_used": False,
              "status": "acquiring", "official_sizes_verified": False, "dataset_license": "unconfirmed",
              "evidence_boundary": "Reference acquisition only; no model or quality result"}
    try:
        for name in args.files:
            report["artifacts"].append(acquire(out, name, FILES[name]))
            (out / "acquisition_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report["artifacts"][-1]), flush=True)
        report["status"] = "reference_artifacts_acquired_no_inference"
        report["official_sizes_verified"] = True
    except Exception as exc:
        report.update({"status": "incomplete", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        report["observed_at"] = datetime.now(timezone.utc).isoformat()
        (out / "acquisition_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "run_summary.md").write_text("# Public point-model reference acquisition\n\n```json\n" +
                                           json.dumps(report, indent=2) + "\n```\n")


if __name__ == "__main__":
    main()
