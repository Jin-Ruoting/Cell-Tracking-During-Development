#!/usr/bin/env python3
"""Run the frozen E042 smoke after a verified live E041 full process finishes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path("/data/zqjinruoting/Kaggle/Cell Tracking During Development")
DEPENDENCY = ROOT / "logs/s222_hoct_scaled_full_20260924v1"
EXPECTED_PROTOCOL = "c08b20c469a683f40b28dc68116a3b8dc55d7dfc3df3646addd44461da133bdf"


def live_dependency(pid):
    path = Path(f"/proc/{pid}/cmdline")
    try:
        command = path.read_bytes().replace(b"\0", b" ").decode()
    except FileNotFoundError:
        return False
    if not command.strip():
        return False  # Exiting process; allow its shell to write the final receipt.
    if "run_hoct_scaled_consensus.py run " not in command or DEPENDENCY.name not in command:
        raise ValueError("Dependency PID no longer belongs to the expected full run")
    return True


def main():
    if Path(__file__).resolve().parents[1] != ROOT or Path(sys.prefix).name != "Kaggle":
        raise RuntimeError("Queue executes only on Uestc-220 conda Kaggle")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-pid", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.dependency_pid <= 0 or not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_name):
        raise ValueError("Invalid queue arguments")
    if not live_dependency(args.dependency_pid):
        raise ValueError("Confirm the original dependency process is live before queueing")
    folder = ROOT / "logs" / (args.run_name + "_queue")
    folder.mkdir(parents=True, exist_ok=False)
    state = {"dependency": str(DEPENDENCY), "dependency_pid": args.dependency_pid,
             "target_run": args.run_name, "expected_protocol_sha256": EXPECTED_PROTOCOL,
             "created_utc": datetime.now(timezone.utc).isoformat(), "status": "waiting_dependency",
             "queued_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}

    def record(status):
        state["status"] = status
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        text = json.dumps(state, indent=2) + "\n"
        (folder / "queue_receipt.json").write_text(text)
        (folder / "run_summary.md").write_text("# E042 dependency queue\n\n```json\n" + text + "```\n")
        print(text, flush=True)

    record("waiting_dependency")
    deadline = time.monotonic() + 14_400
    done = Path(str(DEPENDENCY) + ".done")
    try:
        missing_since = None
        while not done.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Dependency wait budget exceeded")
            if live_dependency(args.dependency_pid):
                missing_since = None
            else:
                missing_since = missing_since or time.monotonic()
                if time.monotonic() - missing_since > 20:
                    raise RuntimeError("Dependency exited without a completion receipt")
            time.sleep(10)
        receipt = json.loads((DEPENDENCY / "paired_receipt.json").read_text())
        if (done.read_text().strip() != "0" or receipt.get("technical_check_passed") is not True
                or receipt["mode"] != "full" or len(receipt["datasets"]) != 64
                or receipt["protocol_sha256"] != "af0df0fd39358e4a266c720bdd406266c8826761fde52baa8937f4cf4422bcaa"):
            raise ValueError("E041 full technical completion did not pass")
        record("waiting_gpu_idle")
        while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip():
            if time.monotonic() > deadline:
                raise TimeoutError("GPU availability wait budget exceeded")
            time.sleep(20)
        if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
            raise ValueError("Server tree must be clean")
        with (folder / "git_pull.log").open("x") as handle:
            subprocess.run(["git", "pull", "--ff-only"], stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=90)
        state["launch_git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        # Import only after the final pull; the task must use the reviewed sources.
        import run_hoct_v1_consensus as experiment

        if experiment.protocol() != EXPECTED_PROTOCOL:
            raise ValueError("Queued v1 inference protocol changed; review before running")
        record("running_smoke")
        subprocess.run(["bash", "kaggle/run_frozen_hoct_v1.sh", "smoke", args.run_name], check=True, timeout=3900)
        result = json.loads((ROOT / "logs" / args.run_name / "paired_receipt.json").read_text())
        if result.get("technical_check_passed") is not True or result["protocol_sha256"] != EXPECTED_PROTOCOL:
            raise ValueError("V1 smoke did not complete its technical protocol")
        record("technical_smoke_complete")
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        record("failed")
        raise


if __name__ == "__main__":
    main()
