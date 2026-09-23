#!/usr/bin/env python3
"""Independently audit E035 actual Kernel outputs before formal submission."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import build_motion_ema_submission as builder

reference = builder.reference


def check_receipt(receipt, proof):
    expected = {
        "external_reference_sha256": reference.REFERENCE_SHA256,
        "frozen_overrides": reference.FROZEN_OVERRIDES,
        "motion_function_sha256": proof["motion_function_sha256"],
        "ema_alpha": builder.ema.EMA_ALPHA,
        "velocity_weight": reference.FROZEN_OVERRIDES["MOTION_RELINK_VELOCITY_WEIGHT"],
        "offline_promotion": proof, "status": "kernel_output_audited_public_score_pending", "passed": True,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Executed receipt differs from the approved E035 package")
    if (proof.get("selection_policy") != "all_frozen_e035_gates"
            or proof.get("outside_author_selection", {}).get("passed") is not True):
        raise ValueError("Complete E035 promotion evidence is missing")
    environment = receipt.get("effective_environment", {})
    if environment.get("BIOHUB_VALIDATOR_ENABLE") != "0":
        raise ValueError("Runtime label validator was not disabled")
    for key, value in reference.FROZEN_OVERRIDES.items():
        if environment.get("BIOHUB_" + key) != str(value):
            raise ValueError("Effective E029 override changed")
    if not 0 < receipt.get("ema_changed_velocities", 0) <= receipt.get("ema_updates", 0):
        raise ValueError("Per-track EMA was not active")


def audit(args):
    sys.path.insert(0, str(args.runtime_dir.resolve()))
    load = lambda path: json.loads(path.read_text())
    proof = load(args.package_dir / "development_proof.json")
    internal = load(args.kernel_output / "e035_output_audit.json")
    check_receipt(internal, proof)
    names = sorted(path.stem for path in args.test_dir.glob("*.zarr") if path.is_dir())
    if not names:
        raise ValueError("Actual test volumes are required")
    actual = reference.validate_submission(args.kernel_output / "submission.csv", args.test_dir, names)
    for key in ("rows", "datasets", "submission_sha256"):
        if actual[key] != internal.get(key):
            raise ValueError(f"Independent CSV audit disagrees on {key}")
    reexport = reference.normalize_export_boundary(args.kernel_output / "raw_submission.csv",
        args.output_dir / "independent_reexport.csv", args.test_dir, names)
    if (reexport != load(args.kernel_output / "export_boundary_audit.json")
            or reexport["submission_sha256"] != actual["submission_sha256"]):
        raise ValueError("Raw-to-final reexport differs from downloaded output")
    with (args.kernel_output / "run_stats.csv").open() as handle:
        stats = list(csv.DictReader(handle))
    if sorted(row["dataset"] for row in stats) != names:
        raise ValueError("EMA statistics do not cover the actual test movies")
    counts = {key: sum(int(float(row.get("motion_" + key) or 0)) for row in stats)
              for key in ("ema_updates", "ema_changed_velocities")}
    if any(internal.get(key) != value for key, value in counts.items()):
        raise ValueError("Actual EMA counters disagree with the output receipt")
    report = {**actual, **counts, "independent_reexport_passed": True,
              "export_boundary_changes": len(reexport["changes"]),
              "package_notebook_sha256": reference.stability.file_sha256(args.package_dir / "biohub_e035_motion_ema.ipynb"),
              "kernel": args.kernel, "kernel_version": args.kernel_version,
              "public_score": None, "status": "independent_output_audit_passed_not_submitted"}
    (args.output_dir / "independent_output_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "run_summary.md").write_text(
        "# E035 independent Kernel output audit\n\nDownloaded CSV, configuration, active EMA "
        "and raw reexport pass independently. This is executable example-test evidence, "
        "not a public leaderboard score.\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package-dir", "kernel-output", "test-dir", "runtime-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--kernel-version", type=int, required=True)
    args = parser.parse_args()
    if args.kernel_version < 1:
        parser.error("Kernel version must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        audit(args)
    except Exception as exc:
        (args.output_dir / "run_summary.md").write_text(
            f"# E035 independent output audit failed\n\n{type(exc).__name__}: {exc}\n\nDo not submit.\n")
        raise


if __name__ == "__main__":
    main()
