#!/usr/bin/env python3
"""Independently audit downloaded E031 Kernel output before formal submission."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import build_flow_submission as builder

flow = builder.flow
reference = builder.reference


def check_receipt(receipt, proof):
    expected = {
        "external_reference_sha256": reference.REFERENCE_SHA256,
        "frozen_overrides": reference.FROZEN_OVERRIDES,
        "flow_reference_sha256": flow.FLOW_REFERENCE_SHA256,
        "flow_function_sha256": proof["flow_function_sha256"],
        "flow_config": flow.FLOW_CONFIG,
        "offline_promotion": proof,
        "status": "kernel_output_audited_public_score_pending",
        "passed": True,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Executed output receipt differs from the approved E031 package")
    if proof.get("selection_policy") != "all_frozen_e031_gates" or proof.get("outside_author_selection", {}).get("passed") is not True:
        raise ValueError("Complete E031 promotion evidence is missing")
    environment = receipt.get("effective_environment", {})
    if environment.get("BIOHUB_VALIDATOR_ENABLE") != "0":
        raise ValueError("Runtime label validator was not disabled")
    for key, value in reference.FROZEN_OVERRIDES.items():
        if environment.get("BIOHUB_" + key) != str(value):
            raise ValueError("Effective E029 override changed")
    if receipt.get("flow_frames", 0) <= 0:
        raise ValueError("Neighborhood flow was not active")


def audit(args):
    sys.path.insert(0, str(args.runtime_dir.resolve()))
    load = lambda path: json.loads(path.read_text())
    proof = load(args.package_dir / "development_proof.json")
    internal = load(args.kernel_output / "e031_output_audit.json")
    check_receipt(internal, proof)
    names = sorted(path.stem for path in args.test_dir.glob("*.zarr") if path.is_dir())
    if not names:
        raise ValueError("No actual test volumes available for independent audit")
    actual = reference.validate_submission(args.kernel_output / "submission.csv", args.test_dir, names)
    for key in ("rows", "datasets", "submission_sha256"):
        if actual[key] != internal.get(key):
            raise ValueError(f"Independent CSV audit disagrees on {key}")
    reexport = reference.normalize_export_boundary(args.kernel_output / "raw_submission.csv",
        args.output_dir / "independent_reexport.csv", args.test_dir, names)
    expected_export = load(args.kernel_output / "export_boundary_audit.json")
    if reexport != expected_export or reexport["submission_sha256"] != actual["submission_sha256"]:
        raise ValueError("Independent raw-to-final export did not reproduce the downloaded output")
    with (args.kernel_output / "run_stats.csv").open() as handle:
        stats = list(csv.DictReader(handle))
    flow_frames = sum(int(float(row.get("motion_relink_flow_frames") or 0)) for row in stats)
    if flow_frames != internal["flow_frames"] or flow_frames <= 0:
        raise ValueError("Executed flow statistics disagree with the output receipt")
    report = {**actual, "flow_frames": flow_frames, "independent_reexport_passed": True,
              "export_boundary_changes": len(reexport["changes"]),
              "package_notebook_sha256": reference.stability.file_sha256(args.package_dir / "biohub_e031_flow.ipynb"),
              "kernel": args.kernel, "kernel_version": args.kernel_version,
              "public_score": None, "status": "independent_output_audit_passed_not_submitted"}
    (args.output_dir / "independent_output_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "run_summary.md").write_text(
        "# E031 independent Kernel output audit\n\nThe downloaded CSV and raw reexport pass independently. "
        "This verifies executable example-test output, not a public leaderboard score.\n\n```json\n"
        + json.dumps(report, indent=2) + "\n```\n")
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
            f"# E031 independent output audit failed\n\n{type(exc).__name__}: {exc}\n\nDo not submit this output.\n")
        raise


if __name__ == "__main__":
    main()
