#!/usr/bin/env python3
"""Audit downloaded E045-E049 example outputs without running a model."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

import build_public_953_submission as builder


def check_receipt(receipt, experiment, commit, sources):
    expected = {
        "source_kernel": builder.SOURCE,
        "source_notebook_sha256": builder.SOURCE_SHA256,
        "git_commit": commit,
        "experiment": experiment,
        "config_overrides": builder.PROFILES[experiment],
        "original_inference_cells_sha256": [builder.sha(s.encode()) for s in sources],
        "head_sha256": builder.HEAD_SHA256,
        "training_validator_packaged": False,
        "full64_promotion_claimed": False,
    }
    if receipt.get("source_proof") != expected or receipt.get("experiment") != experiment:
        raise ValueError("Executed source differs from the declared package")
    if (receipt.get("passed") is not True
            or receipt.get("coordinate_head_sha256") != builder.HEAD_SHA256
            or receipt.get("repair_fallback_movies") != 0
            or receipt.get("deadline_degraded_movies") != 0):
        raise ValueError("Example output failed its model or runtime checks")
    environment = receipt.get("actual_environment", {})
    settings = {
        "BIOHUB_VALIDATOR_ENABLE": "0", "V1284_MODE": "candidate",
        "BIOHUB_ILP_DIVISION_WEIGHT": "1.2", "BIOHUB_READMIT_MIN_SCORE": "0.965",
        "BIOHUB_DET_THRESHOLD": "0.965",
        "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION": "0.90",
        **builder.PROFILES[experiment],
    }
    if any(environment.get(key) != value for key, value in settings.items()):
        raise ValueError("Executed configuration differs from the declared profile")


def retention_audit(output, test_dir, names):
    import zarr

    frames = {name: {} for name in names}
    for path in sorted(output.glob("retention_guard_*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            name, frame = row["dataset"], row["frame"]
            if name not in frames or frame in frames[name]:
                raise ValueError("Unexpected or duplicate retention frame")
            primary, blended = row["primary_candidates"], row["blended_candidates"]
            if not all(type(value) is int and value >= 0 for value in (frame, primary, blended)):
                raise ValueError("Invalid candidate count or frame")
            ratio = blended / primary if primary else 1.0
            use_primary = bool(primary and ratio < 0.9)
            if (row["minimum_retention"] != 0.9 or row["use_primary"] is not use_primary
                    or not math.isclose(row["retention"], ratio, rel_tol=0, abs_tol=1e-12)):
                raise ValueError("Retention guard did not apply its frozen rule")
            frames[name][frame] = use_primary
    report = {}
    for name in names:
        count = zarr.open(str(test_dir / (name + ".zarr")), mode="r")["0"].shape[0]
        if set(frames[name]) != set(range(count)):
            raise ValueError(f"{name}: retention diagnostics miss actual frames")
        report[name] = {"frames": count, "primary_fallback_frames": sum(frames[name].values())}
    return report


def audit(args):
    sys.path.insert(0, str(args.runtime_dir.resolve()))
    sources = builder.read_sources(args.reference_notebook)
    receipt_dir = args.kernel_output / "logs" / (args.experiment.lower() + "-public953")
    receipt = json.loads((receipt_dir / "output_audit.json").read_text())
    check_receipt(receipt, args.experiment, args.source_commit, sources)
    names = sorted(path.stem for path in args.test_dir.glob("*.zarr") if path.is_dir())
    if not names:
        raise ValueError("Actual test volumes are required")
    coverage = receipt.get("peak_cache_files", {})
    if sorted(coverage) != names or any(
        item.get("bytes", 0) <= 0 or not {"coords", "low_coords", "low_score"}.issubset(item.get("fields", []))
        for item in coverage.values()
    ):
        raise ValueError("Runtime real-peak cache receipt does not cover every movie")
    actual = builder.reference.validate_submission(args.kernel_output / "submission.csv", args.test_dir, names)
    if any(actual[key] != receipt.get(key) for key in ("rows", "datasets", "submission_sha256")):
        raise ValueError("Downloaded CSV differs from the runtime audit")
    export = builder.reference.normalize_export_boundary(
        receipt_dir / "raw_submission.csv", args.output_dir / "independent_reexport.csv", args.test_dir, names)
    if export != receipt.get("export") or export["submission_sha256"] != actual["submission_sha256"]:
        raise ValueError("Raw-to-final export is not reproducible")
    with (receipt_dir / "run_stats.csv").open() as handle:
        stats = list(csv.DictReader(handle))
    if sorted(row["dataset"] for row in stats) != names:
        raise ValueError("Statistics do not cover the actual movies exactly once")
    if any(float(row[key]) != 0 for row in stats for key in ("repair_fallback", "deadline_degraded")):
        raise ValueError("Example inference used a repair or deadline fallback")
    retention = retention_audit(args.kernel_output, args.test_dir, names)
    report = {**actual, "experiment": args.experiment, "source_commit": args.source_commit,
              "kernel": args.kernel, "kernel_version": args.kernel_version,
              "config_overrides": builder.PROFILES[args.experiment],
              "independent_reexport_passed": True, "export_boundary_changes": len(export["changes"]),
              "retention": retention, "repair_fallback_movies": 0, "deadline_degraded_movies": 0,
              "public_score": None, "status": "independent_example_output_audit_passed_not_submitted"}
    (args.output_dir / "independent_output_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "run_summary.md").write_text(
        "# Public method independent example audit\n\n"
        "Downloaded CSV, frozen configuration, raw reexport, runtime diagnostics and "
        "all-frame retention checks pass. This is example execution evidence; "
        "public scoring and hidden-run diagnostics remain pending.\n\n```json\n"
        + json.dumps(report, indent=2) + "\n```\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("kernel-output", "test-dir", "runtime-dir", "output-dir", "reference-notebook"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--experiment", choices=builder.PROFILES, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--kernel-version", type=int, required=True)
    args = parser.parse_args()
    if args.kernel_version < 1 or not re.fullmatch("[0-9a-f]{40}", args.source_commit):
        parser.error("A positive Kernel version and full source commit are required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        audit(args)
    except Exception as error:
        (args.output_dir / "run_summary.md").write_text(
            f"# Independent output audit failed\n\n{type(error).__name__}: {error}\n\nDo not submit.\n")
        raise


if __name__ == "__main__":
    main()
