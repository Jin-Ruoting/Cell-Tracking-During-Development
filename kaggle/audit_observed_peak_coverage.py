#!/usr/bin/env python3
"""Diagnose how many unmatched training labels have a nearby unused E029 peak.

This is a many-to-one geometric coverage bound, not a new tracking prediction
or precision estimate. It does not alter the preregistered recovery arms.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess

import audit_e000_error_budget as graphs
import run_coordinate_calibration_experiment as coordinate
import run_observed_peak_experiment as experiment


def potential_coverage(gt_rows, missing_ids, node_rows, low, scores, threshold):
    import numpy as np
    from scipy.spatial import cKDTree

    covered = set()
    scale = np.asarray(graphs.VOXEL_SCALE_UM)
    for t in sorted({int(row["t"]) for row in gt_rows if int(row["node_id"]) in missing_ids}):
        targets = [row for row in gt_rows if int(row["t"]) == t and int(row["node_id"]) in missing_ids]
        peaks = low[(low[:, 0] == t) & (scores >= threshold), 1:].astype(float) * scale
        existing = [row for row in node_rows if int(row["t"]) == t]
        if not len(peaks):
            continue
        if existing:
            current = np.array([[row[key] for key in ("z", "y", "x")] for row in existing]) * scale
            nearest, _ = cKDTree(current).query(peaks)
            peaks = peaks[nearest > 2.0]
        if not len(peaks):
            continue
        positions = np.array([[row[key] for key in ("z", "y", "x")] for row in targets]) * scale
        nearest, _ = cKDTree(peaks).query(positions)
        covered.update(int(row["node_id"]) for row, distance in zip(targets, nearest) if distance <= 7.0)
    return covered


def run(args):
    os.environ.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                       "OPENBLAS_NUM_THREADS": "4", "POLARS_MAX_THREADS": "4"})
    graphs.prepare_imports(args.runtime_dir, args.scorer_dir)
    import numpy as np
    import tracksdata as td

    reference = experiment.reference
    names = reference.selected_names(args.control_dir, "full")
    if experiment.flow.graph_tree_sha256(sorted(args.control_dir.glob("*.geff"))) != coordinate.E029_GEFF_TREE_SHA256:
        raise ValueError("Frozen E029 scored graph bytes changed")
    caches, receipts = experiment.verified_caches(args.cache_dir, names, "full", args.raw_run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    groups = {key: Counter() for key in ("all", "44b6", "6bba")}
    results = {}
    for name in names:
        pred = graphs.load_graph(args.control_dir / f"{name}.geff", td)
        gt = graphs.load_graph(args.data_dir / "train" / f"{name}.geff", td)
        graphs.match_graph(pred, gt, graphs.VOXEL_SCALE_UM, 7.0)
        matched = graphs.matched_view(pred, td)["matched_gt_nodes"]
        gt_rows = graphs.rows(gt.node_attrs())
        node_rows = graphs.rows(pred.node_attrs())
        missing = {int(row["node_id"]) for row in gt_rows} - matched
        gt_edges = graphs.ground_truth_edges(gt, td)
        incomplete = {(a, b) for a, b in gt_edges if a in missing or b in missing}
        with np.load(caches[name], allow_pickle=False) as data:
            low, scores = data["low_coords"], data["low_score"]
        record = {"gt_nodes": len(gt_rows), "matched_gt_nodes": len(matched), "unmatched_gt_nodes": len(missing),
                  "gt_edges": len(gt_edges), "gt_edges_missing_endpoint": len(incomplete)}
        for label, threshold in (("p050", 0.5), ("p0965", 0.965)):
            covered = potential_coverage(gt_rows, missing, node_rows, low, scores, threshold)
            record[f"unmatched_near_free_peak_{label}"] = len(covered)
            available = matched | covered
            record[f"missing_endpoint_edges_geometrically_coverable_{label}"] = sum(
                a in available and b in available for a, b in incomplete)
        results[name] = record
        groups["all"].update(record)
        groups[name.split("_")[0]].update(record)
        print(json.dumps({"dataset": name, **record}), flush=True)
    report = {"groups": {key: dict(value) for key, value in groups.items()}, "per_movie": results,
              "datasets": names, "control_graph_tree_sha256": coordinate.E029_GEFF_TREE_SHA256,
              "cache_sha256": {name: value["cache_sha256"] for name, value in receipts.items()},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "gt_match_radius_um": 7.0, "peak_to_gt_radius_um": 7.0, "existing_node_exclusion_um": 2.0,
              "ground_truth_used": "training diagnostic only", "submission_created": False,
              "evidence_boundary": "Many-to-one geometric coverage bound; ignores new-node conflicts, endpoint/direction gates and false positives in unlabelled regions. Not a score or precision estimate."}
    (args.output_dir / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Observed-peak coverage diagnostic", "",
             "| Group | Unmatched GT nodes | Near free peak >=0.5 | Near free peak >=0.965 |",
             "|---|---:|---:|---:|"]
    for key, values in groups.items():
        lines.append(f"| {key} | {values['unmatched_gt_nodes']} | {values['unmatched_near_free_peak_p050']} | {values['unmatched_near_free_peak_p0965']} |")
    lines += ["", report["evidence_boundary"], "", "The E033/E034 configuration and advancement gates are unchanged.", ""]
    (args.output_dir / "run_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data-dir", "runtime-dir", "scorer-dir", "control-dir", "cache-dir", "raw-run-dir", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    try:
        run(args)
    except Exception as exc:
        if args.output_dir.exists():
            (args.output_dir / "run_summary.md").write_text(
                f"# Peak coverage diagnostic failed\n\n{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
