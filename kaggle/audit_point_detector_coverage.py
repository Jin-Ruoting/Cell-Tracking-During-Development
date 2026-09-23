#!/usr/bin/env python3
"""Audit frozen v5 peaks near E029's unmatched training labels.

This reports a geometric, many-to-one opportunity bound, not new recall or a
tracking score. Prediction caches must be complete before any labels are read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import audit_e000_error_budget as graphs
from audit_observed_peak_coverage import potential_coverage
import check_point_detector_runtime as runtime
import run_coordinate_calibration_experiment as coordinate
import run_flow_relink_experiment as flow
import run_observed_peak_experiment as observed

reference = flow.reference


def verified_caches(directories, names, mode):
    caches, records = {}, {}
    for folder in directories:
        manifest = json.loads((folder / "run_manifest.json").read_text())
        expected = {"purpose": "reviewed_v5_real_image_peaks", "mode": mode,
                    "complete": True, "ground_truth_accessed": False,
                    "downsample": [1, 4, 4], "sampling": "strided",
                    "quantiles": ["0.001", "0.999"], "clip": "minimum_zero_only",
                    "peak_threshold": 0.2, "peak_kernel": 3}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Incomplete or changed point-detector prediction contract")
        if manifest.get("verified_sha256") != {name: runtime.PINNED[name]
                                               for name in ("model_v5.py", "00000030.pth")}:
            raise ValueError("Unexpected point-detector source/checkpoint")
        if set(manifest["datasets"]) != set(manifest["movies"]):
            raise ValueError("Incomplete cached movie coverage")
        for name, record in manifest["movies"].items():
            path = folder / "peaks" / f"{name}.npz"
            if (name in caches or record["frames"] != 100
                    or reference.stability.file_sha256(path) != record["cache_sha256"]):
                raise ValueError("Changed, duplicated or partial cache")
            caches[name], records[name] = path, record
    if set(caches) != set(names):
        raise ValueError("Fixed movie coverage mismatch")
    return caches, records


def run(args):
    names = reference.selected_names(args.control_dir, args.mode)
    if flow.graph_tree_sha256(sorted(args.control_dir.glob("*.geff"))) != coordinate.E029_GEFF_TREE_SHA256:
        raise ValueError("Frozen E029 scored graph bytes changed")
    caches, receipts = verified_caches(args.cache_dir, names, args.mode)
    original_caches, original_receipts = {}, {}
    if args.e029_cache_dir is not None:
        original_caches, original_receipts = observed.verified_caches(
            args.e029_cache_dir, reference.selected_names(args.control_dir, "full"), "full", args.raw_run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    graphs.prepare_imports(args.runtime_dir, args.scorer_dir)
    import numpy as np
    import tracksdata as td

    groups = {key: Counter() for key in ("all", "44b6", "6bba")}
    results = {}
    for name in names:
        with np.load(caches[name], allow_pickle=False) as data:
            coords, scores, frames = data["coords"], data["scores"], data["processed_frames"]
            if frames.tolist() != list(range(100)):
                raise ValueError("Incomplete prediction-frame coverage")
        pred = graphs.load_graph(args.control_dir / f"{name}.geff", td)
        gt = graphs.load_graph(args.data_dir / "train" / f"{name}.geff", td)
        graphs.match_graph(pred, gt, graphs.VOXEL_SCALE_UM, 7.0)
        matched = graphs.matched_view(pred, td)["matched_gt_nodes"]
        gt_rows, node_rows = graphs.rows(gt.node_attrs()), graphs.rows(pred.node_attrs())
        missing = {int(row["node_id"]) for row in gt_rows} - matched
        record = {"gt_nodes": len(gt_rows), "e029_matched_gt_nodes": len(matched),
                  "e029_unmatched_gt_nodes": len(missing), "v5_peaks_ge_020": len(coords),
                  "v5_peaks_ge_050": int((scores >= 0.5).sum())}
        coverage_by_threshold = {}
        for label, threshold in (("p020", 0.2), ("p050", 0.5)):
            covered = potential_coverage(gt_rows, missing, node_rows, coords, scores, threshold)
            coverage_by_threshold[label] = covered
            record[f"e029_unmatched_near_free_v5_{label}"] = len(covered)
        if original_caches:
            with np.load(original_caches[name], allow_pickle=False) as data:
                original_coords, original_scores = data["low_coords"], data["low_score"]
            v5_high = coverage_by_threshold["p050"]
            for label, threshold in (("p050", 0.5), ("p0965", 0.965)):
                original = potential_coverage(gt_rows, missing, node_rows, original_coords, original_scores, threshold)
                record[f"e029_unmatched_near_original_free_peaks_{label}"] = len(original)
                record[f"v5_p050_additional_to_original_{label}"] = len(v5_high - original)
                record[f"v5_p050_overlap_original_{label}"] = len(v5_high & original)
                record[f"combined_v5_p050_original_{label}"] = len(v5_high | original)
        results[name] = record
        groups["all"].update(record)
        groups[name.split("_")[0]].update(record)
        print(json.dumps({"dataset": name, **record}), flush=True)
    report = {"mode": args.mode, "datasets": names, "groups": {key: dict(value) for key, value in groups.items()},
              "per_movie": results, "control_graph_sha256": coordinate.E029_GEFF_TREE_SHA256,
              "cache_sha256": {name: value["cache_sha256"] for name, value in receipts.items()},
              "original_e029_peak_cache_sha256": {name: original_receipts[name]["cache_sha256"]
                                                   for name in names if original_receipts},
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "official_e029_match_radius_um": 7.0, "peak_to_gt_radius_um": 7.0,
              "existing_node_exclusion_um": 2.0, "training_exclusions_confirmed": False,
              "submission_created": False, "tracking_score": None,
              "evidence_boundary": "Many-to-one geometric coverage of missing training labels only. Not new recall, precision, held-out performance or graph-score improvement."}
    (args.output_dir / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "run_summary.md").write_text("# V5 complementary-coverage diagnostic\n\n```json\n" +
        json.dumps(report, indent=2) + "\n```\n\n" + report["evidence_boundary"] + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data-dir", "control-dir", "runtime-dir", "scorer-dir", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--e029-cache-dir", type=Path)
    parser.add_argument("--raw-run-dir", type=Path)
    args = parser.parse_args()
    if (args.e029_cache_dir is None) != (args.raw_run_dir is None):
        parser.error("The original peak cache and its frozen raw-run provenance are required together")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    try:
        run(args)
    except Exception as exc:
        if args.output_dir.exists():
            (args.output_dir / "run_summary.md").write_text(f"# V5 diagnostic failed\n\n{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
