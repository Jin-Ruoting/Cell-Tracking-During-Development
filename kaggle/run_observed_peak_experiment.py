#!/usr/bin/env python3
"""Score independently frozen readmission (E033) or observed gap filling (E034)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import capture_frozen_detection_peaks as capture
import observed_peak_recovery as recovery

flow = recovery.flow
reference = flow.reference


def verified_caches(cache_dir, names, mode, raw_run_dir):
    import numpy as np

    original = capture.original_coordinates(raw_run_dir)
    paths, receipts = {}, {}
    for shard in (0, 1):
        directory = cache_dir / f"shard{shard}"
        manifest = json.loads((directory / "run_manifest.json").read_text())
        expected = {"purpose": "frozen_e029_detection_peaks", "mode": mode, "shard": shard,
                    "predictor_sha256": capture.PREDICTOR_SHA256, "low_threshold": capture.LOW_THRESHOLD,
                    "detection_config": capture.DETECTION_CONFIG, "ground_truth_accessed": False,
                    "primary_sha256": reference.stability.EXPECTED_PRIMARY_WEIGHT_SHA256,
                    "secondary_sha256": reference.stability.EXPECTED_SECONDARY_WEIGHT_SHA256,
                    "complete": True, "datasets": names[shard::2]}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Incomplete or changed frozen peak-cache provenance")
        if sorted(manifest["movies"]) != sorted(names[shard::2]):
            raise ValueError("Peak cache shard coverage changed")
        for name, record in manifest["movies"].items():
            path = directory / "peaks" / f"{name}.npz"
            if (name in paths or record.get("detector_parity_passed") is not True
                    or record["high_coordinate_sha256"] != original[name]["coordinate_sha256"]
                    or record["cache_sha256"] != reference.stability.file_sha256(path)):
                raise ValueError("Peak cache bytes or original detection parity changed")
            with np.load(path, allow_pickle=False) as data:
                capture.validate_arrays(data["low_coords"], data["low_score"], data["processed_frames"], record["shape"])
                if data["image_shape"].tolist() != record["shape"] or len(data["low_coords"]) != record["peaks"]:
                    raise ValueError("Peak cache geometry or size changed")
            paths[name], receipts[name] = path, record
    if sorted(paths) != names:
        raise ValueError("Missing peak caches")
    return paths, receipts


def run(args):
    sys.path[:0] = [str(args.runtime_dir), str(args.data_dir / "support-pack/repo/src")]
    os.environ["PYTHONPATH"] = os.pathsep.join([str(args.runtime_dir), str(args.scorer_dir / "src"),
                                               str(args.scorer_dir / "scripts")])
    os.environ.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                       "OPENBLAS_NUM_THREADS": "4", "POLARS_MAX_THREADS": "4"})
    names = reference.selected_names(args.control_dir, args.mode)
    sources = reference.read_reference(args.reference_notebook)
    helpers = recovery.selected_functions(args.peak_reference)
    caches, receipts = verified_caches(args.cache_dir, names, args.mode, args.raw_run_dir)
    if reference.stability.file_sha256(args.control_csv) != flow.E029_CSV_SHA256:
        raise ValueError("Frozen E029 control CSV changed")
    original = json.loads((args.raw_run_dir / "run_manifest.json").read_text())
    all_names = reference.selected_names(args.control_dir, "full")
    found = sorted((args.raw_run_dir / "tracking_repo/predictions").glob("*/unet_transformer/split_0/*.geff"))
    if (original.get("reference_sha256") != reference.REFERENCE_SHA256
            or original.get("frozen_overrides") != reference.FROZEN_OVERRIDES
            or original.get("deepcenter_sha256") != reference.DEEPCENTER_SHA256
            or original.get("datasets") != all_names or sorted(p.stem for p in found) != all_names):
        raise ValueError("Raw E029 provenance or coverage changed")
    graphs = [p for p in found if p.stem in names]
    raw_hash = flow.graph_tree_sha256(graphs)
    deepcenter = args.data_dir / "deepcenter-v1-full/weights/full_frame_center/best.pt"
    if reference.stability.file_sha256(deepcenter) != reference.DEEPCENTER_SHA256:
        raise ValueError("DeepCenter checkpoint changed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    image_dir = args.output_dir / "input/test"
    graph_dir = args.output_dir / "tracking_repo/predictions/frozen/unet_transformer/split_0"
    peak_dir = args.output_dir / "peaks"
    for path in (image_dir, graph_dir, peak_dir, args.output_dir / "control"):
        path.mkdir(parents=True)
    for graph in graphs:
        (image_dir / f"{graph.stem}.zarr").symlink_to(args.data_dir / "train" / f"{graph.stem}.zarr", target_is_directory=True)
        (graph_dir / graph.name).symlink_to(graph, target_is_directory=True)
        (peak_dir / f"{graph.stem}.npz").symlink_to(caches[graph.stem])
    experiment = "E033" if args.arm == "readmit" else "E034"
    manifest = {"experiment": experiment, "arm": args.arm, "mode": args.mode, "datasets": names,
                "reference_sha256": reference.REFERENCE_SHA256, "peak_reference_sha256": flow.FLOW_REFERENCE_SHA256,
                "helper_source_sha256": hashlib.sha256(helpers.encode()).hexdigest(),
                "recovery_config": recovery.CONFIG, "e029_overrides": reference.FROZEN_OVERRIDES,
                "raw_graph_tree_sha256": raw_hash, "control_csv_sha256": flow.E029_CSV_SHA256,
                "peak_cache_receipts": receipts, "deepcenter_sha256": reference.DEEPCENTER_SHA256,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "runtime_parameter_search": False, "minimum_pooled_delta": 0.001}
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    namespace = flow.setup_namespace(args, names, sources, deepcenter)
    control = args.output_dir / "control"
    if args.mode == "smoke":
        namespace.update({"SUBMISSION_PATH": control / "raw_submission.csv", "RUN_STATS_PATH": control / "run_stats.csv"})
        namespace["write_test_submission"]("e029_exact_replay")
        reference.normalize_export_boundary(control / "raw_submission.csv", control / "submission.csv", image_dir, names)
        expected = control / "expected_submission.csv"
        flow.write_subset(args.control_csv, expected, names)
        parity = flow.require_replay_parity(control / "submission.csv", expected)
        manifest["control_replay_source"] = "fresh_two_movie_replay"
    else:
        # Full replay was already generated by the unchanged E029 function in
        # E031. Verify and reuse those exact bytes, rather than repeat 64 movies.
        if args.replayed_control_csv is None:
            raise ValueError("Full mode requires the existing exact E029 replay")
        parity = flow.require_replay_parity(args.replayed_control_csv, args.control_csv)
        (control / "submission.csv").symlink_to(args.replayed_control_csv)
        manifest["control_replay_source"] = str(args.replayed_control_csv)
    print(f"E029 CONTROL BYTE PARITY PASSED: {parity}", flush=True)
    namespace.update(recovery.CONFIG)
    os.environ["BIOHUB_CACHE_DIR"] = str(peak_dir)
    exec(compile(helpers, "x138:observed-peak-helpers", "exec"), namespace)
    filtered = recovery.patch_filter(sources[5], args.arm)
    exec(compile(filtered, f"{experiment}:isolated-filter", "exec"), namespace)
    namespace.update({"SUBMISSION_PATH": args.output_dir / "raw_submission.csv",
                      "RUN_STATS_PATH": args.output_dir / "run_stats.csv"})
    namespace["write_test_submission"](f"{experiment.lower()}_{args.arm}")
    with (args.output_dir / "run_stats.csv").open() as handle:
        stats = list(csv.DictReader(handle))
    if len(stats) != len(names) or any(int(float(row.get("peak_component_calls") or 0)) != 1 for row in stats):
        raise ValueError("Observed-peak component did not execute on every movie")
    if any(int(float(row.get("peak_cache_loads") or 0)) != 1 for row in stats):
        raise ValueError("Peak cache not read exactly once for each movie")
    counters = {key: sum(int(float(row.get(key) or 0)) for row in stats) for key in (
        "readmitted_nodes", "readmitted_retained_nodes", "gapfill_added_nodes", "gapfill_retained_nodes",
        "gapfill_added_edges", "gapfill_synthetic_nodes", "gapfill_pool_peaks", "peak_component_calls")}
    if counters["gapfill_synthetic_nodes"] != 0:
        raise ValueError("Unobserved synthetic gap-fill nodes are forbidden")
    if (args.arm == "readmit" and counters["gapfill_added_nodes"]) or (args.arm == "gapfill" and counters["readmitted_nodes"]):
        raise ValueError("Both recovery components were activated")
    if flow.graph_tree_sha256(graphs) != raw_hash:
        raise ValueError("Original E029 graphs changed")
    for name, path in caches.items():
        if reference.stability.file_sha256(path) != receipts[name]["cache_sha256"]:
            raise ValueError("Frozen peak cache changed")
    manifest.update({"control_replay_sha256": parity, "raw_graphs_unchanged": True, "peak_caches_unchanged": True,
                     "component_counts": counters, "filter_source_sha256": hashlib.sha256(filtered.encode()).hexdigest()})
    args.expected_control_score = flow.E029_SCORE
    args.minimum_pooled_delta = 0.001
    args.experiment = f"{experiment} isolated {args.arm} on E029"
    reference.complete_export(args, names, manifest, args.output_dir / "raw_submission.csv")
    result = json.loads((args.output_dir / "stability.json").read_text())
    lines = [f"# {experiment}: {args.arm}", "", "| Group | E029 | Candidate | Difference |", "|---|---:|---:|---:|"]
    for key, group in result["groups"].items():
        lines.append(f"| {key} | {group['control']['score']:.9f} | {group['candidate']['score']:.9f} | {group['delta']['score']:+.9f} |")
    lines += ["", f"Frozen promotion gates passed: {result['promotion_passed']}", "",
              "```json", json.dumps(counters, indent=2), "```", "", result["evidence_boundary"], ""]
    (args.output_dir / "run_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("reference-notebook", "peak-reference", "cache-dir", "data-dir", "control-dir", "control-csv",
                "raw-run-dir", "runtime-dir", "scorer-dir", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--replayed-control-csv", type=Path)
    parser.add_argument("--arm", choices=("readmit", "gapfill"), required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    try:
        run(args)
    except Exception as exc:
        if args.output_dir.exists():
            (args.output_dir / "run_summary.md").write_text(
                f"# {args.arm} failed\n\n{type(exc).__name__}: {exc}\n\nNo promotion established.\n")
        raise


if __name__ == "__main__":
    main()
