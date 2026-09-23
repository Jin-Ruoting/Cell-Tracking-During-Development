#!/usr/bin/env python3
"""E035: fixed per-track EMA velocity on E029; no detector or graph additions.

Motivation: JunhaoLiXD/Biohub_Cell_Tracking's public motion-EMA experiments.
This independently implemented adaptation retains E029's velocity weight 0.25;
it does not claim to reproduce that repository's complete configuration.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import run_flow_relink_experiment as flow

reference = flow.reference
EMA_ALPHA = 0.4
RAW_GRAPH_SHA256 = "c15c2c5599c32a7558d8cb698c750b266de4885054515c96776f3965789b632f"


def motion_function(source):
    """Keep the original assignment/gates; smooth only accepted-track velocity."""
    function = ast.unparse(flow.function_node(source))
    replacements = {
        "predecessor_position_um: dict[int, np.ndarray] = {}":
            "predecessor_position_um: dict[int, np.ndarray] = {}\n    ema_velocity = {}",
        "predicted = source_pos + MOTION_RELINK_VELOCITY_WEIGHT * (source_pos - prev_pos)":
            "predicted = source_pos + MOTION_RELINK_VELOCITY_WEIGHT * ema_velocity.get(source_id, source_pos - prev_pos)",
        "predecessor_position_um[target_id] = position_um[source_id]":
            "step = position_um[target_id] - position_um[source_id]\n"
            "            previous_velocity = ema_velocity.get(source_id, step)\n"
            f"            ema_velocity[target_id] = {EMA_ALPHA!r} * step + {1 - EMA_ALPHA!r} * previous_velocity\n"
            "            stats['motion_ema_updates'] = stats.get('motion_ema_updates', 0) + 1\n"
            "            if np.linalg.norm(ema_velocity[target_id] - step) > 1e-10:\n"
            "                stats['motion_ema_changed_velocities'] = stats.get('motion_ema_changed_velocities', 0) + 1\n"
            "            predecessor_position_um[target_id] = position_um[source_id]",
    }
    for before, after in replacements.items():
        if function.count(before) != 1:
            raise ValueError("Frozen EMA insertion anchor changed")
        function = function.replace(before, after, 1)
    compile(function, "e035-motion-ema", "exec")
    return function


def run(args):
    names = reference.selected_names(args.control_dir, args.mode)
    sources = reference.read_reference(args.reference_notebook)
    function = motion_function(sources[5])
    if reference.stability.file_sha256(args.control_csv) != flow.E029_CSV_SHA256:
        raise ValueError("E029 control bytes changed")
    parity = flow.require_replay_parity(args.replayed_control_csv, args.control_csv)
    original = json.loads((args.raw_run_dir / "run_manifest.json").read_text())
    all_names = reference.selected_names(args.control_dir, "full")
    all_graphs = sorted((args.raw_run_dir / "tracking_repo/predictions").glob("*/unet_transformer/split_0/*.geff"))
    if (original.get("reference_sha256") != reference.REFERENCE_SHA256
            or original.get("frozen_overrides") != reference.FROZEN_OVERRIDES
            or original.get("deepcenter_sha256") != reference.DEEPCENTER_SHA256
            or original.get("datasets") != all_names or sorted(p.stem for p in all_graphs) != all_names
            or flow.graph_tree_sha256(all_graphs) != RAW_GRAPH_SHA256):
        raise ValueError("Frozen E029 raw graph provenance/bytes changed")
    deepcenter = args.data_dir / "deepcenter-v1-full/weights/full_frame_center/best.pt"
    if reference.stability.file_sha256(deepcenter) != reference.DEEPCENTER_SHA256:
        raise ValueError("DeepCenter checkpoint changed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    image_dir = args.output_dir / "input/test"
    graph_dir = args.output_dir / "tracking_repo/predictions/frozen/unet_transformer/split_0"
    for path in (image_dir, graph_dir):
        path.mkdir(parents=True)
    for graph in all_graphs:
        if graph.stem in names:
            (image_dir / f"{graph.stem}.zarr").symlink_to(args.data_dir / "train" / f"{graph.stem}.zarr", target_is_directory=True)
            (graph_dir / graph.name).symlink_to(graph, target_is_directory=True)
    manifest = {"experiment": "E035", "mode": args.mode, "datasets": names,
                "reference_sha256": reference.REFERENCE_SHA256, "ema_alpha": EMA_ALPHA,
                "velocity_weight": reference.FROZEN_OVERRIDES["MOTION_RELINK_VELOCITY_WEIGHT"],
                "e029_overrides": reference.FROZEN_OVERRIDES, "deepcenter_sha256": reference.DEEPCENTER_SHA256,
                "raw_graph_tree_sha256": RAW_GRAPH_SHA256, "control_csv_sha256": flow.E029_CSV_SHA256,
                "control_replay_sha256": parity, "control_replay_source": str(args.replayed_control_csv),
                "motion_function_sha256": hashlib.sha256(function.encode()).hexdigest(),
                "runtime_parameter_search": False, "minimum_pooled_delta": 0.001,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sys.path[:0] = [str(args.runtime_dir), str(args.data_dir / "support-pack/repo/src")]
    os.environ["PYTHONPATH"] = os.pathsep.join([str(args.runtime_dir), str(args.scorer_dir / "src"), str(args.scorer_dir / "scripts")])
    os.environ.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                       "OPENBLAS_NUM_THREADS": "4", "POLARS_MAX_THREADS": "4"})
    namespace = flow.setup_namespace(args, names, sources, deepcenter)
    exec(compile(function, "E035:EMA-motion-only", "exec"), namespace)
    namespace.update({"SUBMISSION_PATH": args.output_dir / "raw_submission.csv",
                      "RUN_STATS_PATH": args.output_dir / "run_stats.csv"})
    namespace["write_test_submission"]("e035_frozen_motion_ema")
    with (args.output_dir / "run_stats.csv").open() as handle:
        stats = list(csv.DictReader(handle))
    updates = sum(int(float(row.get("motion_ema_updates") or 0)) for row in stats)
    changed = sum(int(float(row.get("motion_ema_changed_velocities") or 0)) for row in stats)
    if len(stats) != len(names) or changed <= 0 or updates < changed:
        raise ValueError("EMA motion was not active")
    if flow.graph_tree_sha256(all_graphs) != RAW_GRAPH_SHA256:
        raise ValueError("Frozen raw graphs changed")
    manifest.update({"raw_graphs_unchanged": True, "ema_updates": updates, "ema_changed_velocities": changed})
    args.expected_control_score = flow.E029_SCORE
    args.minimum_pooled_delta = 0.001
    args.experiment = "E035 frozen per-track motion EMA on E029"
    reference.complete_export(args, names, manifest, args.output_dir / "raw_submission.csv")
    result = json.loads((args.output_dir / "stability.json").read_text())
    lines = ["# E035 per-track motion EMA", "", "| Group | E029 | E035 | Difference |", "|---|---:|---:|---:|"]
    for key, group in result["groups"].items():
        lines.append(f"| {key} | {group['control']['score']:.9f} | {group['candidate']['score']:.9f} | {group['delta']['score']:+.9f} |")
    lines += ["", f"Frozen promotion gates passed: {result['promotion_passed']}", "",
              f"EMA updates: {updates}; smoothed velocities different from the last step: {changed}.", "",
              result["evidence_boundary"], ""]
    (args.output_dir / "run_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("reference-notebook", "data-dir", "control-dir", "control-csv", "replayed-control-csv",
                "raw-run-dir", "runtime-dir", "scorer-dir", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
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
            (args.output_dir / "run_summary.md").write_text(f"# E035 execution failed\n\n{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
