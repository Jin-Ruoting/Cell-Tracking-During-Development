#!/usr/bin/env python3
"""E038: fixed, observation-supported bridges after the final E029 export."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

import audit_point_detector_coverage as cache_audit
import point_gap_bridge as bridge
import run_coordinate_calibration_experiment as coordinate
import run_flow_relink_experiment as flow

reference = flow.reference


def append_bridges(group, proposals):
    import pandas as pd

    rows = []
    name = group.dataset.iloc[0]
    for item in proposals:
        row = dict.fromkeys(group.columns, -1)
        row.update(dataset=name, row_type="node", node_id=item["node_id"],
                   **dict(zip(("t", "z", "y", "x"), item["point"])))
        rows.append(row)
        for a, b in ((item["source_id"], item["node_id"]), (item["node_id"], item["target_id"])):
            row = dict.fromkeys(group.columns, -1)
            row.update(dataset=name, row_type="edge", source_id=a, target_id=b)
            rows.append(row)
    if not rows:
        return group.copy()
    return pd.concat([group, pd.DataFrame(rows, columns=group.columns)], ignore_index=True)


def run(args):
    import numpy as np
    import pandas as pd

    names = reference.selected_names(args.control_dir, args.mode)
    all_names = reference.selected_names(args.control_dir, "full")
    if reference.stability.file_sha256(args.control_csv) != flow.E029_CSV_SHA256:
        raise ValueError("Frozen E029 CSV changed")
    if flow.graph_tree_sha256(sorted(args.control_dir.glob("*.geff"))) != coordinate.E029_GEFF_TREE_SHA256:
        raise ValueError("Frozen E029 scored graphs changed")
    parity = flow.require_replay_parity(args.replayed_control_csv, args.control_csv)
    caches, receipts = cache_audit.verified_caches(args.cache_dir, all_names, "full")
    if any(record.get("gpu_peak_allocated_bytes", 0) <= 0 for record in receipts.values()):
        raise ValueError("Require the uniformly CUDA-predicted full cache")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    image_dir = args.output_dir / "input/test"
    image_dir.mkdir(parents=True)
    for name in names:
        (image_dir / f"{name}.zarr").symlink_to(args.data_dir / "train" / f"{name}.zarr", target_is_directory=True)
    manifest = {"experiment": "E038", "mode": args.mode, "datasets": names,
                "config": bridge.CONFIG, "control_replay_sha256": parity,
                "control_graph_tree_sha256": coordinate.E029_GEFF_TREE_SHA256,
                "point_cache_sha256": {name: receipts[name]["cache_sha256"] for name in names},
                "bridge_source_sha256": reference.stability.file_sha256(Path(bridge.__file__)),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "minimum_pooled_delta": 0.001, "runtime_parameter_search": False,
                "predictions_used_ground_truth": False, "training_exclusions_confirmed": False,
                "preserve_all_original_nodes_coordinates_edges": True}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    frame = pd.read_csv(args.control_csv)
    frame = frame[frame.dataset.isin(names)]
    outputs, records, totals = [], {}, Counter()
    for name in names:
        group = frame[frame.dataset == name]
        nodes, edges = group[group.row_type == "node"], group[group.row_type == "edge"]
        with np.load(caches[name], allow_pickle=False) as data:
            coords, scores = data["coords"], data["scores"]
            bridge.validate_peaks(coords, scores, data["processed_frames"], data["image_shape"])
            if data["image_shape"].tolist() != receipts[name]["shape"] or len(coords) != receipts[name]["peaks"]:
                raise ValueError("Peak cache shape or count differs from frozen receipt")
        proposals, counts = bridge.bridge_one_frame(
            nodes.node_id.to_numpy(), nodes[["t", "z", "y", "x"]].to_numpy(),
            edges[["source_id", "target_id"]].to_numpy(), coords, scores)
        output = append_bridges(group, proposals)
        # Appending must preserve every original field and row in its original
        # order; only the global row-id column is regenerated during export.
        if not output.iloc[:len(group)].reset_index(drop=True).equals(group.reset_index(drop=True)):
            raise ValueError("Bridge generation modified the original graph")
        outputs.append(output)
        records[name] = {"counts": counts, "bridges": proposals}
        totals.update(counts)
        print(json.dumps({"dataset": name, **counts}), flush=True)
    output = pd.concat(outputs, ignore_index=True)
    output["id"] = np.arange(len(output))
    output.to_csv(args.output_dir / "raw_submission.csv", index=False)
    (args.output_dir / "bridge_audit.json").write_text(json.dumps(records, indent=2) + "\n")
    manifest.update(component_counts=dict(totals), all_candidate_predictions_completed_before_scoring=True)
    args.expected_control_score, args.minimum_pooled_delta = flow.E029_SCORE, 0.001
    args.experiment = "E038 observed v5 one-frame bridges on final E029"
    os.environ["PYTHONPATH"] = os.pathsep.join([str(args.runtime_dir), str(args.scorer_dir / "src"),
                                               str(args.scorer_dir / "scripts")])
    reference.complete_export(args, names, manifest, args.output_dir / "raw_submission.csv")
    if reference.stability.file_sha256(args.control_csv) != flow.E029_CSV_SHA256:
        raise ValueError("Original E029 CSV changed during evaluation")
    if flow.graph_tree_sha256(sorted(args.control_dir.glob("*.geff"))) != coordinate.E029_GEFF_TREE_SHA256:
        raise ValueError("Original E029 scored graphs changed during evaluation")
    for name in names:
        if reference.stability.file_sha256(caches[name]) != receipts[name]["cache_sha256"]:
            raise ValueError("Frozen point cache changed during evaluation")
    result = json.loads((args.output_dir / "stability.json").read_text())
    (args.output_dir / "run_summary.md").write_text(
        "# E038 observation-supported bridges\n\nAll original E029 nodes, coordinates and edges retained. "
        "Frozen parameters; development evidence only.\n\n```json\n" + json.dumps(
            {"groups": result["groups"], "gates": result["gates"], "paired": result["paired"],
             "component_counts": dict(totals)}, indent=2) + "\n```\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data-dir", "control-dir", "control-csv", "replayed-control-csv", "runtime-dir", "scorer-dir", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.cache_dir = [p.resolve() for p in args.cache_dir]
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    sys.path.insert(0, str(args.runtime_dir))
    try:
        run(args)
    except Exception as exc:
        if args.output_dir.exists():
            (args.output_dir / "run_summary.md").write_text(
                f"# E038 failed\n\n{type(exc).__name__}: {exc}\n\nNo fallback or promotion.\n")
        raise


if __name__ == "__main__":
    main()
