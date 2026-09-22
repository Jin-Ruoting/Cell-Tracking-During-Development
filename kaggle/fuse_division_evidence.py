#!/usr/bin/env python3
"""Fuse matched division evidence while preserving the base detection graph.

The rule uses predictions and image spacing only. Ordinary base continuation
edges and every base node are immutable; there is no per-movie routing.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

MATCH_MAX_DISTANCE_UM = 2.0
BASE_CSV_SHA256 = "5916512a9b7e3bdf3737ba3bab6e455aa7b6adffbcd9600f4398657864a238ad"
REFERENCE_CSV_SHA256 = "1d4fd28c02cb54d2279a120794b26e21fa0741d743bf62781d8ed17d5a26e2ce"
CSV_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def image_scale(image_dir: Path, name: str) -> tuple[float, float, float]:
    import zarr
    metadata = zarr.open(str(image_dir / f"{name}.zarr"), mode="r").attrs["multiscales"][0]
    if [axis["name"].lower() for axis in metadata["axes"]] != ["t", "z", "y", "x"]:
        raise ValueError("Unexpected image axis order")
    if any(axis.get("unit") != "micrometer" for axis in metadata["axes"][1:]):
        raise ValueError("Physical matching requires micrometer image metadata")
    dataset = next(item for item in metadata["datasets"] if item["path"] == "0")
    scales = [item["scale"] for item in dataset["coordinateTransformations"] if item["type"] == "scale"]
    if len(scales) != 1 or len(scales[0]) != 4:
        raise ValueError("Missing or ambiguous image scale")
    scale = tuple(float(value) for value in scales[0][1:])
    if not all(math.isfinite(value) and value > 0 for value in scale):
        raise ValueError("Invalid physical image scale")
    return scale


def mutual_mapping(base_nodes: dict, ref_nodes: dict, scale: tuple) -> dict[int, int]:
    """Map reference IDs to base IDs using close, mutual within-frame matches."""
    import numpy as np
    from scipy.spatial import cKDTree
    base_frames, ref_frames = {}, {}
    for nodes, frames in ((base_nodes, base_frames), (ref_nodes, ref_frames)):
        for node_id, coordinates in nodes.items():
            frames.setdefault(coordinates[0], []).append(node_id)
    mapping = {}
    for time in sorted(set(base_frames) & set(ref_frames)):
        base_ids, ref_ids = sorted(base_frames[time]), sorted(ref_frames[time])
        b = np.asarray([base_nodes[node][1:] for node in base_ids]) * np.asarray(scale)
        r = np.asarray([ref_nodes[node][1:] for node in ref_ids]) * np.asarray(scale)
        distances, rb = cKDTree(b).query(r, k=1)
        _, br = cKDTree(r).query(b, k=1)
        for i, (distance, base_index) in enumerate(zip(distances, rb)):
            if distance <= MATCH_MAX_DISTANCE_UM and int(br[base_index]) == i:
                mapping[ref_ids[i]] = base_ids[int(base_index)]
    if len(mapping) != len(set(mapping.values())):
        raise AssertionError("Detection correspondence is not one-to-one")
    return mapping


def fuse_divisions(base_edges: set, ref_edges: set, mapping: dict) -> tuple[set, dict]:
    def adjacency(edges):
        children, parents = {}, {}
        for source, target in edges:
            children.setdefault(source, set()).add(target)
            if target in parents and parents[target] != source:
                raise ValueError("Input graph has multiple parents")
            parents[target] = source
        if any(len(values) > 2 for values in children.values()):
            raise ValueError("Input graph has a nonbinary division")
        return children, parents

    original_children, _ = adjacency(base_edges)
    children, parents = adjacency(base_edges)
    ref_children, ref_parents = adjacency(ref_edges)
    inverse = {base: ref for ref, base in mapping.items()}
    if len(inverse) != len(mapping):
        raise ValueError("Ambiguous detection mapping")
    result, removed, added = set(base_edges), [], []
    mapped_ids = set(mapping)
    ordinary = {edge for edge in base_edges if len(original_children[edge[0]]) == 1}

    for source, targets in sorted(original_children.items()):
        if len(targets) != 2 or source not in inverse:
            continue
        ref_source = inverse[source]
        supported = {mapping[c] for c in ref_children.get(ref_source, ()) if c in mapping} & targets
        if len(supported) != 1:
            continue
        other = next(iter(targets - supported))
        ref_other = inverse.get(other)
        alternative_parent = ref_parents.get(ref_other)
        if alternative_parent is None or alternative_parent == ref_source or alternative_parent not in mapping:
            continue
        result.remove((source, other))
        children[source].remove(other)
        del parents[other]
        removed.append([source, other])

    for ref_source, ref_targets in sorted(ref_children.items()):
        if len(ref_targets) != 2 or not {ref_source, *ref_targets} <= mapped_ids:
            continue
        source = mapping[ref_source]
        targets = {mapping[child] for child in ref_targets}
        existing = children.get(source, set())
        if len(existing) != 1 or not existing <= targets:
            continue
        missing = next(iter(targets - existing))
        if missing in parents:
            continue
        result.add((source, missing))
        children.setdefault(source, set()).add(missing)
        parents[missing] = source
        added.append([source, missing])
    if not ordinary <= result:
        raise AssertionError("Ordinary continuation edges changed")
    adjacency(result)
    return result, {"mapped_nodes": len(mapping), "ordinary_edges_preserved": len(ordinary),
                    "removed_division_edges": removed, "added_division_edges": added}


def graph_records(frame) -> tuple[dict, set]:
    nodes = {int(row.node_id): (int(row.t), int(row.z), int(row.y), int(row.x))
             for row in frame[frame.row_type == "node"].itertuples(index=False)}
    edges = {(int(row.source_id), int(row.target_id))
             for row in frame[frame.row_type == "edge"].itertuples(index=False)}
    return nodes, edges


def fuse_csv(base_csv: Path, ref_csv: Path, output_csv: Path, image_dir: Path) -> dict:
    import pandas as pd
    base, ref = pd.read_csv(base_csv), pd.read_csv(ref_csv)
    if sorted(base.dataset.unique()) != sorted(ref.dataset.unique()):
        raise ValueError("Input movie coverage differs")
    ref_groups = {name: group for name, group in ref.groupby("dataset", sort=True)}
    reports, row_id = {}, 0
    with output_csv.open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for name, group in base.groupby("dataset", sort=True):
            nodes, edges = graph_records(group)
            ref_nodes, ref_edges = graph_records(ref_groups[name])
            scale = image_scale(image_dir, name)
            mapping = mutual_mapping(nodes, ref_nodes, scale)
            fused, report = fuse_divisions(edges, ref_edges, mapping)
            for source, target in fused:
                if nodes[target][0] != nodes[source][0] + 1:
                    raise ValueError("Fusion created a nonconsecutive edge")
            for node_id, coordinates in sorted(nodes.items()):
                writer.writerow(dict(zip(CSV_COLUMNS, [row_id, name, "node", node_id, *coordinates, -1, -1])))
                row_id += 1
            for source, target in sorted(fused):
                writer.writerow(dict(zip(CSV_COLUMNS, [row_id, name, "edge", -1, -1, -1, -1, -1, source, target])))
                row_id += 1
            report.update({"nodes_preserved": len(nodes), "voxel_scale_um": scale})
            reports[name] = report
            print(f"FUSED {name}: +{len(report['added_division_edges'])} / -{len(report['removed_division_edges'])} division edges", flush=True)
    return {"matching_max_distance_um": MATCH_MAX_DISTANCE_UM, "rows": row_id, "datasets": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-csv", "reference-csv", "data-dir", "runtime-dir", "scorer-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        setattr(args, key, value.resolve())
    import run_geometric_reference as reference
    for path, expected in ((args.base_csv, BASE_CSV_SHA256), (args.reference_csv, REFERENCE_CSV_SHA256)):
        if reference.stability.file_sha256(path) != expected:
            raise ValueError("Frozen input prediction bytes changed")
    os.environ.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "POLARS_MAX_THREADS": "4"})
    sys.path.insert(0, str(args.runtime_dir))
    os.environ["PYTHONPATH"] = os.pathsep.join([str(args.runtime_dir), str(args.scorer_dir / "src"), str(args.scorer_dir / "scripts")])
    import pandas as pd
    names = sorted(pd.read_csv(args.base_csv, usecols=["dataset"]).dataset.unique())
    if len(names) != 64 or reference.stability.movie_names_sha256(names) != reference.CORPUS_SHA256:
        raise ValueError("Frozen movie corpus changed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    image_dir = args.data_dir / "train"
    control_csv = args.output_dir / "e025_bounded.csv"
    export = reference.normalize_export_boundary(args.base_csv, control_csv, image_dir, names)
    (args.output_dir / "baseline_export_audit.json").write_text(json.dumps(export, indent=2) + "\n")
    base_audit = reference.validate_submission(control_csv, image_dir, names)
    (args.output_dir / "baseline_topology_audit.json").write_text(json.dumps(base_audit, indent=2) + "\n")
    fusion = fuse_csv(control_csv, args.reference_csv, args.output_dir / "submission.csv", image_dir)
    (args.output_dir / "fusion_audit.json").write_text(json.dumps(fusion, indent=2) + "\n")
    audit = reference.validate_submission(args.output_dir / "submission.csv", image_dir, names)
    (args.output_dir / "topology_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    args.control_dir = args.output_dir / "control_geffs"
    subprocess.run([sys.executable, str(args.scorer_dir / "scripts/csv_to_geffs.py"),
                    "--csv", str(control_csv), "--out-dir", str(args.control_dir)], check=True)
    manifest = {"experiment": "E030", "mode": "full", "datasets": names,
                "base_source_sha256": BASE_CSV_SHA256, "reference_source_sha256": REFERENCE_CSV_SHA256,
                "matching_max_distance_um": MATCH_MAX_DISTANCE_UM,
                "base_export_policy": reference.OUTPUT_BOUNDS_POLICY,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "inference_and_topology_passed": True}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    args.experiment = "E030 division evidence fusion on fixed E025 detections"
    # The two old E025 boundary overshoots are normalized before both arms.
    # Report this control's fresh score instead of asserting the old bytes' score.
    args.expected_control_score = None
    result = reference.evaluate(args, names)
    result["base_export_score_change"] = result["groups"]["all"]["control"]["score"] - 0.9014331472135052
    (args.output_dir / "stability.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"groups": result["groups"], "gates": result["gates"],
                      "base_export_score_change": result["base_export_score_change"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
