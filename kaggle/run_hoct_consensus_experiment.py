#!/usr/bin/env python3
"""E036/E037: fixed HOCT v0 edge intersection on the final E029 node set."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd

import probe_hoct_consensus as geometry
import run_flow_relink_experiment as flow

reference = flow.reference
CONTROL_GRAPH_SHA256 = "50ac680ef53c7456ea00fd1358ea706b33c61d62fb6e5f4b3ed7748cf6581aff"


def unique_positions(ids, points):
    """Represent each coincident position once, without inventing coordinates."""
    order = np.argsort(ids)
    _, first, inverse, counts = np.unique(points[order], axis=0, return_index=True,
                                          return_inverse=True, return_counts=True)
    selected = order[np.sort(first)]
    ambiguous = set(map(int, ids[order][counts[inverse] > 1]))
    return ids[selected], points[selected], ambiguous


def retain_edges(pairs, consensus, protect_divisions, ambiguous_ids=()):
    degree = Counter(source for source, _ in pairs)
    ambiguous_ids = set(ambiguous_ids)
    return [pair[0] in ambiguous_ids or pair[1] in ambiguous_ids or pair in consensus
            or (protect_divisions and degree[pair[0]] == 2) for pair in pairs]


def infer_movie(args):
    import torch
    import zarr
    from hoct import load_model, predict
    from tracksdata.functional import TilingScheme

    if any((args.output_dir / name).exists() for name in ("pairs.npz", "receipt.json")):
        raise FileExistsError("Do not overwrite prior HOCT inference")
    if (geometry.digest(args.input_nodes) != args.input_sha256
            or geometry.digest(args.weight) != geometry.WEIGHT_SHA256
            or importlib.metadata.version("hoct") != "0.2.0"):
        raise ValueError("Frozen HOCT inputs changed")
    with np.load(args.input_nodes, allow_pickle=False) as data:
        ids, points = data["ids"], data["points"]
    original_count = len(ids)
    ids, points, ambiguous = unique_positions(ids, points)
    started = time.monotonic()
    images = np.asarray(zarr.open_group(str(args.image), mode="r")["0"][:])
    if images.ndim != 4 or images.shape[0] != 100 or images.dtype != np.uint16:
        raise ValueError("Full-movie raw image contract changed")
    labels = geometry.rasterize(points, images.shape)
    prepared = time.monotonic()
    model = load_model(args.weight, device="cuda")
    with torch.inference_mode():
        solution = predict(model, labels=labels, images=images,
                           scale=(1.0, *geometry.SCALE), max_delta_t=1,
                           tiling_scheme=TilingScheme(tile_shape=(5, 32, 128, 128), overlap_shape=(1, 8, 16, 16)))
    nodes = solution.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    edges = solution.edge_attrs(attr_keys=[])
    out_points = np.column_stack([nodes[key].to_numpy() for key in ("t", "z", "y", "x")])
    mapping, max_distance = geometry.snap_nodes(out_points, points)
    remap = {int(node): int(ids[index]) for node, index in zip(nodes["node_id"].to_list(), mapping)}
    pairs = [(remap[int(a)], remap[int(b)]) for a, b in zip(edges["source_id"].to_list(), edges["target_id"].to_list())]
    times = dict(zip(map(int, ids), points[:, 0].astype(int)))
    if (not pairs or len(pairs) != len(set(pairs)) or any(times[b] - times[a] != 1 for a, b in pairs)
            or max(Counter(a for a, _ in pairs).values(), default=0) > 2
            or max(Counter(b for _, b in pairs).values(), default=0) > 1):
        raise ValueError("Invalid HOCT temporal solution")
    # The veto preserves all E029 nodes, including any not selected by HOCT.
    # Record HOCT coverage explicitly instead of silently replacing its graph.
    np.savez_compressed(args.output_dir / "pairs.npz", pairs=np.asarray(pairs, dtype=np.int64))
    receipt = {"movie": args.image.stem, "frames": len(images), "input_nodes": original_count,
               "represented_positions": len(ids), "ambiguous_node_ids": sorted(ambiguous),
               "solution_nodes": len(mapping), "solution_node_fraction": len(mapping) / original_count,
               "represented_position_coverage": len(mapping) / len(ids),
               "edges": len(pairs), "max_snap_distance_um": max_distance,
               "prepare_seconds": prepared - started, "inference_audit_seconds": time.monotonic() - prepared,
               "input_sha256": args.input_sha256,
               "pairs_sha256": geometry.digest(args.output_dir / "pairs.npz"),
               "checkpoint_sha256": geometry.WEIGHT_SHA256,
               "versions": {key: importlib.metadata.version(key) for key in ("hoct", "tracksdata", "spatial-graph", "torch")},
               "ground_truth_read": False, "fallback_used": False, "passed": True}
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def run(args):
    names = reference.selected_names(args.control_dir, args.mode)
    if geometry.digest(args.control_csv) != geometry.CONTROL_SHA256:
        raise ValueError("Frozen E029 CSV changed")
    flow.require_replay_parity(args.replayed_control_csv, args.control_csv)
    graphs = sorted(args.control_dir.glob("*.geff"))
    if flow.graph_tree_sha256(graphs) != CONTROL_GRAPH_SHA256:
        raise ValueError("Official E029 control graph bytes changed")
    frame = pd.read_csv(args.control_csv)
    frame = frame.loc[frame.dataset.isin(names)].copy()
    manifest = {"experiments": ["E036", "E037"], "mode": args.mode, "datasets": names,
                "control_sha256": geometry.CONTROL_SHA256, "control_graph_sha256": CONTROL_GRAPH_SHA256,
                "checkpoint_sha256": geometry.WEIGHT_SHA256, "hoct_version": "0.2.0",
                "radius_um": 3.0, "max_delta_t": 1, "tile": [5, 32, 128, 128],
                "overlap": [1, 8, 16, 16], "solver": "HOCT 0.2.0 defaults",
                "max_seconds_per_movie": 900, "minimum_pooled_delta": 0.001,
                "collision_policy": "one lowest-id representative per exact TZYX position; preserve all edges incident to every coincident node",
                "runtime_parameter_search": False, "all_e029_nodes_preserved": True,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tasks = {}
    for name in names:
        folder = args.output_dir / "inference" / name
        folder.mkdir(parents=True)
        nodes = frame.loc[(frame.dataset == name) & (frame.row_type == "node")].sort_values("node_id")
        np.savez_compressed(folder / "input_nodes.npz", ids=nodes.node_id.to_numpy(dtype=np.int64),
                            points=nodes[["t", "z", "y", "x"]].to_numpy(dtype=float))
        tasks[name] = folder
    stop = threading.Event()

    def shard(index):
        for name in names[index::2]:
            if stop.is_set():
                raise RuntimeError("Peer shard failed; no partial-corpus quality claim")
            folder = tasks[name]
            command = ["timeout", "900", sys.executable, str(Path(__file__).resolve()), "infer",
                       "--input-nodes", str(folder / "input_nodes.npz"),
                       "--input-sha256", geometry.digest(folder / "input_nodes.npz"),
                       "--image", str(args.data_dir / "train" / f"{name}.zarr"),
                       "--weight", str(args.data_dir / "hoct-general-v0/general_v0.pt"),
                       "--output-dir", str(folder)]
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(index)}
            try:
                with (folder / "inference.log").open("x") as handle:
                    subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT,
                                   check=True, start_new_session=True)
                receipt = json.loads((folder / "receipt.json").read_text())
                if (receipt.get("passed") is not True or receipt.get("fallback_used") is not False
                        or receipt["pairs_sha256"] != geometry.digest(folder / "pairs.npz")
                        or receipt["input_sha256"] != geometry.digest(folder / "input_nodes.npz")):
                    raise ValueError("Incomplete or changed HOCT prediction evidence")
                print(f"INFERRED {name} nodes={receipt['solution_nodes']}/{receipt['input_nodes']} edges={receipt['edges']}", flush=True)
            except Exception:
                stop.set()
                raise

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(shard, index) for index in (0, 1)]
        for future in futures:
            future.result()
    for experiment, protect in (("E036", True), ("E037", False)):
        current = copy.copy(args)
        current.output_dir = args.output_dir / experiment.lower()
        image_dir = current.output_dir / "input/test"
        image_dir.mkdir(parents=True)
        keep = frame.row_type.eq("node").copy()
        counts = []
        for name in names:
            (image_dir / f"{name}.zarr").symlink_to(args.data_dir / "train" / f"{name}.zarr", target_is_directory=True)
            edges = frame.loc[(frame.dataset == name) & (frame.row_type == "edge")]
            pairs = list(zip(edges.source_id.astype(int), edges.target_id.astype(int)))
            with np.load(tasks[name] / "pairs.npz", allow_pickle=False) as data:
                consensus = set(map(tuple, data["pairs"].tolist()))
            receipt = json.loads((tasks[name] / "receipt.json").read_text())
            ambiguous = set(receipt["ambiguous_node_ids"])
            mask = retain_edges(pairs, consensus, protect, ambiguous)
            keep.loc[edges.index] = mask
            counts.append({"dataset": name, "edges_before": len(pairs), "edges_after": sum(mask),
                           "removed_edges": len(pairs) - sum(mask), "ambiguous_nodes": len(ambiguous),
                           "protected_ambiguous_incident_edges": sum(a in ambiguous or b in ambiguous for a, b in pairs)})
        candidate = frame.loc[keep].copy()
        original_nodes = frame.loc[frame.row_type == "node"].drop(columns="id")
        candidate_nodes = candidate.loc[candidate.row_type == "node"].drop(columns="id")
        if not original_nodes.equals(candidate_nodes):
            raise ValueError("Veto changed frozen nodes")
        candidate["id"] = np.arange(len(candidate))
        source_csv = current.output_dir / "raw_submission.csv"
        candidate.to_csv(source_csv, index=False)
        current.experiment = f"{experiment} HOCT consensus, protect_divisions={protect}"
        current.expected_control_score = flow.E029_SCORE
        current.minimum_pooled_delta = 0.001
        proof = {**manifest, "experiment": experiment, "protect_divisions": protect,
                 "inference_receipts": {name: geometry.digest(tasks[name] / "receipt.json") for name in names},
                 "per_movie_counts": counts, "all_nodes_exactly_preserved": True}
        reference.complete_export(current, names, proof, source_csv)
        result = json.loads((current.output_dir / "stability.json").read_text())
        (current.output_dir / "run_summary.md").write_text(
            f"# {experiment} HOCT edge intersection\n\nAll E029 nodes unchanged. "
            f"Protect divisions: {protect}. Development only.\n\n```json\n" +
            json.dumps({"groups": result["groups"], "gates": result["gates"], "paired": result["paired"]}, indent=2) + "\n```\n")
    if (geometry.digest(args.control_csv) != geometry.CONTROL_SHA256
            or flow.graph_tree_sha256(graphs) != CONTROL_GRAPH_SHA256):
        raise ValueError("Frozen controls changed during evaluation")
    (args.output_dir / "run_summary.md").write_text(
        "# E036/E037 comparison completed\n\nAll fixed movies processed without fallback; "
        "both arms independently scored against unchanged E029. See each arm's stability.json. "
        "No package or formal submission was created.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inference = sub.add_parser("infer")
    for key in ("input-nodes", "image", "weight", "output-dir"):
        inference.add_argument("--" + key, type=Path, required=True)
    inference.add_argument("--input-sha256", required=True)
    experiment = sub.add_parser("run")
    for key in ("data-dir", "control-dir", "control-csv", "replayed-control-csv", "runtime-dir", "scorer-dir", "output-dir"):
        experiment.add_argument("--" + key, type=Path, required=True)
    experiment.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.command == "infer":
        infer_movie(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as exc:
        (args.output_dir / "run_summary.md").write_text(f"# HOCT comparison failed\n\n{type(exc).__name__}: {exc}\n\nNo fallback or promotion.\n")
        raise


if __name__ == "__main__":
    main()
