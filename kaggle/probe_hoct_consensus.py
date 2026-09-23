#!/usr/bin/env python3
"""Bounded, label-free HOCT v0 compatibility check on frozen E029 nodes.

Method reference: https://www.kaggle.com/code/sjlee101/biohub-lf-hoctveto-div-b
This independently written probe uses two fixed movies and their first five
frames. It establishes executable compatibility, never a tracking-quality gain.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.spatial import cKDTree

NAMES = ("44b6_eb2880fc", "6bba_969618f6")
FRAMES = 5
SCALE = np.array((1.625, 0.40625, 0.40625))
RADIUS = 3.0
CONTROL_SHA256 = "1d4fd28c02cb54d2279a120794b26e21fa0741d743bf62781d8ed17d5a26e2ce"
WEIGHT_SHA256 = "024c2e4606275c96667907abfc9e0c27487b543480caf99d9ebd1d267cef8e4a"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rasterize(points, shape):
    """Physical 3-um balls; overlap belongs to the nearest input centre."""
    if (points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all()
            or np.any(points < 0) or np.any(points >= np.asarray(shape))
            or np.any(points[:, 0] != points[:, 0].astype(int))):
        raise ValueError("Invalid TZYX input coordinates")
    labels = np.zeros(shape, dtype=np.int16)
    reach = np.ceil(RADIUS / SCALE).astype(int)
    for frame in range(shape[0]):
        centres = points[points[:, 0] == frame, 1:]
        if len(centres) > np.iinfo(np.int16).max:
            raise ValueError("Too many per-frame labels")
        nearest = np.full(shape[1:], np.inf, dtype=np.float32)
        for label, centre in enumerate(centres, 1):
            lo = np.maximum(0, np.floor(centre).astype(int) - reach)
            hi = np.minimum(shape[1:], np.ceil(centre).astype(int) + reach + 1)
            axes = [(np.arange(a, b) - c) * s for a, b, c, s in zip(lo, hi, centre, SCALE)]
            d2 = (axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2
                  + axes[2][None, None, :] ** 2).astype(np.float32)
            box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            closest = nearest[box]
            selected = (d2 <= RADIUS ** 2) & (d2 < closest)
            closest[selected] = d2[selected]
            labels[frame][box][selected] = label
        observed = np.unique(labels[frame])
        observed = observed[observed != 0]
        if not np.array_equal(observed, np.arange(1, len(centres) + 1)):
            raise ValueError(f"Rasterization lost an input node in frame {frame}")
    return labels


def snap_nodes(output_points, input_points):
    """Require bounded, one-to-one physical correspondence within each frame."""
    if (len(output_points) == 0 or not np.isfinite(output_points).all()
            or np.any(output_points[:, 0] != output_points[:, 0].astype(int))):
        raise ValueError("Invalid or empty HOCT node coordinates")
    mapping = np.full(len(output_points), -1, dtype=int)
    distances = np.zeros(len(output_points))
    for frame in np.unique(output_points[:, 0]):
        target = np.flatnonzero(output_points[:, 0] == frame)
        source = np.flatnonzero(input_points[:, 0] == frame)
        if not len(source):
            raise ValueError("HOCT produced an unexpected frame")
        distance, index = cKDTree(input_points[source, 1:] * SCALE).query(
            output_points[target, 1:] * SCALE)
        mapping[target], distances[target] = source[index], distance
    if distances.max() > RADIUS + 1e-6 or len(np.unique(mapping)) != len(mapping):
        raise ValueError("HOCT node correspondence is distant or ambiguous")
    return mapping, float(distances.max())


def load_nodes(path):
    if digest(path) != CONTROL_SHA256:
        raise ValueError("Frozen E029 export changed")
    nodes = {name: [] for name in NAMES}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] in nodes and row["row_type"] == "node" and 0 <= int(row["t"]) < FRAMES:
                nodes[row["dataset"]].append([int(row["node_id"]), *[float(row[k]) for k in ("t", "z", "y", "x")]])
    return {name: np.asarray(sorted(rows), dtype=float) for name, rows in nodes.items()}


def run(args):
    import torch
    import zarr
    from hoct import load_model, predict
    from tracksdata.functional import TilingScheme

    weight = args.data_dir / "hoct-general-v0/general_v0.pt"
    if digest(weight) != WEIGHT_SHA256 or importlib.metadata.version("hoct") != "0.2.0":
        raise ValueError("Frozen HOCT version or checkpoint changed")
    if not torch.cuda.is_available():
        raise RuntimeError("The bounded probe requires an available GPU")
    inputs = load_nodes(args.control_csv)
    model = load_model(weight, device="cuda")
    report = {"kind": "compatibility_only", "quality_score": None, "ground_truth_read": False,
              "movies": list(NAMES), "frames": FRAMES, "scale_tzyx": [1.0, *SCALE.tolist()],
              "radius_um": RADIUS, "max_delta_t": 1, "test_time_augs": 0,
              "tile": [5, 32, 128, 128], "overlap": [1, 8, 16, 16],
              "solver_configuration": "HOCT 0.2.0 defaults; outer process timeout 900 seconds",
              "checkpoint_sha256": WEIGHT_SHA256, "control_sha256": CONTROL_SHA256,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "versions": {name: importlib.metadata.version(name) for name in ("hoct", "tracksdata", "spatial-graph", "torch")},
              "results": []}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in NAMES:
        started = time.monotonic()
        data = inputs[name]
        if len(data) == 0:
            raise ValueError(f"No frozen nodes for {name}")
        ids, points = data[:, 0].astype(np.int64), data[:, 1:]
        images = np.asarray(zarr.open_group(str(args.data_dir / "train" / f"{name}.zarr"), mode="r")["0"][:FRAMES])
        if images.shape != (FRAMES, 64, 256, 256) or images.dtype != np.uint16:
            raise ValueError("Unexpected raw image contract")
        labels = rasterize(points, images.shape)
        painted = time.monotonic()
        with torch.inference_mode():
            solution = predict(model, labels=labels, images=images, scale=(1.0, *SCALE), max_delta_t=1,
                               tiling_scheme=TilingScheme(tile_shape=(5, 32, 128, 128), overlap_shape=(1, 8, 16, 16)))
        nodes = solution.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        edges = solution.edge_attrs(attr_keys=[])
        out_points = np.column_stack([nodes[key].to_numpy() for key in ("t", "z", "y", "x")])
        mapping, max_distance = snap_nodes(out_points, points)
        output_ids = nodes["node_id"].to_list()
        remap = {int(node): int(ids[index]) for node, index in zip(output_ids, mapping)}
        pairs = [(remap[int(a)], remap[int(b)]) for a, b in zip(edges["source_id"].to_list(), edges["target_id"].to_list())]
        source_degree, target_degree = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
        times = {int(node): int(point[0]) for node, point in zip(ids, points)}
        if (len(pairs) == 0 or len(pairs) != len(set(pairs))
                or any(times[b] - times[a] != 1 for a, b in pairs)
                or max(source_degree.values(), default=0) > 2 or max(target_degree.values(), default=0) > 1):
            raise ValueError("HOCT solution violates expected temporal topology")
        result = {"movie": name, "input_nodes": len(ids), "solution_nodes": len(mapping),
                  "complete_node_coverage": len(mapping) == len(ids), "edges": len(pairs),
                  "max_snap_distance_um": max_distance, "prepare_seconds": painted - started,
                  "predict_and_audit_seconds": time.monotonic() - painted,
                  "cuda_peak_bytes": torch.cuda.max_memory_allocated()}
        report["results"].append(result)
        (args.output_dir / "probe_results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        del solution, labels, images
        torch.cuda.empty_cache()
    report["passed"] = all(row["complete_node_coverage"] for row in report["results"])
    report["full_scale_solver_tested"] = False
    (args.output_dir / "probe_results.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise ValueError("HOCT did not retain all frozen nodes in the compatibility probe")
    (args.output_dir / "run_summary.md").write_text(
        "# HOCT compatibility probe passed\n\nTwo fixed five-frame real-image runs pass raster coverage, "
        "bounded one-to-one node mapping and temporal topology checks. No quality evaluation or "
        "full-movie solver/runtime claim is established.\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data-dir", "control-csv", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as exc:
        (args.output_dir / "run_summary.md").write_text(f"# HOCT compatibility probe failed\n\n{type(exc).__name__}: {exc}\n\nNo quality or promotion evidence.\n")
        raise


if __name__ == "__main__":
    main()
