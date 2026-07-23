#!/usr/bin/env python3
"""Evaluate submission CSV edges with an independent patched-spec diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)
DEFAULT_MAX_MATCH_UM = 7.0
NODE_PENALTY_A = 0.1
OFFICIAL_SPEC_REPOSITORY = "royerlab/kaggle-cell-tracking-competition"
OFFICIAL_SPEC_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"
OFFICIAL_SPEC_DOCUMENT = "metrics.md"
CSV_COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


@dataclass(frozen=True)
class Node:
    t: int
    z: float
    y: float
    x: float

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.z, self.y, self.x)


@dataclass(frozen=True)
class Graph:
    nodes: dict[int, Node]
    edges: list[tuple[int, int]]
    estimated_total_nodes: float = math.nan


def _chunk_path(array_path: Path, dimensions: int) -> Path:
    return array_path.joinpath("c", *(["0"] * dimensions))


def read_single_chunk_zarr(array_path: Path):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to decode GEFF arrays") from exc

    metadata = json.loads(
        (array_path / "zarr.json").read_text(encoding="utf-8")
    )
    shape = tuple(int(value) for value in metadata["shape"])
    chunk_shape = tuple(
        int(value)
        for value in metadata["chunk_grid"]["configuration"]["chunk_shape"]
    )
    if shape != chunk_shape:
        raise ValueError(
            f"{array_path}: expected a single chunk, "
            f"shape={shape}, chunk_shape={chunk_shape}"
        )

    encoding = metadata.get("chunk_key_encoding", {})
    if (
        encoding.get("name", "default") != "default"
        or encoding.get("configuration", {}).get("separator", "/") != "/"
    ):
        raise ValueError(f"{array_path}: unsupported chunk key encoding")

    codecs = metadata.get("codecs", [])
    codec_names = [codec.get("name") for codec in codecs]
    if codec_names != ["bytes", "zstd"]:
        raise ValueError(
            f"{array_path}: expected bytes+zstd codecs, got {codec_names}"
        )
    endian = codecs[0].get("configuration", {}).get("endian")
    if endian not in ("little", None):
        raise ValueError(f"{array_path}: unsupported endian {endian!r}")

    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError("the zstd executable is required to decode GEFF")
    chunk = _chunk_path(array_path, len(shape))
    result = subprocess.run(
        [zstd, "--decompress", "--quiet", "--stdout", str(chunk)],
        check=True,
        stdout=subprocess.PIPE,
    )
    dtype = np.dtype(str(metadata["data_type"])).newbyteorder("<")
    array = np.frombuffer(result.stdout, dtype=dtype)
    expected = math.prod(shape)
    if array.size != expected:
        raise ValueError(
            f"{array_path}: decoded {array.size} values, expected {expected}"
        )
    return array.reshape(shape).copy()


def read_geff_graph(path: Path) -> Graph:
    metadata = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
    geff = metadata["attributes"]["geff"]
    estimated_value = geff.get("extra", {}).get(
        "estimated_number_of_nodes"
    )
    estimated = (
        float(estimated_value)
        if estimated_value is not None
        else math.nan
    )

    node_ids = read_single_chunk_zarr(path / "nodes" / "ids")
    properties = {
        name: read_single_chunk_zarr(
            path / "nodes" / "props" / name / "values"
        )
        for name in ("t", "z", "y", "x")
    }
    nodes = {
        int(node_id): Node(
            t=int(properties["t"][index]),
            z=float(properties["z"][index]),
            y=float(properties["y"][index]),
            x=float(properties["x"][index]),
        )
        for index, node_id in enumerate(node_ids)
    }

    edge_array = read_single_chunk_zarr(path / "edges" / "ids")
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError(f"{path}: expected an N x 2 edge array")
    edges = [
        (int(source_id), int(target_id))
        for source_id, target_id in edge_array
    ]
    return Graph(nodes=nodes, edges=edges, estimated_total_nodes=estimated)


def read_prediction_graphs(path: Path) -> dict[str, Graph]:
    nodes: dict[str, dict[int, Node]] = defaultdict(dict)
    edges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(
                f"unexpected CSV header: {reader.fieldnames!r}; "
                f"expected {CSV_COLUMNS!r}"
            )
        for row in reader:
            dataset = row["dataset"]
            if row["row_type"] == "node":
                nodes[dataset][int(row["node_id"])] = Node(
                    t=int(row["t"]),
                    z=float(row["z"]),
                    y=float(row["y"]),
                    x=float(row["x"]),
                )
            elif row["row_type"] == "edge":
                edges[dataset].append(
                    (int(row["source_id"]), int(row["target_id"]))
                )
    return {
        dataset: Graph(nodes=dataset_nodes, edges=edges.get(dataset, []))
        for dataset, dataset_nodes in nodes.items()
    }


def match_nodes(
    predicted: dict[int, Node],
    ground_truth: dict[int, Node],
    max_distance_um: float,
) -> dict[int, int]:
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError(
            "NumPy and SciPy are required for official-spec node matching"
        ) from exc

    predicted_by_time: dict[int, list[int]] = defaultdict(list)
    ground_truth_by_time: dict[int, list[int]] = defaultdict(list)
    for node_id, node in predicted.items():
        predicted_by_time[node.t].append(node_id)
    for node_id, node in ground_truth.items():
        ground_truth_by_time[node.t].append(node_id)

    scale = np.asarray(VOXEL_SCALE_UM, dtype=np.float64)
    mapping: dict[int, int] = {}
    for time in sorted(set(predicted_by_time) & set(ground_truth_by_time)):
        predicted_ids = sorted(predicted_by_time[time])
        ground_truth_ids = sorted(ground_truth_by_time[time])
        predicted_positions = np.asarray(
            [predicted[node_id].position for node_id in predicted_ids],
            dtype=np.float64,
        ) * scale
        ground_truth_positions = np.asarray(
            [ground_truth[node_id].position for node_id in ground_truth_ids],
            dtype=np.float64,
        ) * scale
        distances = np.sqrt(
            (
                (
                    ground_truth_positions[:, None, :]
                    - predicted_positions[None, :, :]
                )
                ** 2
            ).sum(axis=2)
        )
        blocked = max_distance_um * 1000.0 + 1.0
        costs = np.where(distances <= max_distance_um, distances, blocked)
        ground_rows, predicted_columns = linear_sum_assignment(costs)
        for ground_row, predicted_column in zip(
            ground_rows,
            predicted_columns,
        ):
            distance = float(distances[ground_row, predicted_column])
            if distance <= max_distance_um:
                mapping[predicted_ids[int(predicted_column)]] = (
                    ground_truth_ids[int(ground_row)]
                )
    return mapping


def normalise_edges(
    edges: Iterable[tuple[int, int]],
    nodes: dict[int, Node],
) -> list[tuple[int, int]]:
    valid = set(nodes)
    unique = sorted(
        {
            (int(source_id), int(target_id))
            for source_id, target_id in edges
            if int(source_id) != int(target_id)
            and int(source_id) in valid
            and int(target_id) in valid
            and nodes[int(target_id)].t == nodes[int(source_id)].t + 1
        }
    )

    limited: list[tuple[int, int]] = []
    outgoing_counts: dict[int, int] = defaultdict(int)
    for source_id, target_id in unique:
        if outgoing_counts[source_id] >= 2:
            continue
        limited.append((source_id, target_id))
        outgoing_counts[source_id] += 1
    return limited


def edge_counts(
    predicted_edges: Iterable[tuple[int, int]],
    ground_truth_edges: Iterable[tuple[int, int]],
    predicted_to_ground_truth: dict[int, int],
) -> tuple[int, int, int]:
    ground_truth_set = set(ground_truth_edges)
    ground_truth_successors: dict[int, list[int]] = defaultdict(list)
    ground_truth_predecessors: dict[int, list[int]] = defaultdict(list)
    for source_id, target_id in ground_truth_set:
        ground_truth_successors[source_id].append(target_id)
        ground_truth_predecessors[target_id].append(source_id)

    matched_ground_truth_edges: set[tuple[int, int]] = set()
    false_positives = 0
    for predicted_source, predicted_target in predicted_edges:
        ground_source = predicted_to_ground_truth.get(predicted_source)
        ground_target = predicted_to_ground_truth.get(predicted_target)
        mapped_edge = (
            (ground_source, ground_target)
            if ground_source is not None and ground_target is not None
            else None
        )
        if (
            mapped_edge is not None
            and mapped_edge in ground_truth_set
            and mapped_edge not in matched_ground_truth_edges
        ):
            matched_ground_truth_edges.add(mapped_edge)
            continue

        evaluable = (
            ground_source is not None
            and bool(ground_truth_successors.get(ground_source))
        ) or (
            ground_target is not None
            and bool(ground_truth_predecessors.get(ground_target))
        )
        false_positives += int(evaluable)

    true_positives = len(matched_ground_truth_edges)
    false_negatives = max(0, len(ground_truth_set) - true_positives)
    return true_positives, false_positives, false_negatives


def score_counts(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    predicted_nodes: int,
    estimated_total_nodes: float,
) -> dict[str, float | int]:
    denominator = true_positives + false_positives + false_negatives
    edge_jaccard = (
        true_positives / denominator if denominator else 1.0
    )
    if math.isfinite(estimated_total_nodes) and estimated_total_nodes > 0:
        node_delta_ratio = (
            predicted_nodes - estimated_total_nodes
        ) / estimated_total_nodes
        adjustment = 1.0 - NODE_PENALTY_A * node_delta_ratio
        adjusted = max(0.0, edge_jaccard * adjustment)
    else:
        node_delta_ratio = math.nan
        adjusted = edge_jaccard
    return {
        "edge_tp": true_positives,
        "edge_fp": false_positives,
        "edge_fn": false_negatives,
        "predicted_nodes": predicted_nodes,
        "estimated_total_nodes": estimated_total_nodes,
        "node_delta_ratio": node_delta_ratio,
        "edge_jaccard": edge_jaccard,
        "adjusted_edge_jaccard": adjusted,
        "edge_weight": denominator,
    }


def evaluate_dataset(
    predicted: Graph,
    ground_truth: Graph,
    max_distance_um: float,
) -> dict[str, float | int]:
    predicted_edges = normalise_edges(predicted.edges, predicted.nodes)
    ground_truth_edges = normalise_edges(
        ground_truth.edges,
        ground_truth.nodes,
    )
    mapping = match_nodes(
        predicted.nodes,
        ground_truth.nodes,
        max_distance_um,
    )
    counts = edge_counts(predicted_edges, ground_truth_edges, mapping)
    result = score_counts(
        *counts,
        predicted_nodes=len(predicted.nodes),
        estimated_total_nodes=ground_truth.estimated_total_nodes,
    )
    result["matched_ground_truth_nodes"] = len(set(mapping.values()))
    result["ground_truth_nodes"] = len(ground_truth.nodes)
    result["node_recall"] = (
        len(set(mapping.values())) / len(ground_truth.nodes)
        if ground_truth.nodes
        else 1.0
    )
    return result


def summarise(rows: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    total_weight = sum(int(row["edge_weight"]) for row in rows.values())
    adjusted = (
        sum(
            float(row["adjusted_edge_jaccard"])
            * int(row["edge_weight"])
            for row in rows.values()
        )
        / total_weight
        if total_weight
        else 0.0
    )
    return {
        "datasets": len(rows),
        "adjusted_edge_jaccard": adjusted,
        "edge_tp": sum(int(row["edge_tp"]) for row in rows.values()),
        "edge_fp": sum(int(row["edge_fp"]) for row in rows.values()),
        "edge_fn": sum(int(row["edge_fn"]) for row in rows.values()),
        "predicted_nodes": sum(
            int(row["predicted_nodes"]) for row in rows.values()
        ),
        "matched_ground_truth_nodes": sum(
            int(row["matched_ground_truth_nodes"]) for row in rows.values()
        ),
        "ground_truth_nodes": sum(
            int(row["ground_truth_nodes"]) for row in rows.values()
        ),
        "edge_weight": total_weight,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("ground_truth_dir", type=Path)
    parser.add_argument(
        "--datasets",
        help="Comma-separated dataset IDs; default is every prediction dataset.",
    )
    parser.add_argument(
        "--max-distance-um",
        type=float,
        default=DEFAULT_MAX_MATCH_UM,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_distance_um <= 0:
        parser.error("--max-distance-um must be positive")
    return args


def main() -> None:
    args = parse_args()
    predictions = read_prediction_graphs(args.predictions)
    datasets = (
        [value.strip() for value in args.datasets.split(",") if value.strip()]
        if args.datasets
        else sorted(predictions)
    )
    missing = sorted(set(datasets) - set(predictions))
    if missing:
        raise ValueError(f"datasets absent from predictions: {missing}")

    rows: dict[str, dict[str, float | int]] = {}
    for dataset in datasets:
        ground_truth_path = args.ground_truth_dir / f"{dataset}.geff"
        if not ground_truth_path.is_dir():
            raise FileNotFoundError(ground_truth_path)
        rows[dataset] = evaluate_dataset(
            predictions[dataset],
            read_geff_graph(ground_truth_path),
            args.max_distance_um,
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "patched_official_spec_edge_term_only",
        "exact_official_source_copy": False,
        "official_repository": OFFICIAL_SPEC_REPOSITORY,
        "official_spec_commit": OFFICIAL_SPEC_COMMIT,
        "official_spec_document": OFFICIAL_SPEC_DOCUMENT,
        "predictions": str(args.predictions.resolve()),
        "ground_truth_dir": str(args.ground_truth_dir.resolve()),
        "max_match_um": args.max_distance_um,
        "summary": summarise(rows),
        "per_dataset": rows,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
