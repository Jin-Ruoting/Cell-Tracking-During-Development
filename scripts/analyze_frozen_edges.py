#!/usr/bin/env python3
"""Compare predicted edge motion on frozen and ordinary frame transitions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


@dataclass(frozen=True)
class Node:
    t: int
    z: int
    y: int
    x: int

    @property
    def position(self) -> tuple[int, int, int]:
        return (self.z, self.y, self.x)


def edge_distance_um(source: Node, target: Node) -> float:
    return math.sqrt(
        sum(
            ((left - right) * scale) ** 2
            for left, right, scale in zip(
                source.position,
                target.position,
                VOXEL_SCALE_UM,
            )
        )
    )


def quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[index]


def distance_summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    count = len(ordered)
    return {
        "edges": count,
        "mean_um": sum(ordered) / count if count else 0.0,
        "p50_um": quantile(ordered, 0.50),
        "p90_um": quantile(ordered, 0.90),
        "p95_um": quantile(ordered, 0.95),
        "p99_um": quantile(ordered, 0.99),
        "max_um": ordered[-1] if ordered else 0.0,
        "exact_zero_edges": sum(value == 0.0 for value in ordered),
        "within_0_5_um": sum(value <= 0.5 for value in ordered),
        "within_1_um": sum(value <= 1.0 for value in ordered),
        "within_3_um": sum(value <= 3.0 for value in ordered),
        "over_7_um": sum(value > 7.0 for value in ordered),
    }


def load_frozen_pairs(report_path: Path) -> dict[str, set[tuple[int, int]]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        str(row["clip"]): {
            (int(pair["source_t"]), int(pair["target_t"]))
            for pair in row["duplicate_pairs"]
        }
        for row in report["clips"]
    }


def load_predictions(
    predictions_path: Path,
) -> tuple[
    dict[str, dict[int, Node]],
    dict[str, list[tuple[int, int]]],
]:
    nodes: dict[str, dict[int, Node]] = defaultdict(dict)
    edges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset = row["dataset"]
            if row["row_type"] == "node":
                nodes[dataset][int(row["node_id"])] = Node(
                    t=int(row["t"]),
                    z=int(row["z"]),
                    y=int(row["y"]),
                    x=int(row["x"]),
                )
            elif row["row_type"] == "edge":
                edges[dataset].append(
                    (int(row["source_id"]), int(row["target_id"]))
                )
    return nodes, edges


def _transition_detail(
    dataset: str,
    pair: tuple[int, int],
    nodes: dict[int, Node],
    edge_rows: list[tuple[int, int, float]],
) -> dict[str, Any]:
    source_t, target_t = pair
    source_positions = Counter(
        node.position for node in nodes.values() if node.t == source_t
    )
    target_positions = Counter(
        node.position for node in nodes.values() if node.t == target_t
    )
    exact_overlap = sum((source_positions & target_positions).values())
    distances = [
        distance
        for edge_source_t, edge_target_t, distance in edge_rows
        if (edge_source_t, edge_target_t) == pair
    ]
    return {
        "dataset": dataset,
        "embryo": dataset.split("_", 1)[0],
        "source_t": source_t,
        "target_t": target_t,
        "source_nodes": sum(source_positions.values()),
        "target_nodes": sum(target_positions.values()),
        "exact_position_overlap": exact_overlap,
        "source_overlap_fraction": (
            exact_overlap / sum(source_positions.values())
            if source_positions
            else 0.0
        ),
        "target_overlap_fraction": (
            exact_overlap / sum(target_positions.values())
            if target_positions
            else 0.0
        ),
        "predicted_edges": len(distances),
        "exact_zero_edges": sum(distance == 0.0 for distance in distances),
        "exact_zero_edge_fraction": (
            sum(distance == 0.0 for distance in distances) / len(distances)
            if distances
            else 0.0
        ),
        "edge_distance": distance_summary(distances),
    }


def _aggregate_transition_details(
    details: list[dict[str, Any]],
) -> dict[str, float | int]:
    source_nodes = sum(int(row["source_nodes"]) for row in details)
    target_nodes = sum(int(row["target_nodes"]) for row in details)
    overlap = sum(int(row["exact_position_overlap"]) for row in details)
    edges = sum(int(row["predicted_edges"]) for row in details)
    zero_edges = sum(int(row["exact_zero_edges"]) for row in details)
    return {
        "transitions": len(details),
        "source_nodes": source_nodes,
        "target_nodes": target_nodes,
        "exact_position_overlap": overlap,
        "source_overlap_fraction": overlap / source_nodes if source_nodes else 0.0,
        "target_overlap_fraction": overlap / target_nodes if target_nodes else 0.0,
        "predicted_edges": edges,
        "exact_zero_edges": zero_edges,
        "exact_zero_edge_fraction": zero_edges / edges if edges else 0.0,
    }


def analyze_frozen_edges(
    predictions_path: Path,
    frozen_report_path: Path,
) -> dict[str, Any]:
    frozen_pairs = load_frozen_pairs(frozen_report_path)
    nodes_by_dataset, edges_by_dataset = load_predictions(predictions_path)
    datasets = sorted(set(nodes_by_dataset) | set(edges_by_dataset))

    distances_by_dataset: dict[str, dict[str, list[float]]] = {}
    all_distances = {"frozen": [], "normal": []}
    embryo_distances: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"frozen": [], "normal": []}
    )
    transition_details: list[dict[str, Any]] = []

    for dataset in datasets:
        nodes = nodes_by_dataset[dataset]
        pairs = frozen_pairs.get(dataset, set())
        distance_groups = {"frozen": [], "normal": []}
        edge_rows: list[tuple[int, int, float]] = []

        for source_id, target_id in edges_by_dataset[dataset]:
            source = nodes[source_id]
            target = nodes[target_id]
            distance = edge_distance_um(source, target)
            pair = (source.t, target.t)
            group = "frozen" if pair in pairs else "normal"
            distance_groups[group].append(distance)
            all_distances[group].append(distance)
            embryo_distances[dataset.split("_", 1)[0]][group].append(distance)
            edge_rows.append((source.t, target.t, distance))

        distances_by_dataset[dataset] = distance_groups
        for pair in sorted(pairs):
            transition_details.append(
                _transition_detail(dataset, pair, nodes, edge_rows)
            )

    per_dataset = {
        dataset: {
            group: distance_summary(values)
            for group, values in distances_by_dataset[dataset].items()
        }
        for dataset in datasets
    }
    by_embryo = {
        embryo: {
            group: distance_summary(values)
            for group, values in groups.items()
        }
        for embryo, groups in sorted(embryo_distances.items())
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions": str(predictions_path.resolve()),
        "frozen_report": str(frozen_report_path.resolve()),
        "datasets": datasets,
        "overall": {
            group: distance_summary(values)
            for group, values in all_distances.items()
        },
        "by_embryo": by_embryo,
        "per_dataset": per_dataset,
        "frozen_detection_and_link_summary": _aggregate_transition_details(
            transition_details
        ),
        "frozen_transition_details": transition_details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("frozen_report", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_frozen_edges(args.predictions, args.frozen_report)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(json.dumps({
            "overall": report["overall"],
            "by_embryo": report["by_embryo"],
            "frozen_detection_and_link_summary":
                report["frozen_detection_and_link_summary"],
        }, indent=2, sort_keys=True))
        print(f"Wrote {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
