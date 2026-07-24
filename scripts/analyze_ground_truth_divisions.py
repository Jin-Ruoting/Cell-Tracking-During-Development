#!/usr/bin/env python3
"""Summarize annotated division geometry from Biohub GEFF graphs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from scripts.evaluate_edge_predictions import Graph, Node, read_geff_graph
except ModuleNotFoundError:
    from evaluate_edge_predictions import Graph, Node, read_geff_graph


VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


@dataclass(frozen=True)
class DivisionEvent:
    dataset: str
    embryo: str
    parent_id: int
    t: int
    child_a_um: float
    child_b_um: float
    sister_um: float
    midpoint_um: float
    child_distance_delta_um: float
    predecessor_um: float | None


def physical_distance(left: Node, right: Node) -> float:
    squared = sum(
        ((a - b) * scale) ** 2
        for a, b, scale in zip(left.position, right.position, VOXEL_SCALE_UM)
    )
    return math.sqrt(squared)


def midpoint_distance(parent: Node, left: Node, right: Node) -> float:
    midpoint = tuple(
        (a + b) / 2.0 for a, b in zip(left.position, right.position)
    )
    squared = sum(
        ((value - parent_value) * scale) ** 2
        for value, parent_value, scale in zip(
            midpoint,
            parent.position,
            VOXEL_SCALE_UM,
        )
    )
    return math.sqrt(squared)


def extract_divisions(dataset: str, graph: Graph) -> list[DivisionEvent]:
    successors: dict[int, list[int]] = defaultdict(list)
    predecessors: dict[int, list[int]] = defaultdict(list)
    for source_id, target_id in graph.edges:
        if source_id not in graph.nodes or target_id not in graph.nodes:
            continue
        successors[source_id].append(target_id)
        predecessors[target_id].append(source_id)

    embryo = dataset.split("_", 1)[0]
    events: list[DivisionEvent] = []
    for parent_id, child_ids in successors.items():
        if len(child_ids) != 2:
            continue
        parent = graph.nodes[parent_id]
        left = graph.nodes[child_ids[0]]
        right = graph.nodes[child_ids[1]]
        if left.t != parent.t + 1 or right.t != parent.t + 1:
            continue
        left_um = physical_distance(parent, left)
        right_um = physical_distance(parent, right)
        pred_ids = predecessors.get(parent_id, [])
        predecessor_um = (
            physical_distance(graph.nodes[pred_ids[0]], parent)
            if len(pred_ids) == 1
            else None
        )
        events.append(
            DivisionEvent(
                dataset=dataset,
                embryo=embryo,
                parent_id=parent_id,
                t=parent.t,
                child_a_um=left_um,
                child_b_um=right_um,
                sister_um=physical_distance(left, right),
                midpoint_um=midpoint_distance(parent, left, right),
                child_distance_delta_um=abs(left_um - right_um),
                predecessor_um=predecessor_um,
            )
        )
    return events


def quantile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = list(values)
    return {
        "count": len(materialized),
        "min": min(materialized) if materialized else None,
        "p25": quantile(materialized, 0.25),
        "median": quantile(materialized, 0.50),
        "p75": quantile(materialized, 0.75),
        "p90": quantile(materialized, 0.90),
        "p95": quantile(materialized, 0.95),
        "p99": quantile(materialized, 0.99),
        "max": max(materialized) if materialized else None,
    }


def summarize_events(events: list[DivisionEvent]) -> dict[str, object]:
    child_distances = [
        distance
        for event in events
        for distance in (event.child_a_um, event.child_b_um)
    ]
    return {
        "divisions": len(events),
        "datasets_with_divisions": len({event.dataset for event in events}),
        "timepoint": distribution(event.t for event in events),
        "parent_child_um": distribution(child_distances),
        "sister_um": distribution(event.sister_um for event in events),
        "parent_to_daughter_midpoint_um": distribution(
            event.midpoint_um for event in events
        ),
        "parent_child_distance_delta_um": distribution(
            event.child_distance_delta_um for event in events
        ),
        "predecessor_parent_um": distribution(
            event.predecessor_um
            for event in events
            if event.predecessor_um is not None
        ),
    }


def analyze_directory(ground_truth_dir: Path) -> dict[str, object]:
    paths = sorted(ground_truth_dir.glob("*.geff"))
    if not paths:
        raise FileNotFoundError(
            f"no .geff directories found under {ground_truth_dir}"
        )

    all_events: list[DivisionEvent] = []
    graph_counts: dict[str, dict[str, int]] = {}
    for path in paths:
        graph = read_geff_graph(path)
        events = extract_divisions(path.stem, graph)
        all_events.extend(events)
        graph_counts[path.stem] = {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "divisions": len(events),
        }

    events_by_embryo: dict[str, list[DivisionEvent]] = defaultdict(list)
    for event in all_events:
        events_by_embryo[event.embryo].append(event)

    datasets_by_embryo = Counter(
        dataset.split("_", 1)[0] for dataset in graph_counts
    )
    return {
        "ground_truth_dir": str(ground_truth_dir.resolve()),
        "datasets": len(paths),
        "nodes": sum(row["nodes"] for row in graph_counts.values()),
        "edges": sum(row["edges"] for row in graph_counts.values()),
        "summary": summarize_events(all_events),
        "by_embryo": {
            embryo: {
                "datasets": datasets_by_embryo[embryo],
                **summarize_events(events),
            }
            for embryo, events in sorted(events_by_embryo.items())
        },
        "per_dataset": graph_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_directory(args.ground_truth_dir)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
