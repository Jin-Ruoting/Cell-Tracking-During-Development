#!/usr/bin/env python3
"""Replace links on content-detected frozen transitions with zero-motion LAP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


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
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)
AssignmentSolver = Callable[
    [Sequence[Sequence[float]]],
    Sequence[tuple[int, int]],
]


def load_frozen_pairs(report_path: Path) -> dict[str, set[tuple[int, int]]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        str(row["clip"]): {
            (int(pair["source_t"]), int(pair["target_t"]))
            for pair in row["duplicate_pairs"]
        }
        for row in report["clips"]
    }


def _distance_um(
    source: tuple[float, float, float],
    target: tuple[float, float, float],
) -> float:
    return math.sqrt(
        sum(
            ((left - right) * scale) ** 2
            for left, right, scale in zip(
                source,
                target,
                VOXEL_SCALE_UM,
            )
        )
    )


def _scipy_assignment(
    costs: Sequence[Sequence[float]],
) -> list[tuple[int, int]]:
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError(
            "NumPy and SciPy are required for frozen-transition relinking"
        ) from exc

    rows, columns = linear_sum_assignment(np.asarray(costs, dtype=np.float64))
    return [
        (int(row_index), int(column_index))
        for row_index, column_index in zip(rows, columns)
    ]


def minimum_distance_matches(
    source_ids: Sequence[int],
    target_ids: Sequence[int],
    positions: dict[int, tuple[float, float, float]],
    max_distance_um: float,
    assignment_solver: AssignmentSolver | None = None,
) -> tuple[list[tuple[int, int, float]], int, int]:
    if not source_ids or not target_ids:
        return [], len(source_ids), len(target_ids)

    real_costs = [
        [
            _distance_um(positions[source_id], positions[target_id])
            for target_id in target_ids
        ]
        for source_id in source_ids
    ]
    unmatched_cost = max_distance_um + 1.0
    blocked_cost = unmatched_cost * 1000.0
    costs = [
        [
            distance if distance <= max_distance_um else blocked_cost
            for distance in row
        ]
        + [unmatched_cost] * len(source_ids)
        for row in real_costs
    ]

    solver = assignment_solver or _scipy_assignment
    assignments = solver(costs)
    matches = []
    for source_index, target_index in assignments:
        if target_index >= len(target_ids):
            continue
        distance = real_costs[source_index][target_index]
        if distance <= max_distance_um:
            matches.append(
                (
                    int(source_ids[source_index]),
                    int(target_ids[target_index]),
                    float(distance),
                )
            )

    return matches, len(source_ids) - len(matches), len(target_ids) - len(matches)


def read_submission(
    path: Path,
) -> tuple[
    list[dict[str, str]],
    dict[str, dict[int, dict[str, Any]]],
]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(
                f"unexpected CSV header: {reader.fieldnames!r}; "
                f"expected {CSV_COLUMNS!r}"
            )
        rows = list(reader)

    nodes: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["row_type"] != "node":
            continue
        node_id = int(row["node_id"])
        nodes[row["dataset"]][node_id] = {
            "t": int(row["t"]),
            "position": (
                float(row["z"]),
                float(row["y"]),
                float(row["x"]),
            ),
        }
    return rows, nodes


def relink_rows(
    rows: list[dict[str, str]],
    nodes_by_dataset: dict[str, dict[int, dict[str, Any]]],
    frozen_pairs: dict[str, set[tuple[int, int]]],
    max_distance_um: float,
    assignment_solver: AssignmentSolver | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    replacement_edges: dict[str, list[tuple[int, int, float]]] = defaultdict(
        list
    )
    dataset_stats: dict[str, dict[str, Any]] = {}

    for dataset, pairs in sorted(frozen_pairs.items()):
        nodes = nodes_by_dataset.get(dataset)
        if not nodes or not pairs:
            continue

        ids_by_time: dict[int, list[int]] = defaultdict(list)
        positions: dict[int, tuple[float, float, float]] = {}
        for node_id, node in nodes.items():
            ids_by_time[int(node["t"])].append(node_id)
            positions[node_id] = node["position"]
        for node_ids in ids_by_time.values():
            node_ids.sort()

        stats = {
            "frozen_transitions": len(pairs),
            "matched_edges": 0,
            "exact_zero_edges": 0,
            "unmatched_sources": 0,
            "unmatched_targets": 0,
            "distance_sum_um": 0.0,
            "max_distance_um": 0.0,
        }
        for source_t, target_t in sorted(pairs):
            matches, unmatched_sources, unmatched_targets = (
                minimum_distance_matches(
                    ids_by_time.get(source_t, []),
                    ids_by_time.get(target_t, []),
                    positions,
                    max_distance_um,
                    assignment_solver=assignment_solver,
                )
            )
            replacement_edges[dataset].extend(matches)
            distances = [distance for _, _, distance in matches]
            stats["matched_edges"] += len(matches)
            stats["exact_zero_edges"] += sum(
                distance == 0.0 for distance in distances
            )
            stats["unmatched_sources"] += unmatched_sources
            stats["unmatched_targets"] += unmatched_targets
            stats["distance_sum_um"] += sum(distances)
            if distances:
                stats["max_distance_um"] = max(
                    float(stats["max_distance_um"]),
                    max(distances),
                )
        dataset_stats[dataset] = stats

    removed_by_dataset: dict[str, int] = defaultdict(int)
    retained_rows: list[dict[str, str]] = []
    for row in rows:
        if row["row_type"] != "edge":
            retained_rows.append(dict(row))
            continue

        dataset = row["dataset"]
        nodes = nodes_by_dataset.get(dataset, {})
        source = nodes.get(int(row["source_id"]))
        target = nodes.get(int(row["target_id"]))
        pair = (
            int(source["t"]),
            int(target["t"]),
        ) if source is not None and target is not None else None
        if pair in frozen_pairs.get(dataset, set()):
            removed_by_dataset[dataset] += 1
            continue
        retained_rows.append(dict(row))

    for dataset in sorted(replacement_edges):
        for source_id, target_id, _ in replacement_edges[dataset]:
            retained_rows.append(
                {
                    "id": "-1",
                    "dataset": dataset,
                    "row_type": "edge",
                    "node_id": "-1",
                    "t": "-1",
                    "z": "-1",
                    "y": "-1",
                    "x": "-1",
                    "source_id": str(source_id),
                    "target_id": str(target_id),
                }
            )

    for row_id, row in enumerate(retained_rows):
        row["id"] = str(row_id)

    for dataset, stats in dataset_stats.items():
        stats["removed_edges"] = removed_by_dataset.get(dataset, 0)
        matched = int(stats["matched_edges"])
        stats["mean_distance_um"] = (
            float(stats.pop("distance_sum_um")) / matched
            if matched
            else 0.0
        )

    totals = {
        key: sum(
            int(stats.get(key, 0))
            for stats in dataset_stats.values()
        )
        for key in (
            "frozen_transitions",
            "removed_edges",
            "matched_edges",
            "exact_zero_edges",
            "unmatched_sources",
            "unmatched_targets",
        )
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "max_distance_um": max_distance_um,
        "datasets": dataset_stats,
        "totals": totals,
        "input_rows": len(rows),
        "output_rows": len(retained_rows),
    }
    return retained_rows, report


def write_submission(path: Path, rows: list[dict[str, str]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("frozen_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--max-distance-um",
        type=float,
        default=3.0,
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.max_distance_um <= 0:
        parser.error("--max-distance-um must be positive")
    return args


def main() -> None:
    args = parse_args()
    rows, nodes = read_submission(args.predictions)
    frozen_pairs = load_frozen_pairs(args.frozen_report)
    output_rows, report = relink_rows(
        rows,
        nodes,
        frozen_pairs,
        args.max_distance_um,
    )
    report["predictions"] = str(args.predictions.resolve())
    report["frozen_report"] = str(args.frozen_report.resolve())
    report["output"] = str(args.output.resolve())
    report["output_sha256"] = write_submission(args.output, output_rows)

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
