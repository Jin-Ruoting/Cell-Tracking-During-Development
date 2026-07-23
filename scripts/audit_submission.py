#!/usr/bin/env python3
"""Validate a Biohub submission and report graph-connectivity statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_COLUMNS = [
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
INTEGER_COLUMNS = ["id", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


@dataclass(frozen=True)
class Node:
    t: int
    z: int
    y: int
    x: int


class Violations:
    def __init__(self, max_examples: int = 5) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: dict[str, list[str]] = defaultdict(list)
        self.max_examples = max_examples

    def add(self, category: str, detail: str) -> None:
        self.counts[category] += 1
        if len(self.examples[category]) < self.max_examples:
            self.examples[category].append(detail)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": sum(self.counts.values()),
            "counts": dict(sorted(self.counts.items())),
            "examples": dict(sorted(self.examples.items())),
        }


class UnionFind:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        while value != parent:
            next_value = self.parent[value]
            self.parent[value] = parent
            value = next_value
        return parent

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]


def _parse_integers(
    row: dict[str, str],
    row_number: int,
    violations: Violations,
) -> dict[str, int] | None:
    parsed: dict[str, int] = {}
    for column in INTEGER_COLUMNS:
        try:
            parsed[column] = int(row[column])
        except (KeyError, TypeError, ValueError):
            violations.add(
                "invalid_integer",
                f"row={row_number} column={column} value={row.get(column)!r}",
            )
            return None
    return parsed


def _resolve_array_path(data_root: Path, dataset: str) -> Path | None:
    candidates = [
        data_root / "test" / f"{dataset}.zarr" / "0" / "zarr.json",
        data_root / f"{dataset}.zarr" / "0" / "zarr.json",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _read_shape(
    data_root: Path,
    dataset: str,
    violations: Violations,
) -> tuple[int, int, int, int] | None:
    metadata_path = _resolve_array_path(data_root, dataset)
    if metadata_path is None:
        violations.add("missing_dataset_array", dataset)
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shape = tuple(int(value) for value in metadata["shape"])
    if len(shape) != 4:
        violations.add("invalid_dataset_shape", f"{dataset}: {shape}")
        return None
    return shape


def _quantile(sorted_values: list[int], quantile: float) -> int:
    if not sorted_values:
        return 0
    index = max(0, math.ceil(quantile * len(sorted_values)) - 1)
    return sorted_values[index]


def _dataset_stats(
    dataset: str,
    nodes: dict[int, Node],
    edges: list[tuple[int, int]],
    violations: Violations,
    shape: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    indegree: Counter[int] = Counter()
    outdegree: Counter[int] = Counter()
    union_find = UnionFind(list(nodes))

    for source_id, target_id in edges:
        source = nodes.get(source_id)
        target = nodes.get(target_id)
        if source is None or target is None:
            violations.add(
                "dangling_edge",
                f"{dataset}: {source_id}->{target_id}",
            )
            continue
        if source_id == target_id:
            violations.add("self_loop", f"{dataset}: {source_id}")
        if target.t != source.t + 1:
            violations.add(
                "nonconsecutive_edge",
                f"{dataset}: {source_id}@{source.t}->{target_id}@{target.t}",
            )
        indegree[target_id] += 1
        outdegree[source_id] += 1
        union_find.union(source_id, target_id)

    for node_id, count in indegree.items():
        if count > 1:
            violations.add(
                "multiple_parents",
                f"{dataset}: node={node_id} indegree={count}",
            )
    for node_id, count in outdegree.items():
        if count > 2:
            violations.add(
                "excessive_outdegree",
                f"{dataset}: node={node_id} outdegree={count}",
            )

    if shape is not None:
        for node_id, node in nodes.items():
            values = (node.t, node.z, node.y, node.x)
            if any(value < 0 or value >= limit for value, limit in zip(values, shape)):
                violations.add(
                    "out_of_bounds_node",
                    f"{dataset}: node={node_id} tzyx={values} shape={shape}",
                )

    component_sizes: Counter[int] = Counter()
    for node_id in nodes:
        component_sizes[union_find.find(node_id)] += 1
    sorted_sizes = sorted(component_sizes.values())
    node_count = len(nodes)
    edge_count = len(edges)

    return {
        "dataset": dataset,
        "nodes": node_count,
        "edges": edge_count,
        "edge_to_node_ratio": edge_count / max(node_count, 1),
        "components": len(sorted_sizes),
        "singleton_components": sum(size == 1 for size in sorted_sizes),
        "median_component_nodes": (
            sorted_sizes[len(sorted_sizes) // 2] if sorted_sizes else 0
        ),
        "p95_component_nodes": _quantile(sorted_sizes, 0.95),
        "largest_component_nodes": sorted_sizes[-1] if sorted_sizes else 0,
        "largest_component_fraction": (
            sorted_sizes[-1] / node_count if sorted_sizes and node_count else 0.0
        ),
        "division_like_sources": sum(count == 2 for count in outdegree.values()),
        "max_indegree": max(indegree.values(), default=0),
        "max_outdegree": max(outdegree.values(), default=0),
    }


def audit_submission(
    submission_path: Path,
    data_root: Path | None = None,
) -> dict[str, Any]:
    violations = Violations()
    nodes_by_dataset: dict[str, dict[int, Node]] = defaultdict(dict)
    edges_by_dataset: dict[str, list[tuple[int, int]]] = defaultdict(list)
    edge_sets: dict[str, set[tuple[int, int]]] = defaultdict(set)
    seen_row_ids: set[int] = set()
    rows = 0

    with submission_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"unexpected CSV columns: expected {EXPECTED_COLUMNS}, "
                f"got {reader.fieldnames}"
            )

        for rows, row in enumerate(reader, start=1):
            parsed = _parse_integers(row, rows, violations)
            if parsed is None:
                continue
            row_id = parsed["id"]
            if row_id in seen_row_ids:
                violations.add("duplicate_row_id", f"row={rows} id={row_id}")
            seen_row_ids.add(row_id)
            if row_id != rows - 1:
                violations.add(
                    "nonsequential_row_id",
                    f"row={rows} id={row_id} expected={rows - 1}",
                )

            dataset = row["dataset"].strip()
            if not dataset:
                violations.add("empty_dataset", f"row={rows}")
                continue

            row_type = row["row_type"]
            if row_type == "node":
                node_id = parsed["node_id"]
                if node_id < 0:
                    violations.add(
                        "negative_node_id",
                        f"{dataset}: row={rows} node={node_id}",
                    )
                if parsed["source_id"] != -1 or parsed["target_id"] != -1:
                    violations.add(
                        "invalid_node_sentinel",
                        f"{dataset}: row={rows}",
                    )
                node = Node(
                    t=parsed["t"],
                    z=parsed["z"],
                    y=parsed["y"],
                    x=parsed["x"],
                )
                if min(node.t, node.z, node.y, node.x) < 0:
                    violations.add(
                        "negative_node_coordinate",
                        f"{dataset}: node={node_id} node={node}",
                    )
                if node_id in nodes_by_dataset[dataset]:
                    violations.add(
                        "duplicate_node_id",
                        f"{dataset}: node={node_id}",
                    )
                else:
                    nodes_by_dataset[dataset][node_id] = node
            elif row_type == "edge":
                if any(
                    parsed[column] != -1
                    for column in ("node_id", "t", "z", "y", "x")
                ):
                    violations.add(
                        "invalid_edge_sentinel",
                        f"{dataset}: row={rows}",
                    )
                edge = (parsed["source_id"], parsed["target_id"])
                if min(edge) < 0:
                    violations.add(
                        "negative_edge_endpoint",
                        f"{dataset}: edge={edge}",
                    )
                if edge in edge_sets[dataset]:
                    violations.add(
                        "duplicate_edge",
                        f"{dataset}: edge={edge}",
                    )
                else:
                    edge_sets[dataset].add(edge)
                    edges_by_dataset[dataset].append(edge)
            else:
                violations.add(
                    "unknown_row_type",
                    f"row={rows} value={row_type!r}",
                )

    datasets = sorted(set(nodes_by_dataset) | set(edges_by_dataset))
    per_dataset = []
    for dataset in datasets:
        shape = (
            _read_shape(data_root, dataset, violations)
            if data_root is not None
            else None
        )
        per_dataset.append(
            _dataset_stats(
                dataset,
                nodes_by_dataset[dataset],
                edges_by_dataset[dataset],
                violations,
                shape,
            )
        )

    violation_report = violations.as_dict()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "submission": str(submission_path.resolve()),
        "submission_sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(),
        "rows": rows,
        "nodes": sum(len(nodes) for nodes in nodes_by_dataset.values()),
        "edges": sum(len(edges) for edges in edges_by_dataset.values()),
        "datasets": len(datasets),
        "valid": violation_report["total"] == 0,
        "violations": violation_report,
        "per_dataset": per_dataset,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional competition root for tzyx coordinate-bound checks.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report path; otherwise print the full report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_submission(args.submission, args.data_root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "submission_sha256",
                        "rows",
                        "nodes",
                        "edges",
                        "datasets",
                        "valid",
                        "violations",
                        "per_dataset",
                    )
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"Wrote {args.report}")
    else:
        print(rendered, end="")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
