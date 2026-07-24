#!/usr/bin/env python3
"""Measure whether predicted nodes contain recoverable annotated divisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scripts.analyze_ground_truth_divisions import physical_distance
    from scripts.evaluate_edge_predictions import (
        Graph,
        match_nodes,
        read_geff_graph,
    )
except ModuleNotFoundError:
    from analyze_ground_truth_divisions import physical_distance
    from evaluate_edge_predictions import Graph, match_nodes, read_geff_graph


def analyze_pair(
    dataset: str,
    predicted: Graph,
    ground_truth: Graph,
    max_distance_um: float = 7.0,
    predicted_to_ground: dict[int, int] | None = None,
) -> list[dict[str, object]]:
    if predicted_to_ground is None:
        predicted_to_ground = match_nodes(
            predicted.nodes,
            ground_truth.nodes,
            max_distance_um,
        )
    ground_to_predicted = {
        ground_id: predicted_id
        for predicted_id, ground_id in predicted_to_ground.items()
    }

    gt_successors: dict[int, list[int]] = defaultdict(list)
    for source_id, target_id in ground_truth.edges:
        gt_successors[source_id].append(target_id)

    predicted_successors: dict[int, set[int]] = defaultdict(set)
    predicted_predecessors: dict[int, set[int]] = defaultdict(set)
    predicted_edges = set(predicted.edges)
    for source_id, target_id in predicted_edges:
        predicted_successors[source_id].add(target_id)
        predicted_predecessors[target_id].add(source_id)

    rows: list[dict[str, object]] = []
    for gt_parent, gt_children in sorted(gt_successors.items()):
        if len(gt_children) != 2:
            continue
        parent_node = ground_truth.nodes.get(gt_parent)
        child_nodes = [ground_truth.nodes.get(child) for child in gt_children]
        if (
            parent_node is None
            or any(node is None for node in child_nodes)
            or any(node.t != parent_node.t + 1 for node in child_nodes)
        ):
            continue

        predicted_parent = ground_to_predicted.get(gt_parent)
        predicted_children = [
            ground_to_predicted.get(child) for child in gt_children
        ]
        direct_triplet_detected = (
            predicted_parent is not None
            and all(child is not None for child in predicted_children)
            and len(set(predicted_children)) == 2
        )

        existing_correct_edges = 0
        missing_targets_are_roots = False
        addable_direct_fork = False
        geometry: dict[str, float | None] = {
            "predicted_child_a_um": None,
            "predicted_child_b_um": None,
            "predicted_sister_um": None,
        }
        if direct_triplet_detected:
            assert predicted_parent is not None
            child_a = int(predicted_children[0])
            child_b = int(predicted_children[1])
            existing_correct_edges = sum(
                (predicted_parent, child) in predicted_edges
                for child in (child_a, child_b)
            )
            missing_children = [
                child
                for child in (child_a, child_b)
                if (predicted_parent, child) not in predicted_edges
            ]
            missing_targets_are_roots = all(
                not predicted_predecessors.get(child)
                for child in missing_children
            )
            addable_direct_fork = (
                len(predicted_successors.get(predicted_parent, set())) <= 1
                and missing_targets_are_roots
            )
            predicted_parent_node = predicted.nodes[predicted_parent]
            predicted_child_a = predicted.nodes[child_a]
            predicted_child_b = predicted.nodes[child_b]
            geometry = {
                "predicted_child_a_um": physical_distance(
                    predicted_parent_node,
                    predicted_child_a,
                ),
                "predicted_child_b_um": physical_distance(
                    predicted_parent_node,
                    predicted_child_b,
                ),
                "predicted_sister_um": physical_distance(
                    predicted_child_a,
                    predicted_child_b,
                ),
            }

        rows.append(
            {
                "dataset": dataset,
                "embryo": dataset.split("_", 1)[0],
                "gt_parent_id": gt_parent,
                "t": parent_node.t,
                "predicted_parent_id": predicted_parent,
                "predicted_child_ids": predicted_children,
                "direct_triplet_detected": direct_triplet_detected,
                "existing_correct_edges": existing_correct_edges,
                "missing_targets_are_roots": missing_targets_are_roots,
                "addable_direct_fork": addable_direct_fork,
                **geometry,
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    counts = Counter(row["embryo"] for row in rows)
    return {
        "divisions": len(rows),
        "direct_triplet_detected": sum(
            bool(row["direct_triplet_detected"]) for row in rows
        ),
        "addable_direct_fork": sum(
            bool(row["addable_direct_fork"]) for row in rows
        ),
        "already_complete_fork": sum(
            int(row["existing_correct_edges"]) == 2 for row in rows
        ),
        "by_embryo": {
            embryo: {
                "divisions": count,
                "direct_triplet_detected": sum(
                    row["embryo"] == embryo
                    and bool(row["direct_triplet_detected"])
                    for row in rows
                ),
                "addable_direct_fork": sum(
                    row["embryo"] == embryo
                    and bool(row["addable_direct_fork"])
                    for row in rows
                ),
            }
            for embryo, count in sorted(counts.items())
        },
    }


def analyze_directories(
    predictions_dir: Path,
    ground_truth_dir: Path,
    max_distance_um: float = 7.0,
) -> dict[str, object]:
    prediction_paths = sorted(predictions_dir.glob("*.geff"))
    if not prediction_paths:
        raise FileNotFoundError(
            f"no prediction .geff directories found under {predictions_dir}"
        )

    rows: list[dict[str, object]] = []
    for prediction_path in prediction_paths:
        ground_truth_path = ground_truth_dir / prediction_path.name
        if not ground_truth_path.is_dir():
            continue
        rows.extend(
            analyze_pair(
                prediction_path.stem,
                read_geff_graph(prediction_path),
                read_geff_graph(ground_truth_path),
                max_distance_um,
            )
        )
    return {
        "predictions_dir": str(predictions_dir.resolve()),
        "ground_truth_dir": str(ground_truth_dir.resolve()),
        "max_distance_um": max_distance_um,
        "summary": summarize(rows),
        "events": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_dir", type=Path)
    parser.add_argument("ground_truth_dir", type=Path)
    parser.add_argument("--max-distance-um", type=float, default=7.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_directories(
        args.predictions_dir,
        args.ground_truth_dir,
        args.max_distance_um,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
