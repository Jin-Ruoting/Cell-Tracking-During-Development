#!/usr/bin/env python3
"""Extract prediction-only features for conservative direct division forks."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scripts.analyze_ground_truth_divisions import (
        distribution,
        midpoint_distance,
        physical_distance,
    )
    from scripts.evaluate_edge_predictions import (
        Graph,
        match_nodes,
        read_geff_graph,
        read_single_chunk_zarr,
    )
except ModuleNotFoundError:
    from analyze_ground_truth_divisions import (
        distribution,
        midpoint_distance,
        physical_distance,
    )
    from evaluate_edge_predictions import (
        Graph,
        match_nodes,
        read_geff_graph,
        read_single_chunk_zarr,
    )


def read_edge_probabilities(path: Path) -> dict[tuple[int, int], float]:
    edge_ids = read_single_chunk_zarr(path / "edges" / "ids")
    probabilities = read_single_chunk_zarr(
        path / "edges" / "props" / "edge_prob" / "values"
    )
    if len(edge_ids) != len(probabilities):
        raise ValueError(f"{path}: edge probability length mismatch")
    return {
        (int(source_id), int(target_id)): float(probability)
        for (source_id, target_id), probability in zip(
            edge_ids,
            probabilities,
        )
    }


def extract_candidate_rows(
    dataset: str,
    predicted: Graph,
    ground_truth: Graph,
    edge_probabilities: dict[tuple[int, int], float],
    max_parent_distance_um: float = 16.0,
    max_sister_distance_um: float = 22.0,
    predicted_to_ground: dict[int, int] | None = None,
) -> list[dict[str, object]]:
    if predicted_to_ground is None:
        predicted_to_ground = match_nodes(
            predicted.nodes,
            ground_truth.nodes,
            7.0,
        )

    predicted_successors: dict[int, list[int]] = defaultdict(list)
    predicted_predecessors: dict[int, list[int]] = defaultdict(list)
    for source_id, target_id in predicted.edges:
        predicted_successors[source_id].append(target_id)
        predicted_predecessors[target_id].append(source_id)

    ground_successors: dict[int, list[int]] = defaultdict(list)
    for source_id, target_id in ground_truth.edges:
        ground_successors[source_id].append(target_id)

    roots_by_time: dict[int, list[int]] = defaultdict(list)
    for node_id, node in predicted.nodes.items():
        if not predicted_predecessors.get(node_id):
            roots_by_time[node.t].append(node_id)

    rows: list[dict[str, object]] = []
    for parent_id, child_ids in sorted(predicted_successors.items()):
        if len(child_ids) != 1:
            continue
        parent = predicted.nodes.get(parent_id)
        existing_child_id = child_ids[0]
        existing_child = predicted.nodes.get(existing_child_id)
        if (
            parent is None
            or existing_child is None
            or existing_child.t != parent.t + 1
        ):
            continue

        candidates: list[dict[str, object]] = []
        for candidate_id in roots_by_time.get(parent.t + 1, []):
            candidate = predicted.nodes[candidate_id]
            parent_candidate_um = physical_distance(parent, candidate)
            if parent_candidate_um > max_parent_distance_um:
                continue
            sister_um = physical_distance(existing_child, candidate)
            if sister_um > max_sister_distance_um:
                continue

            predecessor_ids = predicted_predecessors.get(parent_id, [])
            parent_motion_um = (
                physical_distance(
                    predicted.nodes[predecessor_ids[0]],
                    parent,
                )
                if len(predecessor_ids) == 1
                else None
            )
            existing_um = physical_distance(parent, existing_child)
            midpoint_um = midpoint_distance(
                parent,
                existing_child,
                candidate,
            )

            ground_parent = predicted_to_ground.get(parent_id)
            ground_existing = predicted_to_ground.get(existing_child_id)
            ground_candidate = predicted_to_ground.get(candidate_id)
            ground_children = (
                set(ground_successors.get(ground_parent, []))
                if ground_parent is not None
                else set()
            )
            is_true_fork = (
                len(ground_children) == 2
                and {ground_existing, ground_candidate} == ground_children
            )
            candidates.append(
                {
                    "dataset": dataset,
                    "embryo": dataset.split("_", 1)[0],
                    "parent_id": parent_id,
                    "existing_child_id": existing_child_id,
                    "candidate_child_id": candidate_id,
                    "t": parent.t,
                    "candidate_edge_probability": edge_probabilities.get(
                        (parent_id, candidate_id)
                    ),
                    "existing_edge_probability": edge_probabilities.get(
                        (parent_id, existing_child_id)
                    ),
                    "parent_candidate_um": parent_candidate_um,
                    "parent_existing_um": existing_um,
                    "sister_um": sister_um,
                    "midpoint_um": midpoint_um,
                    "child_distance_delta_um": abs(
                        parent_candidate_um - existing_um
                    ),
                    "parent_motion_um": parent_motion_um,
                    "existing_child_has_successor": bool(
                        predicted_successors.get(existing_child_id)
                    ),
                    "candidate_child_has_successor": bool(
                        predicted_successors.get(candidate_id)
                    ),
                    "ground_truth_parent_id": ground_parent,
                    "ground_truth_is_division": len(ground_children) == 2,
                    "is_true_fork": is_true_fork,
                }
            )

        candidates.sort(
            key=lambda row: (
                -float(row["candidate_edge_probability"])
                if row["candidate_edge_probability"] is not None
                else 1.0,
                float(row["midpoint_um"]),
                float(row["parent_candidate_um"]),
                int(row["candidate_child_id"]),
            )
        )
        for rank, row in enumerate(candidates, start=1):
            row["candidate_rank"] = rank
            rows.append(row)
    return rows


def _feature_distributions(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    numeric_features = (
        "candidate_edge_probability",
        "existing_edge_probability",
        "parent_candidate_um",
        "parent_existing_um",
        "sister_um",
        "midpoint_um",
        "child_distance_delta_um",
        "parent_motion_um",
        "candidate_rank",
    )
    return {
        feature: distribution(
            float(row[feature])
            for row in rows
            if row[feature] is not None
        )
        for feature in numeric_features
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    true_rows = [row for row in rows if bool(row["is_true_fork"])]
    false_rows = [row for row in rows if not bool(row["is_true_fork"])]
    by_embryo = Counter(str(row["embryo"]) for row in rows)
    return {
        "candidates": len(rows),
        "candidates_with_preilp_edge": sum(
            row["candidate_edge_probability"] is not None for row in rows
        ),
        "true_candidates": len(true_rows),
        "recoverable_divisions": len(
            {
                (row["dataset"], row["ground_truth_parent_id"])
                for row in true_rows
            }
        ),
        "by_embryo": {
            embryo: {
                "candidates": count,
                "true_candidates": sum(
                    row["embryo"] == embryo
                    and bool(row["is_true_fork"])
                    for row in rows
                ),
            }
            for embryo, count in sorted(by_embryo.items())
        },
        "true_feature_distributions": _feature_distributions(true_rows),
        "false_feature_distributions": _feature_distributions(false_rows),
    }


def analyze_directories(
    baseline_dir: Path,
    preilp_root: Path,
    ground_truth_dir: Path,
    max_parent_distance_um: float,
    max_sister_distance_um: float,
) -> dict[str, object]:
    preilp_paths = {
        path.stem: path for path in sorted(preilp_root.rglob("*.geff"))
    }
    rows: list[dict[str, object]] = []
    for baseline_path in sorted(baseline_dir.glob("*.geff")):
        dataset = baseline_path.stem
        preilp_path = preilp_paths.get(dataset)
        ground_truth_path = ground_truth_dir / baseline_path.name
        if preilp_path is None or not ground_truth_path.is_dir():
            raise FileNotFoundError(f"incomplete inputs for {dataset}")
        rows.extend(
            extract_candidate_rows(
                dataset,
                read_geff_graph(baseline_path),
                read_geff_graph(ground_truth_path),
                read_edge_probabilities(preilp_path),
                max_parent_distance_um,
                max_sister_distance_um,
            )
        )
    return {
        "baseline_dir": str(baseline_dir.resolve()),
        "preilp_root": str(preilp_root.resolve()),
        "ground_truth_dir": str(ground_truth_dir.resolve()),
        "max_parent_distance_um": max_parent_distance_um,
        "max_sister_distance_um": max_sister_distance_um,
        "summary": summarize(rows),
        "candidates": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("preilp_root", type=Path)
    parser.add_argument("ground_truth_dir", type=Path)
    parser.add_argument("--max-parent-distance-um", type=float, default=16.0)
    parser.add_argument("--max-sister-distance-um", type=float, default=22.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_directories(
        args.baseline_dir,
        args.preilp_root,
        args.ground_truth_dir,
        args.max_parent_distance_um,
        args.max_sister_distance_um,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
