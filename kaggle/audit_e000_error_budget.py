#!/usr/bin/env python3
"""Attribute E000 edge errors to detection, association, and candidate recall.

The audit uses the pinned official node matcher. It never edits a graph or
creates a submission. For each ground-truth edge it asks whether both endpoint
nodes were detected by the final E000 graph and, if so, whether the edge was
present in the final, raw-ILP, or pre-ILP candidate graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any


VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--raw-ilp-dir", type=Path, required=True)
    parser.add_argument("--preilp-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-distance-um", type=float, default=7.0)
    parser.add_argument("--max-datasets", type=int, default=0)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_imports(runtime_dir: Path, scorer_dir: Path) -> None:
    for path in (runtime_dir, scorer_dir / "src"):
        if not path.is_dir():
            raise NotADirectoryError(path)
        sys.path.insert(0, str(path.resolve()))


def geff_paths(directory: Path, recursive: bool) -> dict[str, Path]:
    iterator = directory.rglob("*.geff") if recursive else directory.glob("*.geff")
    paths: dict[str, Path] = {}
    for path in sorted(iterator):
        if path.stem in paths:
            raise RuntimeError(
                f"Duplicate dataset {path.stem}: {paths[path.stem]} and {path}"
            )
        paths[path.stem] = path
    if not paths:
        raise FileNotFoundError(f"No GEFF graphs under {directory}")
    return paths


def load_graph(path: Path, td: Any) -> Any:
    result = td.graph.IndexedRXGraph.from_geff(path)
    return result[0] if isinstance(result, tuple) else result


def rows(frame: Any) -> list[dict[str, Any]]:
    return list(frame.iter_rows(named=True))


def matched_view(graph: Any, td: Any) -> dict[str, Any]:
    keys = td.DEFAULT_ATTR_KEYS
    node_rows = rows(
        graph.node_attrs(
            attr_keys=[keys.NODE_ID, keys.MATCHED_NODE_ID],
        )
    )
    node_map: dict[int, int] = {}
    matched_gt_nodes: set[int] = set()
    for row in node_rows:
        matched = row.get(keys.MATCHED_NODE_ID)
        if matched is None or int(matched) < 0:
            continue
        node_id = int(row[keys.NODE_ID])
        matched_id = int(matched)
        node_map[node_id] = matched_id
        matched_gt_nodes.add(matched_id)

    edge_rows = rows(
        graph.edge_attrs(
            attr_keys=[keys.EDGE_SOURCE, keys.EDGE_TARGET],
        )
    )
    mapped_edges: set[tuple[int, int]] = set()
    for row in edge_rows:
        source = node_map.get(int(row[keys.EDGE_SOURCE]))
        target = node_map.get(int(row[keys.EDGE_TARGET]))
        if source is not None and target is not None:
            mapped_edges.add((source, target))
    return {
        "matched_gt_nodes": matched_gt_nodes,
        "mapped_edges": mapped_edges,
        "predicted_nodes": graph.num_nodes(),
        "predicted_edges": graph.num_edges(),
    }


def match_graph(
    graph: Any,
    gt_graph: Any,
    scale: tuple[float, ...],
    max_distance: float,
) -> None:
    from tracksdata.metrics import DistanceMatching
    from tracksdata.options import get_options, set_options

    previous = get_options().show_progress
    set_options(show_progress=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph.match(
                gt_graph,
                matching=DistanceMatching(
                    max_distance=max_distance,
                    scale=scale,
                ),
            )
    finally:
        set_options(show_progress=previous)


def ground_truth_edges(gt_graph: Any, td: Any) -> set[tuple[int, int]]:
    keys = td.DEFAULT_ATTR_KEYS
    edge_rows = rows(
        gt_graph.edge_attrs(
            attr_keys=[keys.EDGE_SOURCE, keys.EDGE_TARGET],
        )
    )
    return {
        (int(row[keys.EDGE_SOURCE]), int(row[keys.EDGE_TARGET]))
        for row in edge_rows
    }


def division_events(
    gt_edges: set[tuple[int, int]],
) -> dict[int, set[int]]:
    children: dict[int, set[int]] = {}
    for source, target in gt_edges:
        children.setdefault(source, set()).add(target)
    return {
        source: targets
        for source, targets in children.items()
        if len(targets) == 2
    }


def dataset_result(
    name: str,
    prediction_path: Path,
    raw_path: Path,
    preilp_path: Path,
    gt_path: Path,
    gt_root: Path,
    max_distance: float,
    td: Any,
) -> dict[str, Any]:
    from geff import GeffMetadata
    from tracking_cellmot.io import open_dataset
    from tracking_cellmot.metrics import evaluate

    gt_graph = load_graph(gt_path, td)
    prediction = load_graph(prediction_path, td)
    raw_graph = load_graph(raw_path, td)
    preilp_graph = load_graph(preilp_path, td)

    try:
        scale = tuple(
            float(value)
            for value in open_dataset(
                gt_root / name,
                load_image=False,
            ).scale
        )
    except FileNotFoundError:
        scale = VOXEL_SCALE_UM

    official = evaluate(
        prediction,
        gt_graph,
        scale=scale,
        max_distance=max_distance,
    )
    final_view = matched_view(prediction, td)
    match_graph(raw_graph, gt_graph, scale, max_distance)
    raw_view = matched_view(raw_graph, td)
    match_graph(preilp_graph, gt_graph, scale, max_distance)
    preilp_view = matched_view(preilp_graph, td)

    gt_edges = ground_truth_edges(gt_graph, td)
    gt_nodes_final = final_view["matched_gt_nodes"]
    final_edges = final_view["mapped_edges"]
    raw_edges = raw_view["mapped_edges"]
    preilp_edges = preilp_view["mapped_edges"]

    counts: Counter[str] = Counter()
    for source, target in gt_edges:
        source_found = source in gt_nodes_final
        target_found = target in gt_nodes_final
        if source_found and target_found:
            counts["endpoint_available"] += 1
        elif source_found:
            counts["target_missing"] += 1
        elif target_found:
            counts["source_missing"] += 1
        else:
            counts["both_missing"] += 1

        edge = (source, target)
        if edge in final_edges:
            counts["final_correct"] += 1
        if edge in raw_edges:
            counts["raw_correct"] += 1
        if edge in preilp_edges:
            counts["preilp_correct"] += 1
        if edge in final_edges or edge in preilp_edges:
            counts["final_preilp_union_correct"] += 1

        if source_found and target_found and edge not in final_edges:
            counts["association_limited"] += 1
            if edge in raw_edges:
                counts["association_raw_available"] += 1
            if edge in preilp_edges:
                counts["association_preilp_available"] += 1
            else:
                counts["association_candidate_missing"] += 1

    if counts["final_correct"] != official.edge_tp:
        raise AssertionError(
            f"{name}: mapped final TP {counts['final_correct']} "
            f"!= official TP {official.edge_tp}"
        )

    divisions = division_events(gt_edges)
    division_endpoint_available = sum(
        source in gt_nodes_final
        and all(target in gt_nodes_final for target in targets)
        for source, targets in divisions.items()
    )

    metadata = GeffMetadata.read(gt_path)
    estimated_nodes = float(
        (metadata.extra or {}).get("estimated_number_of_nodes", math.nan)
    )
    predicted_nodes = int(final_view["predicted_nodes"])
    node_ratio = (
        (predicted_nodes - estimated_nodes) / estimated_nodes
        if estimated_nodes > 0
        else math.nan
    )
    adjustment = (
        max(0.0, 1.0 - 0.1 * node_ratio)
        if math.isfinite(node_ratio)
        else math.nan
    )

    gt_edge_count = len(gt_edges)
    endpoint_oracle = counts["endpoint_available"] / max(gt_edge_count, 1)
    union_oracle = counts["final_preilp_union_correct"] / max(
        gt_edge_count,
        1,
    )
    return {
        "dataset": name,
        "embryo": name.split("_", 1)[0],
        "scale": scale,
        "gt_nodes": gt_graph.num_nodes(),
        "gt_edges": gt_edge_count,
        "gt_divisions": len(divisions),
        "predicted_nodes": predicted_nodes,
        "predicted_edges": int(final_view["predicted_edges"]),
        "estimated_total_nodes": estimated_nodes,
        "node_count_ratio": node_ratio,
        "node_adjustment_factor": adjustment,
        "matched_gt_nodes": len(gt_nodes_final),
        "official": {
            "edge_tp": official.edge_tp,
            "edge_fp": official.edge_fp,
            "edge_fn": official.edge_fn,
            "division_tp": official.division_tp,
            "division_fp": official.division_fp,
            "division_fn": official.division_fn,
        },
        "edge_budget": dict(sorted(counts.items())),
        "division_endpoint_available": division_endpoint_available,
        "endpoint_oracle_edge_jaccard": endpoint_oracle,
        "endpoint_oracle_adjusted_edge_jaccard": (
            endpoint_oracle * adjustment
            if math.isfinite(adjustment)
            else math.nan
        ),
        "final_preilp_union_oracle_edge_jaccard": union_oracle,
        "final_preilp_union_oracle_adjusted_edge_jaccard": (
            union_oracle * adjustment
            if math.isfinite(adjustment)
            else math.nan
        ),
    }


def aggregate(rows_: list[dict[str, Any]]) -> dict[str, Any]:
    official_keys = (
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "division_tp",
        "division_fp",
        "division_fn",
    )
    official = {
        key: sum(int(row["official"][key]) for row in rows_)
        for key in official_keys
    }
    budget_keys = sorted(
        {
            key
            for row in rows_
            for key in row["edge_budget"]
        }
    )
    budget = {
        key: sum(int(row["edge_budget"].get(key, 0)) for row in rows_)
        for key in budget_keys
    }
    gt_edges = sum(int(row["gt_edges"]) for row in rows_)
    weights = [
        int(row["official"]["edge_tp"])
        + int(row["official"]["edge_fp"])
        + int(row["official"]["edge_fn"])
        for row in rows_
    ]
    total_weight = sum(weights)
    current_adjusted = (
        sum(
            weight
            * (
                int(row["official"]["edge_tp"])
                / max(weight, 1)
                * float(row["node_adjustment_factor"])
            )
            for row, weight in zip(rows_, weights)
            if math.isfinite(float(row["node_adjustment_factor"]))
        )
        / total_weight
        if total_weight
        else math.nan
    )

    def oracle_adjusted(key: str) -> float:
        oracle_weights = [int(row["gt_edges"]) for row in rows_]
        denominator = sum(oracle_weights)
        if not denominator:
            return math.nan
        return sum(
            weight * float(row[key])
            for row, weight in zip(rows_, oracle_weights)
            if math.isfinite(float(row[key]))
        ) / denominator

    edge_denominator = (
        official["edge_tp"] + official["edge_fp"] + official["edge_fn"]
    )
    division_denominator = (
        official["division_tp"]
        + official["division_fp"]
        + official["division_fn"]
    )
    return {
        "datasets": len(rows_),
        "gt_nodes": sum(int(row["gt_nodes"]) for row in rows_),
        "gt_edges": gt_edges,
        "gt_divisions": sum(int(row["gt_divisions"]) for row in rows_),
        "matched_gt_nodes": sum(int(row["matched_gt_nodes"]) for row in rows_),
        "official": official,
        "current_edge_jaccard": (
            official["edge_tp"] / edge_denominator
            if edge_denominator
            else math.nan
        ),
        "current_adjusted_edge_jaccard": current_adjusted,
        "current_division_jaccard": (
            official["division_tp"] / division_denominator
            if division_denominator
            else math.nan
        ),
        "edge_budget": budget,
        "endpoint_oracle_edge_jaccard": (
            budget.get("endpoint_available", 0) / gt_edges
            if gt_edges
            else math.nan
        ),
        "endpoint_oracle_adjusted_edge_jaccard": oracle_adjusted(
            "endpoint_oracle_adjusted_edge_jaccard"
        ),
        "final_preilp_union_oracle_edge_jaccard": (
            budget.get("final_preilp_union_correct", 0) / gt_edges
            if gt_edges
            else math.nan
        ),
        "final_preilp_union_oracle_adjusted_edge_jaccard": oracle_adjusted(
            "final_preilp_union_oracle_adjusted_edge_jaccard"
        ),
        "division_endpoint_available": sum(
            int(row["division_endpoint_available"]) for row in rows_
        ),
    }


def main() -> None:
    args = parse_args()
    if args.max_distance_um <= 0:
        raise ValueError("--max-distance-um must be positive")
    if args.max_datasets < 0:
        raise ValueError("--max-datasets must be non-negative")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    prepare_imports(args.runtime_dir, args.scorer_dir)
    import tracksdata as td

    predictions = geff_paths(args.prediction_dir, recursive=False)
    raw_graphs = geff_paths(args.raw_ilp_dir, recursive=False)
    preilp_graphs = geff_paths(args.preilp_dir, recursive=True)
    gt_graphs = geff_paths(args.ground_truth_dir, recursive=False)
    names = sorted(predictions)
    if args.max_datasets:
        names = names[: args.max_datasets]
    for label, paths in (
        ("raw ILP", raw_graphs),
        ("pre-ILP", preilp_graphs),
        ("ground truth", gt_graphs),
    ):
        missing = sorted(set(names) - set(paths))
        if missing:
            raise RuntimeError(f"{label} is missing datasets: {missing}")

    results = []
    for index, name in enumerate(names, start=1):
        result = dataset_result(
            name=name,
            prediction_path=predictions[name],
            raw_path=raw_graphs[name],
            preilp_path=preilp_graphs[name],
            gt_path=gt_graphs[name],
            gt_root=args.ground_truth_dir,
            max_distance=args.max_distance_um,
            td=td,
        )
        results.append(result)
        budget = result["edge_budget"]
        print(
            f"[{index:02d}/{len(names):02d}] {name}: "
            f"TP={result['official']['edge_tp']} "
            f"FN={result['official']['edge_fn']} "
            f"endpoint={budget.get('endpoint_available', 0)} "
            f"assoc-miss={budget.get('association_limited', 0)}",
            flush=True,
        )

    aggregates = {
        "all": aggregate(results),
        **{
            embryo: aggregate(
                [row for row in results if row["embryo"] == embryo]
            )
            for embryo in sorted({row["embryo"] for row in results})
        },
    }
    report = {
        "hypothesis": (
            "E000 improvement should target the dominant recoverable error "
            "class rather than another postprocessing threshold sweep."
        ),
        "config": {
            "prediction_dir": str(args.prediction_dir.resolve()),
            "raw_ilp_dir": str(args.raw_ilp_dir.resolve()),
            "preilp_dir": str(args.preilp_dir.resolve()),
            "ground_truth_dir": str(args.ground_truth_dir.resolve()),
            "scorer_dir": str(args.scorer_dir.resolve()),
            "scorer_metrics_sha256": file_sha256(
                args.scorer_dir / "src" / "tracking_cellmot" / "metrics.py"
            ),
            "max_distance_um": args.max_distance_um,
        },
        "aggregates": aggregates,
        "datasets": results,
        "notes": [
            "Endpoint availability uses the official 7 um node matching.",
            "Endpoint-oracle scores remove all false edges and therefore are "
            "diagnostic ceilings, not deployable candidate scores.",
            "Pre-ILP availability means the learned candidate graph contained "
            "a correctly mapped edge; it does not prove a safe selection rule.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}", flush=True)
    print(json.dumps(aggregates["all"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
