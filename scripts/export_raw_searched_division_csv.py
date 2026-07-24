#!/usr/bin/env python3
"""Export raw ILP graphs plus the E005 searched forks as a submission CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from scripts.analyze_division_candidates import read_edge_probabilities
    from scripts.evaluate_edge_predictions import read_geff_graph
    from scripts.searched_division_rule import (
        SEARCHED_DIVISION_RULE_NAME,
        apply_searched_division_edges,
        select_searched_division_edges,
    )
except ModuleNotFoundError:
    from analyze_division_candidates import read_edge_probabilities
    from evaluate_edge_predictions import read_geff_graph
    from searched_division_rule import (
        SEARCHED_DIVISION_RULE_NAME,
        apply_searched_division_edges,
        select_searched_division_edges,
    )


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


def export_submission(
    baseline_dir: Path,
    preilp_root: Path,
    output_path: Path,
) -> dict[str, object]:
    preilp_paths = {
        path.stem: path for path in sorted(preilp_root.rglob("*.geff"))
    }
    baseline_paths = sorted(baseline_dir.glob("*.geff"))
    if not baseline_paths:
        raise FileNotFoundError(f"no baseline GEFFs under {baseline_dir}")
    missing = [
        path.stem for path in baseline_paths
        if path.stem not in preilp_paths
    ]
    if missing:
        raise FileNotFoundError(f"missing pre-ILP graphs: {missing[:10]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_id = 0
    totals = {
        "datasets": 0,
        "nodes": 0,
        "baseline_edges": 0,
        "selected_forks": 0,
        "added_forks": 0,
    }
    per_dataset: dict[str, dict[str, int]] = {}
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for baseline_path in baseline_paths:
            dataset = baseline_path.stem
            graph = read_geff_graph(baseline_path)
            nodes_by_id = {
                node_id: {
                    "node_id": node_id,
                    "t": node.t,
                    "z": node.z,
                    "y": node.y,
                    "x": node.x,
                }
                for node_id, node in graph.nodes.items()
            }
            raw_edges = [
                {
                    "source_id": source_id,
                    "target_id": target_id,
                }
                for source_id, target_id in graph.edges
            ]
            selected = select_searched_division_edges(
                nodes_by_id,
                raw_edges,
                read_edge_probabilities(preilp_paths[dataset]),
            )
            edges, apply_stats = apply_searched_division_edges(
                nodes_by_id,
                raw_edges,
                selected,
            )
            if apply_stats["searched_divisions_added"] != len(selected):
                raise AssertionError(
                    f"{dataset}: raw topology rejected selected fork"
                )

            for node_id in sorted(nodes_by_id):
                node = nodes_by_id[node_id]
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": dataset,
                        "row_type": "node",
                        "node_id": node_id,
                        "t": int(node["t"]),
                        "z": max(0, int(round(float(node["z"])))),
                        "y": max(0, int(round(float(node["y"])))),
                        "x": max(0, int(round(float(node["x"])))),
                        "source_id": -1,
                        "target_id": -1,
                    }
                )
                row_id += 1
            for edge in edges:
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": dataset,
                        "row_type": "edge",
                        "node_id": -1,
                        "t": -1,
                        "z": -1,
                        "y": -1,
                        "x": -1,
                        "source_id": int(edge["source_id"]),
                        "target_id": int(edge["target_id"]),
                    }
                )
                row_id += 1

            per_dataset[dataset] = {
                "nodes": len(nodes_by_id),
                "baseline_edges": len(raw_edges),
                "selected_forks": len(selected),
                "added_forks": apply_stats["searched_divisions_added"],
            }
            totals["datasets"] += 1
            totals["nodes"] += len(nodes_by_id)
            totals["baseline_edges"] += len(raw_edges)
            totals["selected_forks"] += len(selected)
            totals["added_forks"] += apply_stats["searched_divisions_added"]

    return {
        "rule": SEARCHED_DIVISION_RULE_NAME,
        "baseline_dir": str(baseline_dir.resolve()),
        "preilp_root": str(preilp_root.resolve()),
        "output_path": str(output_path.resolve()),
        "rows": row_id,
        "totals": totals,
        "per_dataset": per_dataset,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("preilp_root", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = export_submission(
        args.baseline_dir,
        args.preilp_root,
        args.output_path,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
