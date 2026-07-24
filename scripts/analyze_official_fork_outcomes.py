#!/usr/bin/env python3
"""Join added fork features to the patched scorer's TP/FP fork sets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from scripts.analyze_division_candidates import (
        extract_candidate_rows,
        read_edge_probabilities,
    )
    from scripts.evaluate_edge_predictions import read_geff_graph
except ModuleNotFoundError:
    from analyze_division_candidates import (
        extract_candidate_rows,
        read_edge_probabilities,
    )
    from evaluate_edge_predictions import read_geff_graph


VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


def parse_candidate_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("candidate must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise ValueError("candidate must use non-empty NAME=PATH")
    return name, Path(raw_path)


def load_tracksdata_graph(path: Path):
    import tracksdata as td

    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def summarize_outcomes(rows: list[dict[str, object]]) -> dict[str, object]:
    counts = Counter(str(row["outcome"]) for row in rows)
    return {
        "added_edges": len(rows),
        "official_tp_forks": counts["tp"],
        "official_fp_forks": counts["fp"],
        "official_ignored_forks": counts["ignored"],
        "by_embryo": {
            embryo: {
                "added_edges": sum(row["embryo"] == embryo for row in rows),
                "official_tp_forks": sum(
                    row["embryo"] == embryo and row["outcome"] == "tp"
                    for row in rows
                ),
                "official_fp_forks": sum(
                    row["embryo"] == embryo and row["outcome"] == "fp"
                    for row in rows
                ),
            }
            for embryo in sorted({str(row["embryo"]) for row in rows})
        },
    }


def analyze(
    baseline_dir: Path,
    preilp_root: Path,
    ground_truth_dir: Path,
    candidate_dirs: dict[str, Path],
) -> dict[str, object]:
    from tracking_cellmot.division_metrics import score_divisions

    preilp_paths = {
        path.stem: path for path in sorted(preilp_root.rglob("*.geff"))
    }
    rows_by_rule: dict[str, list[dict[str, object]]] = {
        name: [] for name in candidate_dirs
    }

    for baseline_path in sorted(baseline_dir.glob("*.geff")):
        dataset = baseline_path.stem
        ground_truth_path = ground_truth_dir / baseline_path.name
        preilp_path = preilp_paths.get(dataset)
        if preilp_path is None or not ground_truth_path.is_dir():
            raise FileNotFoundError(f"incomplete inputs for {dataset}")

        baseline_simple = read_geff_graph(baseline_path)
        ground_truth_simple = read_geff_graph(ground_truth_path)
        feature_rows = extract_candidate_rows(
            dataset,
            baseline_simple,
            ground_truth_simple,
            read_edge_probabilities(preilp_path),
        )
        features = {
            (int(row["parent_id"]), int(row["candidate_child_id"])): row
            for row in feature_rows
        }
        baseline_edges = set(baseline_simple.edges)
        ground_truth_graph = load_tracksdata_graph(ground_truth_path)

        for name, candidate_dir in candidate_dirs.items():
            candidate_path = candidate_dir / baseline_path.name
            if not candidate_path.is_dir():
                raise FileNotFoundError(candidate_path)
            candidate_graph = load_tracksdata_graph(candidate_path)
            added_edges = sorted(
                set(map(tuple, candidate_graph.edge_list)) - baseline_edges
            )
            division_scores = score_divisions(
                candidate_graph,
                ground_truth_graph,
                VOXEL_SCALE_UM,
                7.0,
            )
            for source_id, target_id in added_edges:
                feature = features.get((source_id, target_id))
                if feature is None:
                    raise ValueError(
                        f"{name}/{dataset}: missing features for added edge"
                    )
                outcome = (
                    "tp"
                    if source_id in division_scores.tp_forks
                    else "fp"
                    if source_id in division_scores.fp_forks
                    else "ignored"
                )
                rows_by_rule[name].append(
                    {
                        **feature,
                        "rule": name,
                        "outcome": outcome,
                    }
                )
        print(dataset, flush=True)

    return {
        "baseline_dir": str(baseline_dir.resolve()),
        "preilp_root": str(preilp_root.resolve()),
        "ground_truth_dir": str(ground_truth_dir.resolve()),
        "candidates": {
            name: str(path.resolve())
            for name, path in candidate_dirs.items()
        },
        "summary": {
            name: summarize_outcomes(rows)
            for name, rows in rows_by_rule.items()
        },
        "outcomes": rows_by_rule,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("preilp_root", type=Path)
    parser.add_argument("ground_truth_dir", type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate directory as NAME=PATH.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_dirs = dict(parse_candidate_spec(value) for value in args.candidate)
    if len(candidate_dirs) != len(args.candidate):
        raise ValueError("candidate names must be unique")
    report = analyze(
        args.baseline_dir,
        args.preilp_root,
        args.ground_truth_dir,
        candidate_dirs,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
