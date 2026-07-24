#!/usr/bin/env python3
"""Add prediction-only direct forks under explicit conservative rules."""

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
    from scripts.evaluate_edge_predictions import Graph, read_geff_graph
except ModuleNotFoundError:
    from analyze_division_candidates import (
        extract_candidate_rows,
        read_edge_probabilities,
    )
    from evaluate_edge_predictions import Graph, read_geff_graph


NUMERIC_RULE_FEATURES = {
    "candidate_edge_probability",
    "candidate_rank",
    "child_distance_delta_um",
    "existing_edge_probability",
    "midpoint_um",
    "parent_candidate_um",
    "parent_existing_um",
    "parent_motion_um",
    "sister_um",
}
BOOLEAN_RULE_FEATURES = {
    "candidate_child_has_successor",
    "existing_child_has_successor",
}


def validate_rule(rule: dict[str, object]) -> None:
    if not isinstance(rule.get("name"), str) or not rule["name"]:
        raise ValueError("every rule needs a non-empty name")
    allowed = {"name", "require_preilp_edge"}
    allowed.update(
        f"{prefix}_{feature}"
        for prefix in ("min", "max")
        for feature in NUMERIC_RULE_FEATURES
    )
    allowed.update(f"require_{feature}" for feature in BOOLEAN_RULE_FEATURES)
    unknown = set(rule) - allowed
    if unknown:
        raise ValueError(f"unknown rule keys: {sorted(unknown)}")


def matches_rule(row: dict[str, object], rule: dict[str, object]) -> bool:
    if rule.get("require_preilp_edge") and (
        row["candidate_edge_probability"] is None
    ):
        return False
    for feature in NUMERIC_RULE_FEATURES:
        value = row.get(feature)
        lower = rule.get(f"min_{feature}")
        upper = rule.get(f"max_{feature}")
        if value is None:
            if lower is not None:
                return False
            continue
        numeric = float(value)
        if lower is not None and numeric < float(lower):
            return False
        if upper is not None and numeric > float(upper):
            return False
    for feature in BOOLEAN_RULE_FEATURES:
        required = rule.get(f"require_{feature}")
        if required is not None and bool(row.get(feature)) != bool(required):
            return False
    return True


def select_rows(
    rows: list[dict[str, object]],
    rule: dict[str, object],
) -> list[dict[str, object]]:
    qualified = [row for row in rows if matches_rule(row, rule)]
    qualified.sort(
        key=lambda row: (
            int(row["candidate_rank"]),
            -float(row["candidate_edge_probability"])
            if row["candidate_edge_probability"] is not None
            else 1.0,
            float(row["midpoint_um"]),
            float(row["parent_candidate_um"]),
            int(row["parent_id"]),
            int(row["candidate_child_id"]),
        )
    )
    selected: list[dict[str, object]] = []
    used_parents: set[int] = set()
    used_candidates: set[int] = set()
    for row in qualified:
        parent_id = int(row["parent_id"])
        candidate_id = int(row["candidate_child_id"])
        if parent_id in used_parents or candidate_id in used_candidates:
            continue
        selected.append(row)
        used_parents.add(parent_id)
        used_candidates.add(candidate_id)
    return sorted(selected, key=lambda row: int(row["parent_id"]))


def prediction_only_rows(
    dataset: str,
    baseline_path: Path,
    preilp_path: Path,
) -> list[dict[str, object]]:
    return extract_candidate_rows(
        dataset,
        read_geff_graph(baseline_path),
        Graph(nodes={}, edges=[]),
        read_edge_probabilities(preilp_path),
        predicted_to_ground={},
    )


def augment_graph(
    baseline_path: Path,
    output_path: Path,
    selected_rows: list[dict[str, object]],
) -> None:
    import tracksdata as td

    from biohub_tracking.io import save_graph

    loaded = td.graph.IndexedRXGraph.from_geff(baseline_path)
    graph = loaded[0] if isinstance(loaded, tuple) else loaded
    for row in selected_rows:
        parent_id = int(row["parent_id"])
        candidate_id = int(row["candidate_child_id"])
        if graph.out_degree(parent_id) != 1:
            raise ValueError(f"{baseline_path.stem}: parent degree changed")
        if graph.in_degree(candidate_id) != 0:
            raise ValueError(f"{baseline_path.stem}: candidate is not a root")
        graph.add_edge(
            parent_id,
            candidate_id,
            {
                "edge_prob": float(
                    row["candidate_edge_probability"]
                    if row["candidate_edge_probability"] is not None
                    else 0.0
                ),
                "edge_dist": float(row["parent_candidate_um"]),
                "solution": True,
            },
        )
    save_graph(graph, output_path)


def run_rules(
    baseline_dir: Path,
    preilp_root: Path,
    output_root: Path,
    rules: list[dict[str, object]],
) -> dict[str, object]:
    for rule in rules:
        validate_rule(rule)
    names = [str(rule["name"]) for rule in rules]
    if len(names) != len(set(names)):
        raise ValueError("rule names must be unique")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)

    preilp_paths = {
        path.stem: path for path in sorted(preilp_root.rglob("*.geff"))
    }
    baseline_paths = sorted(baseline_dir.glob("*.geff"))
    if not baseline_paths:
        raise FileNotFoundError(f"no baseline GEFFs under {baseline_dir}")

    counts = {name: Counter() for name in names}
    per_dataset: dict[str, dict[str, int]] = {}
    for index, baseline_path in enumerate(baseline_paths, start=1):
        dataset = baseline_path.stem
        preilp_path = preilp_paths.get(dataset)
        if preilp_path is None:
            raise FileNotFoundError(f"missing pre-ILP graph for {dataset}")
        rows = prediction_only_rows(dataset, baseline_path, preilp_path)
        per_dataset[dataset] = {}
        for rule in rules:
            name = str(rule["name"])
            selected = select_rows(rows, rule)
            per_dataset[dataset][name] = len(selected)
            counts[name]["datasets"] += 1
            counts[name]["added_edges"] += len(selected)
            counts[name]["datasets_with_additions"] += int(bool(selected))
            augment_graph(
                baseline_path,
                output_root / name / baseline_path.name,
                selected,
            )
        print(
            f"[{index:03d}/{len(baseline_paths):03d}] {dataset}",
            flush=True,
        )

    return {
        "baseline_dir": str(baseline_dir.resolve()),
        "preilp_root": str(preilp_root.resolve()),
        "output_root": str(output_root.resolve()),
        "rules": rules,
        "summary": {
            name: dict(counts[name])
            for name in names
        },
        "per_dataset": per_dataset,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("preilp_root", type=Path)
    parser.add_argument("rules_json", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rules = json.loads(args.rules_json.read_text(encoding="utf-8"))
    if not isinstance(rules, list):
        raise ValueError("rules JSON must be a list")
    report = run_rules(
        args.baseline_dir,
        args.preilp_root,
        args.output_root,
        rules,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
