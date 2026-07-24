#!/usr/bin/env python3
"""Beam-search conjunctions over official fork TP/FP feature outcomes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


FEATURES = (
    "candidate_edge_probability",
    "existing_edge_probability",
    "parent_candidate_um",
    "parent_existing_um",
    "sister_um",
    "midpoint_um",
    "child_distance_delta_um",
    "parent_motion_um",
)


@dataclass(frozen=True)
class Condition:
    operation: str
    feature: str
    threshold: float


def condition_matches(
    row: dict[str, object],
    condition: Condition,
) -> bool:
    value = row.get(condition.feature)
    if value is None:
        return condition.operation == "max"
    numeric = float(value)
    if condition.operation == "min":
        return numeric >= condition.threshold
    if condition.operation == "max":
        return numeric <= condition.threshold
    raise ValueError(f"unknown operation {condition.operation}")


def rule_matches(
    row: dict[str, object],
    conditions: tuple[Condition, ...],
) -> bool:
    return all(condition_matches(row, condition) for condition in conditions)


def canonical_conditions(
    conditions: tuple[Condition, ...],
) -> tuple[Condition, ...] | None:
    bounds: dict[tuple[str, str], float] = {}
    for condition in conditions:
        key = (condition.operation, condition.feature)
        if condition.operation == "min":
            bounds[key] = max(bounds.get(key, condition.threshold), condition.threshold)
        elif condition.operation == "max":
            bounds[key] = min(bounds.get(key, condition.threshold), condition.threshold)
        else:
            raise ValueError(f"unknown operation {condition.operation}")
    for feature in FEATURES:
        lower = bounds.get(("min", feature))
        upper = bounds.get(("max", feature))
        if lower is not None and upper is not None and lower > upper:
            return None
    return tuple(
        Condition(operation, feature, threshold)
        for (operation, feature), threshold in sorted(bounds.items())
    )


def rule_statistics(
    rows: list[dict[str, object]],
    conditions: tuple[Condition, ...],
    division_totals: dict[str, int],
) -> dict[str, object]:
    selected = [row for row in rows if rule_matches(row, conditions)]
    labeled = [row for row in selected if row["outcome"] != "ignored"]
    true_positives = sum(row["outcome"] == "tp" for row in labeled)
    false_positives = sum(row["outcome"] == "fp" for row in labeled)
    total_divisions = sum(division_totals.values())
    overall = true_positives / (false_positives + total_divisions)
    by_embryo = {}
    for embryo, divisions in sorted(division_totals.items()):
        embryo_rows = [row for row in labeled if row["embryo"] == embryo]
        tp = sum(row["outcome"] == "tp" for row in embryo_rows)
        fp = sum(row["outcome"] == "fp" for row in embryo_rows)
        by_embryo[embryo] = {
            "tp": tp,
            "fp": fp,
            "division_jaccard": tp / (fp + divisions),
        }
    return {
        "conditions": [
            {
                "operation": condition.operation,
                "feature": condition.feature,
                "threshold": condition.threshold,
            }
            for condition in conditions
        ],
        "added_edges": len(selected),
        "tp": true_positives,
        "fp": false_positives,
        "division_jaccard": overall,
        "by_embryo": by_embryo,
    }


def search_rules(
    rows: list[dict[str, object]],
    division_totals: dict[str, int],
    max_depth: int,
    beam_width: int,
    result_count: int,
) -> dict[str, object]:
    labeled = [row for row in rows if row["outcome"] != "ignored"]
    atoms = sorted(
        {
            Condition(operation, feature, float(row[feature]))
            for row in labeled
            if row["outcome"] == "tp"
            for feature in FEATURES
            if row.get(feature) is not None
            for operation in ("min", "max")
        },
        key=lambda condition: (
            condition.feature,
            condition.operation,
            condition.threshold,
        ),
    )

    cache: dict[tuple[Condition, ...], dict[str, object]] = {}

    def statistics(conditions: tuple[Condition, ...]) -> dict[str, object]:
        if conditions not in cache:
            cache[conditions] = rule_statistics(
                rows,
                conditions,
                division_totals,
            )
        return cache[conditions]

    def robust_key(item: tuple[Condition, ...]):
        stats = statistics(item)
        embryo_scores = [
            value["division_jaccard"]
            for value in stats["by_embryo"].values()
        ]
        return (
            min(embryo_scores),
            stats["division_jaccard"],
            stats["tp"],
            -stats["fp"],
            -stats["added_edges"],
            -len(item),
        )

    def overall_key(item: tuple[Condition, ...]):
        stats = statistics(item)
        embryo_tp = [
            value["tp"] for value in stats["by_embryo"].values()
        ]
        return (
            stats["division_jaccard"],
            min(embryo_tp),
            stats["tp"],
            -stats["fp"],
            -stats["added_edges"],
            -len(item),
        )

    beam: list[tuple[Condition, ...]] = [tuple()]
    explored: set[tuple[Condition, ...]] = {tuple()}
    for _ in range(max_depth):
        candidates: set[tuple[Condition, ...]] = set()
        for conditions in beam:
            for atom in atoms:
                combined = canonical_conditions((*conditions, atom))
                if combined is None or combined in explored:
                    continue
                stats = statistics(combined)
                if all(
                    value["tp"] >= 1
                    for value in stats["by_embryo"].values()
                ):
                    candidates.add(combined)
        explored.update(candidates)
        beam = sorted(
            candidates,
            key=robust_key,
            reverse=True,
        )[:beam_width]
        if not beam:
            break

    eligible = [
        conditions
        for conditions in explored
        if conditions
        and all(
            value["tp"] >= 1
            for value in statistics(conditions)["by_embryo"].values()
        )
    ]
    top_overall = sorted(
        eligible,
        key=overall_key,
        reverse=True,
    )[:result_count]
    top_robust = sorted(
        eligible,
        key=robust_key,
        reverse=True,
    )[:result_count]
    return {
        "input_rows": len(rows),
        "labeled_rows": len(labeled),
        "atoms": len(atoms),
        "explored_rules": len(explored),
        "division_totals": division_totals,
        "top_overall": [statistics(rule) for rule in top_overall],
        "top_robust": [statistics(rule) for rule in top_robust],
    }


def parse_assignment(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise ValueError("division total must use EMBRYO=COUNT")
    embryo, raw_count = value.split("=", 1)
    return embryo, int(raw_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("outcome_report", type=Path)
    parser.add_argument("--rule", default="preedge_broad")
    parser.add_argument(
        "--division-total",
        action="append",
        required=True,
        help="Annotated division denominator as EMBRYO=COUNT.",
    )
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=500)
    parser.add_argument("--result-count", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.outcome_report.read_text(encoding="utf-8"))
    rows = report["outcomes"][args.rule]
    division_totals = dict(
        parse_assignment(value) for value in args.division_total
    )
    result = search_rules(
        rows,
        division_totals,
        args.max_depth,
        args.beam_width,
        args.result_count,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
