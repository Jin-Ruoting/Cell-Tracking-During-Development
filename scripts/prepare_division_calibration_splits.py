#!/usr/bin/env python3
"""Build balanced, leakage-safe calibration shards from a division report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _closest_negatives(
    positives: list[str],
    negatives: list[str],
    per_dataset: dict[str, dict[str, int]],
) -> list[str]:
    available = set(negatives)
    selected: list[str] = []
    for positive in positives:
        if not available:
            break
        target_edges = int(per_dataset[positive]["edges"])
        choice = min(
            available,
            key=lambda name: (
                abs(int(per_dataset[name]["edges"]) - target_edges),
                name,
            ),
        )
        selected.append(choice)
        available.remove(choice)
    return selected


def select_datasets(
    per_dataset: dict[str, dict[str, int]],
    excluded: set[str],
    positive_limit_per_embryo: int,
) -> tuple[list[str], dict[str, str]]:
    by_embryo: dict[str, list[str]] = defaultdict(list)
    for name in per_dataset:
        if name not in excluded:
            by_embryo[name.split("_", 1)[0]].append(name)

    selected: list[str] = []
    roles: dict[str, str] = {}
    for embryo, names in sorted(by_embryo.items()):
        positives = sorted(
            (
                name
                for name in names
                if int(per_dataset[name]["divisions"]) > 0
            ),
            key=lambda name: (
                -int(per_dataset[name]["divisions"]),
                -int(per_dataset[name]["edges"]),
                name,
            ),
        )[:positive_limit_per_embryo]
        negatives = [
            name
            for name in names
            if int(per_dataset[name]["divisions"]) == 0
        ]
        matched_negatives = _closest_negatives(
            positives,
            negatives,
            per_dataset,
        )
        for name in positives:
            roles[name] = "positive"
        for name in matched_negatives:
            roles[name] = "negative"
        selected.extend(positives)
        selected.extend(matched_negatives)

    return sorted(selected), roles


def shard_datasets(
    selected: list[str],
    per_dataset: dict[str, dict[str, int]],
    shard_count: int,
) -> tuple[list[list[str]], list[int]]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    ordered = sorted(
        selected,
        key=lambda name: (-int(per_dataset[name]["edges"]), name),
    )
    for name in ordered:
        target = min(range(shard_count), key=lambda index: (loads[index], index))
        shards[target].append(name)
        loads[target] += int(per_dataset[name]["edges"])
    for shard in shards:
        shard.sort()
    return shards, loads


def build_manifest(
    report: dict[str, object],
    excluded: set[str],
    positive_limit_per_embryo: int,
    shard_count: int,
) -> dict[str, object]:
    per_dataset = report.get("per_dataset")
    if not isinstance(per_dataset, dict):
        raise ValueError("report must contain a per_dataset object")
    selected, roles = select_datasets(
        per_dataset,
        excluded,
        positive_limit_per_embryo,
    )
    shards, loads = shard_datasets(selected, per_dataset, shard_count)
    return {
        "excluded": sorted(excluded),
        "positive_limit_per_embryo": positive_limit_per_embryo,
        "selected": selected,
        "roles": roles,
        "shard_edge_loads": loads,
        "shards": shards,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("division_report", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Dataset name to exclude. May be supplied multiple times.",
    )
    parser.add_argument("--positive-limit-per-embryo", type=int, default=16)
    parser.add_argument("--shards", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.positive_limit_per_embryo < 1:
        raise ValueError("positive limit must be positive")
    report = json.loads(args.division_report.read_text(encoding="utf-8"))
    manifest = build_manifest(
        report,
        set(args.exclude),
        args.positive_limit_per_embryo,
        args.shards,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, datasets in enumerate(manifest["shards"]):
        split = [{"train": [], "val": [], "test": datasets}]
        (args.output_dir / f"shard_{index}.json").write_text(
            json.dumps(split, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
