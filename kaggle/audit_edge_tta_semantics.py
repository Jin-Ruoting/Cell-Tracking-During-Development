#!/usr/bin/env python3
"""Audit the semantic and structural gates for an edge-TTA candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType


EXPECTED_EVALUATOR_SHA256 = (
    "03ad4049530d3682c77435194e5d921981f331df3462abcde4a7156d1a57b7d3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-preilp-dir", type=Path, required=True)
    parser.add_argument("--candidate-preilp-dir", type=Path, required=True)
    parser.add_argument("--candidate-final-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def movie_paths(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    paths = sorted(directory.glob("*.geff"))
    result = {path.stem: path for path in paths}
    if len(result) != len(paths):
        raise RuntimeError(f"Duplicate GEFF stems under {directory}")
    return result


def load_official_scorer(
    runtime_dir: Path,
    scorer_dir: Path,
) -> ModuleType:
    evaluator_path = scorer_dir / "scripts" / "evaluate.py"
    actual_sha256 = file_sha256(evaluator_path)
    if actual_sha256 != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError(
            "Pinned official evaluator SHA256 changed: "
            f"expected {EXPECTED_EVALUATOR_SHA256}, got {actual_sha256}"
        )
    sys.path[:0] = [
        str(runtime_dir.resolve()),
        str((scorer_dir / "src").resolve()),
        str((scorer_dir / "scripts").resolve()),
    ]
    spec = importlib.util.spec_from_file_location(
        "biohub_official_evaluate_edge_tta_audit",
        evaluator_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official evaluator {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def node_signature(graph) -> list[tuple[int, int, float, float, float]]:
    return sorted(
        (
            int(row["node_id"]),
            int(row["t"]),
            float(row["z"]),
            float(row["y"]),
            float(row["x"]),
        )
        for row in graph.node_attrs().iter_rows(named=True)
    )


def edge_probabilities(graph) -> dict[tuple[int, int], float]:
    return {
        (int(row["source_id"]), int(row["target_id"])): float(
            row["edge_prob"]
        )
        for row in graph.edge_attrs().iter_rows(named=True)
    }


def audit_final_graph(graph) -> dict[str, int]:
    node_rows = list(graph.node_attrs().iter_rows(named=True))
    edge_rows = list(graph.edge_attrs().iter_rows(named=True))
    node_ids = {int(row["node_id"]) for row in node_rows}
    times = {
        int(row["node_id"]): int(row["t"])
        for row in node_rows
    }
    indegree: dict[int, int] = {}
    outdegree: dict[int, int] = {}
    dangling_edges = 0
    nonconsecutive_edges = 0
    duplicate_edges = 0
    seen_edges: set[tuple[int, int]] = set()
    for row in edge_rows:
        source = int(row["source_id"])
        target = int(row["target_id"])
        edge = (source, target)
        if edge in seen_edges:
            duplicate_edges += 1
        seen_edges.add(edge)
        if source not in node_ids or target not in node_ids:
            dangling_edges += 1
        elif times[target] != times[source] + 1:
            nonconsecutive_edges += 1
        indegree[target] = indegree.get(target, 0) + 1
        outdegree[source] = outdegree.get(source, 0) + 1
    return {
        "nodes": len(node_rows),
        "edges": len(edge_rows),
        "duplicate_edges": duplicate_edges,
        "dangling_edges": dangling_edges,
        "nonconsecutive_edges": nonconsecutive_edges,
        "max_indegree": max(indegree.values(), default=0),
        "max_outdegree": max(outdegree.values(), default=0),
        "nonbinary_sources": sum(value > 2 for value in outdegree.values()),
    }


def main() -> None:
    args = parse_args()
    if args.expected_count <= 0:
        raise ValueError("--expected-count must be positive")
    if args.output_json.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_json}")

    control_paths = movie_paths(args.control_preilp_dir)
    candidate_paths = movie_paths(args.candidate_preilp_dir)
    final_paths = movie_paths(args.candidate_final_dir)
    if not (
        set(control_paths) == set(candidate_paths) == set(final_paths)
    ):
        raise RuntimeError("Control pre-ILP, candidate pre-ILP, and final sets differ")
    names = sorted(control_paths)
    if len(names) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} movies, found {len(names)}"
        )

    official = load_official_scorer(args.runtime_dir, args.scorer_dir)
    per_movie: dict[str, dict[str, object]] = {}
    total_changed = 0
    for name in names:
        control = official._load_graph(control_paths[name])
        candidate = official._load_graph(candidate_paths[name])
        final = official._load_graph(final_paths[name])
        control_nodes = node_signature(control)
        candidate_nodes = node_signature(candidate)
        control_edges = edge_probabilities(control)
        candidate_edges = edge_probabilities(candidate)
        changed = sum(
            control_edges[edge] != candidate_edges[edge]
            for edge in control_edges.keys() & candidate_edges.keys()
        )
        total_changed += changed
        topology = audit_final_graph(final)
        per_movie[name] = {
            "control_nodes": len(control_nodes),
            "candidate_nodes": len(candidate_nodes),
            "node_signatures_exact": control_nodes == candidate_nodes,
            "control_edges": len(control_edges),
            "candidate_edges": len(candidate_edges),
            "edge_key_sets_exact": set(control_edges) == set(candidate_edges),
            "changed_common_edge_probabilities": changed,
            "all_control_probabilities_finite": all(
                math.isfinite(value) for value in control_edges.values()
            ),
            "all_candidate_probabilities_finite": all(
                math.isfinite(value) for value in candidate_edges.values()
            ),
            **topology,
        }

    gates = {
        "all_node_signatures_exact": all(
            row["node_signatures_exact"] for row in per_movie.values()
        ),
        "all_edge_key_sets_exact": all(
            row["edge_key_sets_exact"] for row in per_movie.values()
        ),
        "all_probabilities_finite": all(
            row["all_control_probabilities_finite"]
            and row["all_candidate_probabilities_finite"]
            for row in per_movie.values()
        ),
        "at_least_one_probability_changed": total_changed > 0,
        "strict_binary_lineage_topology": all(
            row["duplicate_edges"] == 0
            and row["dangling_edges"] == 0
            and row["nonconsecutive_edges"] == 0
            and row["max_indegree"] <= 1
            and row["max_outdegree"] <= 2
            and row["nonbinary_sources"] == 0
            for row in per_movie.values()
        ),
    }
    report = {
        "schema_version": 1,
        "expected_count": args.expected_count,
        "official_evaluator_sha256": EXPECTED_EVALUATOR_SHA256,
        "total_changed_common_edge_probabilities": total_changed,
        "per_movie": per_movie,
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"gates": gates, "passed": report["passed"]}))
    print(f"Wrote {args.output_json}")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
