#!/usr/bin/env python3
"""Validate E006 on saved raw and pre-ILP GEFF graphs.

The runner loads the committed notebook as the single source of postprocessing
and searched-division logic. It evaluates four controlled variants:

* E000 postprocessing with legacy safe divisions;
* E000 plus searched divisions;
* E000 postprocessing without legacy safe divisions; and
* the no-legacy-safe variant plus searched divisions.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


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
VARIANTS = {
    "e000_safe": (True, False),
    "e006_safe_search": (True, True),
    "e000_no_safe": (False, False),
    "e006_no_safe_search": (False, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--preilp-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--support-src", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-score", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def definition_prefix(source: str) -> ast.Module:
    tree = ast.parse(source)
    body: list[ast.stmt] = []
    for node in tree.body:
        is_main_boundary = (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "DEEPCENTER_VETO_DETECTOR"
                for target in node.targets
            )
        )
        if is_main_boundary:
            break
        body.append(node)
    if not body or not any(
        isinstance(node, ast.FunctionDef) and node.name == "filter_output_graph"
        for node in body
    ):
        raise RuntimeError("Notebook postprocessing definition boundary changed")
    return ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))


def load_notebook_namespace(args: argparse.Namespace) -> dict[str, object]:
    sys.path.insert(0, str(args.support_src))
    sys.path.insert(0, str(args.runtime_dir))

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if len(cells) != 20:
        raise RuntimeError(f"Expected 20 notebook cells, found {len(cells)}")

    namespace: dict[str, object] = {
        "__name__": "biohub_e006_offline_validation",
        "__file__": str(args.notebook),
    }
    exec(compile(cells[6]["source"], "e006-cell-6", "exec"), namespace)
    exec(compile(cells[7]["source"], "e006-cell-7", "exec"), namespace)
    namespace["TEST_DIR"] = args.image_dir
    exec(
        compile(definition_prefix(cells[13]["source"]), "e006-cell-13-defs", "exec"),
        namespace,
    )

    required = {
        "graph_from_geff",
        "edge_probability_map",
        "select_searched_division_edges",
        "apply_searched_division_edges",
        "filter_output_graph",
    }
    missing = sorted(required - namespace.keys())
    if missing:
        raise RuntimeError(f"Notebook functions missing: {missing}")
    return namespace


def graph_rows(namespace: dict[str, object], path: Path):
    graph = namespace["graph_from_geff"](path)
    nodes: dict[int, dict[str, object]] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        nodes[node_id] = {
            "node_id": node_id,
            "t": int(row["t"]),
            "z": float(row["z"]),
            "y": float(row["y"]),
            "x": float(row["x"]),
        }
    edges: list[dict[str, object]] = []
    for row in graph.edge_attrs().iter_rows(named=True):
        edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
        edges.append(
            {
                "source_id": int(row["source_id"]),
                "target_id": int(row["target_id"]),
                "edge_prob": None if edge_prob is None else float(edge_prob),
            }
        )
    return graph, nodes, edges


def audit_graph(
    dataset: str,
    nodes: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
) -> dict[str, int]:
    edge_pairs: set[tuple[int, int]] = set()
    indegree: dict[int, int] = {}
    outdegree: dict[int, int] = {}
    nonconsecutive = 0
    dangling = 0
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        pair = (source_id, target_id)
        if pair in edge_pairs:
            raise AssertionError(f"{dataset}: duplicate edge {pair}")
        edge_pairs.add(pair)
        if source_id not in nodes or target_id not in nodes:
            dangling += 1
            continue
        if int(nodes[target_id]["t"]) != int(nodes[source_id]["t"]) + 1:
            nonconsecutive += 1
        indegree[target_id] = indegree.get(target_id, 0) + 1
        outdegree[source_id] = outdegree.get(source_id, 0) + 1
    maximum_indegree = max(indegree.values(), default=0)
    maximum_outdegree = max(outdegree.values(), default=0)
    if dangling or nonconsecutive or maximum_indegree > 1 or maximum_outdegree > 2:
        raise AssertionError(
            {
                "dataset": dataset,
                "dangling": dangling,
                "nonconsecutive": nonconsecutive,
                "max_indegree": maximum_indegree,
                "max_outdegree": maximum_outdegree,
            }
        )
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "division_sources": sum(count == 2 for count in outdegree.values()),
        "max_indegree": maximum_indegree,
        "max_outdegree": maximum_outdegree,
    }


def node_signature(nodes: dict[int, dict[str, object]]) -> dict[int, tuple[object, ...]]:
    return {
        node_id: (
            int(node["t"]),
            float(node["z"]),
            float(node["y"]),
            float(node["x"]),
        )
        for node_id, node in nodes.items()
    }


def edge_signature(edges: list[dict[str, object]]) -> set[tuple[int, int]]:
    return {
        (int(edge["source_id"]), int(edge["target_id"]))
        for edge in edges
    }


def write_dataset(
    writer: csv.DictWriter,
    row_id: int,
    dataset: str,
    nodes: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
) -> int:
    for node_id in sorted(nodes):
        node = nodes[node_id]
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
    return row_id


def evaluate_csv(args: argparse.Namespace, variant: str, csv_path: Path) -> str:
    score_dir = args.output_dir / f"score_{variant}"
    geff_dir = score_dir / "geffs"
    score_dir.mkdir()
    geff_dir.mkdir()
    python_path = os.pathsep.join(
        [
            str(args.runtime_dir),
            str(args.scorer_dir / "src"),
            str(args.scorer_dir / "scripts"),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    env = {**os.environ, "PYTHONPATH": python_path}
    subprocess.run(
        [
            sys.executable,
            str(args.scorer_dir / "scripts" / "csv_to_geffs.py"),
            "--csv",
            str(csv_path),
            "--out-dir",
            str(geff_dir),
        ],
        env=env,
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(args.scorer_dir / "scripts" / "evaluate.py"),
            "--pred-dir",
            str(geff_dir),
            "--gt-dir",
            str(args.ground_truth_dir),
        ],
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    score_log = score_dir / "official_score.log"
    score_log.write_text(result.stdout, encoding="utf-8")
    print(f"===== OFFICIAL SCORE {variant} =====")
    print(result.stdout, end="")
    return str(score_log)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    namespace = load_notebook_namespace(args)

    baseline_paths = sorted(args.baseline_dir.glob("*.geff"))
    preilp_paths = {
        path.stem: path for path in sorted(args.preilp_dir.rglob("*.geff"))
    }
    if not baseline_paths:
        raise FileNotFoundError(f"No baseline GEFFs under {args.baseline_dir}")
    baseline_names = {path.stem for path in baseline_paths}
    if baseline_names != set(preilp_paths):
        raise RuntimeError(
            {
                "missing_preilp": sorted(baseline_names - set(preilp_paths)),
                "extra_preilp": sorted(set(preilp_paths) - baseline_names),
            }
        )

    files = {
        name: (args.output_dir / f"{name}.csv").open("w", newline="")
        for name in VARIANTS
    }
    writers = {
        name: csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        for name, handle in files.items()
    }
    for writer in writers.values():
        writer.writeheader()

    row_ids = {name: 0 for name in VARIANTS}
    report: dict[str, object] = {
        "notebook": str(args.notebook.resolve()),
        "notebook_sha256": file_sha256(args.notebook),
        "datasets": [],
        "variants": {name: {} for name in VARIANTS},
    }
    try:
        for index, baseline_path in enumerate(baseline_paths, start=1):
            dataset = baseline_path.stem
            _, raw_nodes, raw_edges = graph_rows(namespace, baseline_path)
            preilp_graph = namespace["graph_from_geff"](preilp_paths[dataset])
            probabilities = namespace["edge_probability_map"](
                preilp_graph.edge_attrs().iter_rows(named=True)
            )
            selected = namespace["select_searched_division_edges"](
                raw_nodes,
                raw_edges,
                probabilities,
            )
            report["datasets"].append(dataset)

            dataset_outputs = {}
            variant_snapshots = {}
            for safe_divisions in (True, False):
                namespace["OUTPUT_SAFE_DIVISIONS"] = safe_divisions
                filtered_nodes, filtered_edges, filter_stats = namespace[
                    "filter_output_graph"
                ](
                    copy.deepcopy(raw_nodes),
                    copy.deepcopy(raw_edges),
                    dataset=dataset,
                    deepcenter_bundle=None,
                )
                for searched_divisions in (False, True):
                    variant = next(
                        name
                        for name, settings in VARIANTS.items()
                        if settings == (safe_divisions, searched_divisions)
                    )
                    nodes = copy.deepcopy(filtered_nodes)
                    edges = copy.deepcopy(filtered_edges)
                    searched_stats = {}
                    if searched_divisions:
                        edges, searched_stats = namespace[
                            "apply_searched_division_edges"
                        ](nodes, edges, selected)
                    audit = audit_graph(dataset, nodes, edges)
                    row_ids[variant] = write_dataset(
                        writers[variant],
                        row_ids[variant],
                        dataset,
                        nodes,
                        edges,
                    )
                    dataset_outputs[variant] = {
                        **audit,
                        "safe_divisions_added": int(
                            filter_stats.get("safe_divisions_added", 0)
                        ),
                        "searched_division_candidates": len(selected),
                        "searched_divisions_added": int(
                            searched_stats.get("searched_divisions_added", 0)
                        ),
                    }
                    variant_snapshots[variant] = {
                        "nodes": node_signature(nodes),
                        "edges": edge_signature(edges),
                    }

            for safe_name, search_name in (
                ("e000_safe", "e006_safe_search"),
                ("e000_no_safe", "e006_no_safe_search"),
            ):
                base = dataset_outputs[safe_name]
                searched = dataset_outputs[search_name]
                base_snapshot = variant_snapshots[safe_name]
                searched_snapshot = variant_snapshots[search_name]
                if base_snapshot["nodes"] != searched_snapshot["nodes"]:
                    raise AssertionError(
                        f"{dataset}: searched branch changed node IDs or coordinates"
                    )
                if not base_snapshot["edges"] <= searched_snapshot["edges"]:
                    raise AssertionError(
                        f"{dataset}: searched branch removed an E000 edge"
                    )
                expected_delta = searched["searched_divisions_added"]
                added_edges = searched_snapshot["edges"] - base_snapshot["edges"]
                if (
                    searched["edges"] - base["edges"] != expected_delta
                    or len(added_edges) != expected_delta
                ):
                    raise AssertionError(
                        f"{dataset}: searched edge delta mismatch for {search_name}"
                    )

            for variant in VARIANTS:
                report["variants"][variant][dataset] = dataset_outputs[variant]
            print(
                f"[{index:02d}/{len(baseline_paths):02d}] {dataset} "
                + " ".join(
                    f"{name}=n{dataset_outputs[name]['nodes']}/"
                    f"e{dataset_outputs[name]['edges']}/"
                    f"d{dataset_outputs[name]['division_sources']}"
                    for name in VARIANTS
                ),
                flush=True,
            )
    finally:
        for handle in files.values():
            handle.close()

    report["outputs"] = {}
    for variant in VARIANTS:
        csv_path = args.output_dir / f"{variant}.csv"
        report["outputs"][variant] = {
            "path": str(csv_path),
            "rows": row_ids[variant],
            "sha256": file_sha256(csv_path),
        }

    if not args.skip_score:
        for variant in VARIANTS:
            csv_path = args.output_dir / f"{variant}.csv"
            report["outputs"][variant]["score_log"] = evaluate_csv(
                args, variant, csv_path
            )

    report_path = args.output_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
