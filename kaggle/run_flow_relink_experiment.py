#!/usr/bin/env python3
"""Evaluate one frozen neighborhood-flow change on the E029 detector graph.

Only the motion_relink_edges function is read from the external x138 Notebook.
Its private coordinate head, inference changes, and label-dependent cells are
not executed. Raw E029 predictions and their exact final CSV are required.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import run_geometric_reference as reference

FLOW_REFERENCE_SHA256 = "6b655e39bbfd2d3d6c762badea69847d3f00f5b548f385cb01b07ee2600fde6d"
E029_CSV_SHA256 = "1d4fd28c02cb54d2279a120794b26e21fa0741d743bf62781d8ed17d5a26e2ce"
E029_SCORE = 0.9090442379185286
FLOW_CONFIG = {
    "MOTION_RELINK_FLOW_MODE": "seed",
    "MOTION_RELINK_FLOW_K": 12,
    "MOTION_RELINK_FLOW_RADIUS_UM": 40.0,
    "MOTION_RELINK_FLOW_EXCLUDE_UM": 1.5,
    "MOTION_RELINK_FLOW_MIN_SAMPLES": 4,
    "MOTION_RELINK_FLOW_GATE": True,
    "MOTION_RELINK_FLOW_ITER": 1,
    "MOTION_RELINK_FLOW_SEED_GATE_UM": 0.0,
    "MOTION_RELINK_FLOW_RAW_ADMIT": True,
    "MOTION_RELINK_FLOW_Z_WEIGHT": 1.0,
    "MOTION_RELINK_FLOW_RAW_COST": 0.0,
    "MOTION_RELINK_FLOW_TIGHT_UM": 7.0,
    "MOTION_RELINK_FLOW_RELAXED_UM": 0.0,
}


def function_node(source: str, name: str = "motion_relink_edges") -> ast.FunctionDef:
    matches = [n for n in ast.parse(source).body
               if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1 or matches[0].decorator_list:
        raise ValueError("Expected one undecorated top-level motion function")
    return matches[0]


def read_flow_function(path: Path) -> str:
    if reference.stability.file_sha256(path) != FLOW_REFERENCE_SHA256:
        raise ValueError("Flow reference checksum changed")
    cells = json.loads(path.read_text())["cells"]
    if len(cells) != 12 or cells[5]["cell_type"] != "code":
        raise ValueError("Flow reference layout changed")
    return ast.unparse(function_node("".join(cells[5]["source"])))


def patch_postprocessing(source: str, flow_function: str) -> str:
    """Replace exactly one function; preserve every other statement."""
    tree = ast.parse(source)
    original = function_node(source)
    replacement = function_node(flow_function)
    tree.body = [replacement if isinstance(n, ast.FunctionDef) and n.name == original.name else n
                 for n in tree.body]
    settings = ast.parse("\n".join(f"{key} = {value!r}" for key, value in FLOW_CONFIG.items()))
    tree.body = settings.body + tree.body
    return ast.unparse(ast.fix_missing_locations(tree))


def graph_tree_sha256(graphs: list[Path]) -> str:
    digest = hashlib.sha256()
    for graph in sorted(graphs):
        files = sorted(p for p in graph.rglob("*") if p.is_file())
        if not files:
            raise ValueError(f"Empty raw graph: {graph.name}")
        for path in files:
            identity = f"{graph.name}/{path.relative_to(graph)}"
            digest.update(f"{identity} {reference.stability.file_sha256(path)}\n".encode())
    return digest.hexdigest()


def write_subset(source: Path, output: Path, names: list[str]) -> None:
    selected = set(names)
    with source.open(newline="") as src, output.open("x", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        row_id = 0
        for row in reader:
            if row["dataset"] in selected:
                row["id"] = row_id
                writer.writerow(row)
                row_id += 1


def require_replay_parity(actual: Path, expected: Path) -> str:
    digest = reference.stability.file_sha256(actual)
    if digest != reference.stability.file_sha256(expected):
        raise ValueError("E029 replay is not byte-identical to the frozen control")
    return digest


def setup_namespace(args, names: list[str], sources: list[str], deepcenter: Path) -> dict:
    for key in list(os.environ):
        if key.startswith("BIOHUB_"):
            del os.environ[key]
    namespace = {"__name__": "biohub_e031_replay"}
    for i in (0, 1):
        exec(compile(sources[i], f"E029:cell-{i}", "exec"), namespace)
    os.environ.update({"BIOHUB_" + key: str(value) for key, value in reference.FROZEN_OVERRIDES.items()})
    os.environ.update({
        "BIOHUB_DEEPCENTER_ARTIFACT_MANIFEST": str(args.data_dir / "deepcenter-v1-full/ARTIFACT_MANIFEST.json"),
        "BIOHUB_DEEPCENTER_CHECKPOINT": str(deepcenter), "BIOHUB_VALIDATOR_ENABLE": "0",
    })
    config = sources[2]
    display_import = "from IPython.display import display"
    if config.count(display_import) != 1:
        raise ValueError("Reference display import changed")
    config = reference.adapt_cell(config.replace(display_import, "display = print", 1), {
        "COMP_DIR": f"Path({str(args.output_dir / 'input')!r})",
        "WORKING_DIR": f"Path({str(args.output_dir)!r})",
    })
    exec(compile(config, "E029:cell-2", "exec"), namespace)
    namespace.update({"test_stems": names, "predict_seconds": 0.0})
    definitions = reference.adapt_cell(sources[5], skip_calls=("write_test_submission",))
    exec(compile(definitions, "E029:postprocessing", "exec"), namespace)
    return namespace


def write_summary(args, result: dict, parity: str) -> None:
    lines = ["# E031 frozen neighborhood-flow experiment", "",
             f"Mode: {args.mode}", "", f"Control replay SHA256: {parity}", "",
             "Only association uses neighborhood flow; original E029 raw detections are reused.", "",
             "| Group | E029 control | E031 | Difference |", "|---|---:|---:|---:|"]
    for key, group in result["groups"].items():
        lines.append(f"| {key} | {group['control']['score']:.9f} | {group['candidate']['score']:.9f} | {group['delta']['score']:+.9f} |")
    lines += ["", "## Frozen advancement gates", "",
              "```json", json.dumps(result["gates"], indent=2), "```", "",
              f"Promotion passed: {result['promotion_passed']}", "",
              result["evidence_boundary"], ""]
    (args.output_dir / "run_summary.md").write_text("\n".join(lines))


def run(args) -> None:
    names = reference.selected_names(args.control_dir, args.mode)
    sources = reference.read_reference(args.reference_notebook)
    flow_function = read_flow_function(args.flow_reference)
    if reference.stability.file_sha256(args.control_csv) != E029_CSV_SHA256:
        raise ValueError("Frozen E029 control CSV changed")
    original = json.loads((args.raw_run_dir / "run_manifest.json").read_text())
    if (original.get("reference_sha256") != reference.REFERENCE_SHA256
            or original.get("frozen_overrides") != reference.FROZEN_OVERRIDES
            or original.get("deepcenter_sha256") != reference.DEEPCENTER_SHA256):
        raise ValueError("Raw E029 inference provenance changed")
    found = sorted((args.raw_run_dir / "tracking_repo/predictions").glob("*/unet_transformer/split_0/*.geff"))
    all_names = sorted(reference.stability.movie_paths(args.control_dir))
    if sorted(p.stem for p in found) != all_names or original.get("datasets") != all_names:
        raise ValueError("Raw E029 graph coverage changed")
    graphs = [path for path in found if path.stem in names]
    raw_hash = graph_tree_sha256(graphs)
    deepcenter = args.data_dir / "deepcenter-v1-full/weights/full_frame_center/best.pt"
    if reference.stability.file_sha256(deepcenter) != reference.DEEPCENTER_SHA256:
        raise ValueError("DeepCenter checkpoint changed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    image_dir = args.output_dir / "input/test"
    graph_dir = args.output_dir / "tracking_repo/predictions/frozen/unet_transformer/split_0"
    image_dir.mkdir(parents=True)
    graph_dir.mkdir(parents=True)
    for graph in graphs:
        (image_dir / f"{graph.stem}.zarr").symlink_to(args.data_dir / "train" / f"{graph.stem}.zarr", target_is_directory=True)
        (graph_dir / graph.name).symlink_to(graph, target_is_directory=True)
    control_output = args.output_dir / "control"
    control_output.mkdir()
    manifest = {"experiment": "E031", "mode": args.mode, "datasets": names,
                "reference_sha256": reference.REFERENCE_SHA256,
                "flow_reference_sha256": FLOW_REFERENCE_SHA256,
                "flow_function_sha256": hashlib.sha256(flow_function.encode()).hexdigest(),
                "flow_config": FLOW_CONFIG, "e029_overrides": reference.FROZEN_OVERRIDES,
                "raw_graph_tree_sha256": raw_hash, "control_csv_sha256": E029_CSV_SHA256,
                "raw_manifest_sha256": reference.stability.file_sha256(args.raw_run_dir / "run_manifest.json"),
                "deepcenter_sha256": reference.DEEPCENTER_SHA256,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "minimum_pooled_delta": 0.001, "runtime_parameter_search": False}
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    sys.path[:0] = [str(args.runtime_dir), str(args.data_dir / "support-pack/repo/src")]
    os.environ["PYTHONPATH"] = os.pathsep.join([str(args.runtime_dir), str(args.scorer_dir / "src"),
                                               str(args.scorer_dir / "scripts")])
    os.environ.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                       "OPENBLAS_NUM_THREADS": "4", "POLARS_MAX_THREADS": "4"})
    namespace = setup_namespace(args, names, sources, deepcenter)
    namespace.update({"SUBMISSION_PATH": control_output / "raw_submission.csv",
                      "RUN_STATS_PATH": control_output / "run_stats.csv"})
    namespace["write_test_submission"]("e029_exact_replay")
    control_export = reference.normalize_export_boundary(
        control_output / "raw_submission.csv", control_output / "submission.csv", image_dir, names)
    (control_output / "export_boundary_audit.json").write_text(json.dumps(control_export, indent=2) + "\n")
    control_audit = reference.validate_submission(control_output / "submission.csv", image_dir, names)
    (control_output / "topology_audit.json").write_text(json.dumps(control_audit, indent=2) + "\n")
    expected = args.control_csv
    if args.mode == "smoke":
        expected = control_output / "expected_submission.csv"
        write_subset(args.control_csv, expected, names)
    parity = require_replay_parity(control_output / "submission.csv", expected)
    print(f"E029 CONTROL BYTE PARITY PASSED: {parity}", flush=True)
    namespace.update(FLOW_CONFIG)
    exec(compile(flow_function, "x138:motion_relink_edges-only", "exec"), namespace)
    namespace.update({"SUBMISSION_PATH": args.output_dir / "raw_submission.csv",
                      "RUN_STATS_PATH": args.output_dir / "run_stats.csv"})
    namespace["write_test_submission"]("e031_frozen_flow")
    with (args.output_dir / "run_stats.csv").open() as handle:
        stats = list(csv.DictReader(handle))
    flow_frames = sum(int(float(row.get("motion_relink_flow_frames") or 0)) for row in stats)
    if flow_frames <= 0:
        raise ValueError("Neighborhood flow did not activate")
    if graph_tree_sha256(graphs) != raw_hash:
        raise ValueError("Raw prediction files changed during replay")
    manifest.update({"control_replay_sha256": parity, "flow_frames": flow_frames,
                     "raw_graphs_unchanged": True})
    args.expected_control_score = E029_SCORE
    args.minimum_pooled_delta = 0.001
    args.experiment = "E031 frozen neighborhood flow on E029"
    reference.complete_export(args, names, manifest, args.output_dir / "raw_submission.csv")
    result = json.loads((args.output_dir / "stability.json").read_text())
    write_summary(args, result, parity)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-notebook", "flow-reference", "data-dir", "control-dir", "control-csv",
                 "raw-run-dir", "runtime-dir", "scorer-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    try:
        run(args)
    except Exception as exc:
        if args.output_dir.exists():
            (args.output_dir / "run_summary.md").write_text(
                f"# E031 execution failed\n\n{type(exc).__name__}: {exc}\n\nNo promotion or public score is established.\n")
        raise


if __name__ == "__main__":
    main()
