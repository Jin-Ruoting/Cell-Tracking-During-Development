#!/usr/bin/env python3
"""Reproduce a checksum-pinned public method on a fixed Biohub corpus.

Reference: https://www.kaggle.com/code/amanatar/biohub-geometric-fusion
The externally downloaded Notebook is required, not redistributed here. Its
published postprocessing selection is frozen below; its validator and parameter
sweep are never executed. Temporary runtime copies stay inside the run folder.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import evaluate_ab_stability as stability


REFERENCE_SHA256 = "f82a606e2e3289f1d6dce078d381dd7b0caf148c92bb25f2f05bea80d4e79a37"
CORPUS_SHA256 = "276c09d16cddaf2e865896ce147161a7beb5a62142bf47c1f1bd7648f7643e7f"
DEEPCENTER_SHA256 = "8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0"
SMOKE_NAMES = ("44b6_eb2880fc", "6bba_969618f6")
# Taken from the public run's ppsweep_selected.json before our first score.
FROZEN_OVERRIDES = {
    "DEEPCENTER_SAFE_DIV_THRESHOLD": 0.15,
    "LEAF_PRUNE_MIN_EDGE_PROB": 0.3,
    "MOTION_RELINK_TIGHT_UM": 5.5,
    "MOTION_RELINK_VELOCITY_WEIGHT": 0.25,
    "SAFE_DIV_EXISTING_CHILD_MAX_UM": 12.0,
    "SAFE_DIV_MAX_UM": 11.0,
    "SAFE_DIV_SISTER_MAX_UM": 16.0,
}


def read_reference(path: Path) -> list[str]:
    if stability.file_sha256(path) != REFERENCE_SHA256:
        raise ValueError("Public reference checksum changed; refusing execution")
    notebook = json.loads(path.read_text())
    if len(notebook["cells"]) != 12:
        raise ValueError("Unexpected reference cell layout")
    sources = []
    for cell in notebook["cells"]:
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        if cell["cell_type"] != "code":
            raise ValueError("Unexpected reference cell type")
        compile(source, str(path), "exec")
        sources.append(source)
    return sources


def adapt_cell(source: str, assignments: dict[str, str] | None = None,
               skip_calls: tuple[str, ...] = ()) -> str:
    """Change only named top-level infrastructure statements, fail on drift."""
    assignments = assignments or {}
    tree = ast.parse(source)
    counts = dict.fromkeys([*assignments, *skip_calls], 0)
    result = []
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in assignments):
            name = node.targets[0].id
            node.value = ast.parse(assignments[name], mode="eval").body
            counts[name] += 1
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in skip_calls):
            counts[node.value.func.id] += 1
            continue
        result.append(node)
    if any(value != 1 for value in counts.values()):
        raise ValueError(f"Infrastructure anchors changed: {counts}")
    tree.body = result
    return ast.unparse(ast.fix_missing_locations(tree))


def selected_names(control_dir: Path, mode: str) -> list[str]:
    names = sorted(stability.movie_paths(control_dir))
    if len(names) != 64 or stability.movie_names_sha256(names) != CORPUS_SHA256:
        raise ValueError("Frozen 64-movie E025 corpus changed")
    if mode == "smoke":
        if not set(SMOKE_NAMES) <= set(names):
            raise ValueError("Smoke movies missing from control corpus")
        return list(SMOKE_NAMES)
    return names


def adapt_worker_paths(source: str) -> str:
    anchor = '"PYTHONPATH": "src"'
    if source.count(anchor) != 2:
        raise ValueError("Reference worker environment anchors changed")
    return source.replace(anchor, '"PYTHONPATH": os.pathsep.join(["src", os.environ.get("PYTHONPATH", "")])')


def validate_submission(csv_path: Path, image_dir: Path, names: list[str]) -> dict:
    """Independently validate the actual rounded submission coordinates."""
    import numpy as np
    import pandas as pd
    import zarr

    frame = pd.read_csv(csv_path)
    columns = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
               "source_id", "target_id"]
    if frame.columns.tolist() != columns or frame.empty:
        raise ValueError("Submission schema or row count is invalid")
    if frame["id"].tolist() != list(range(len(frame))):
        raise ValueError("Nonsequential row IDs")
    if sorted(frame.dataset.unique()) != sorted(names):
        raise ValueError("Submission movie coverage mismatch")
    if set(frame.row_type.unique()) != {"node", "edge"}:
        raise ValueError("Invalid row types")
    numeric = frame[[c for c in columns if c not in {"dataset", "row_type"}]].to_numpy()
    if not np.isfinite(numeric).all() or not (numeric == np.rint(numeric)).all():
        raise ValueError("Nonfinite or noninteger submission value")
    reports = {}
    for name, group in frame.groupby("dataset", sort=True):
        nodes = group[group.row_type == "node"]
        edges = group[group.row_type == "edge"]
        if nodes.empty or nodes.node_id.duplicated().any() or (nodes.node_id < 0).any():
            raise ValueError(f"{name}: missing or invalid node IDs")
        shape = zarr.open(str(image_dir / f"{name}.zarr"), mode="r")["0"].shape
        xyz = nodes[["t", "z", "y", "x"]].to_numpy()
        if (xyz < 0).any() or (xyz >= np.asarray(shape)).any():
            raise ValueError(f"{name}: out-of-volume node")
        if not (nodes[["source_id", "target_id"]].to_numpy() == -1).all():
            raise ValueError(f"{name}: invalid node placeholders")
        if not (edges[["node_id", "t", "z", "y", "x"]].to_numpy() == -1).all():
            raise ValueError(f"{name}: invalid edge placeholders")
        times = nodes.set_index("node_id").t
        st, tt = edges.source_id.map(times), edges.target_id.map(times)
        if st.isna().any() or tt.isna().any() or not (tt - st == 1).all():
            raise ValueError(f"{name}: dangling or nonconsecutive edge")
        if edges.duplicated(["source_id", "target_id"]).any():
            raise ValueError(f"{name}: duplicate edge")
        indegree = edges.target_id.value_counts()
        outdegree = edges.source_id.value_counts()
        if (indegree > 1).any() or (outdegree > 2).any():
            raise ValueError(f"{name}: invalid lineage degree")
        reports[name] = {"nodes": len(nodes), "edges": len(edges),
                         "divisions": int((outdegree == 2).sum()),
                         "max_indegree": int(indegree.max()) if len(indegree) else 0,
                         "max_outdegree": int(outdegree.max()) if len(outdegree) else 0}
    return {"passed": True, "rows": len(frame), "datasets": reports,
            "submission_sha256": stability.file_sha256(csv_path)}


def evaluate(args, names: list[str]) -> dict:
    official = stability.load_official_scorer(args.runtime_dir, args.scorer_dir)
    candidate_dir = args.output_dir / "geffs"
    if candidate_dir.exists():
        raise FileExistsError(candidate_dir)
    subprocess.run([sys.executable, str(args.scorer_dir / "scripts/csv_to_geffs.py"),
                    "--csv", str(args.output_dir / "submission.csv"),
                    "--out-dir", str(candidate_dir)], check=True)
    if sorted(stability.movie_paths(candidate_dir)) != names:
        raise ValueError("Converted GEFF movie coverage mismatch")
    rows = {"control": {}, "candidate": {}}
    signatures = {"control": {}, "candidate": {}}
    records = []
    for index, name in enumerate(names):
        for arm, directory in (("control", args.control_dir), ("candidate", candidate_dir)):
            rows[arm][name], signatures[arm][name] = stability.score_one(
                official, name, directory / f"{name}.geff", args.data_dir / "train")
        c = stability.per_movie_diagnostic(rows["control"][name])
        v = stability.per_movie_diagnostic(rows["candidate"][name])
        delta = v["score"] - c["score"]
        record = {"dataset": name, "embryo": name.split("_")[0],
                  "affected": signatures["control"][name] != signatures["candidate"][name],
                  "control_score": c["score"], "candidate_score": v["score"],
                  "delta_score": delta, "outcome": stability.outcome(delta, 1e-12)}
        records.append(record)
        print(f"SCORED {index + 1}/{len(names)} {name} delta={delta:+.8f}", flush=True)
    partitions = stability.build_partitions(names) if len(names) == 64 else {
        "all": names, "44b6": [names[0]], "6bba": [names[1]]}
    groups = {}
    for key, subset in partitions.items():
        c = stability.official_summary(official, rows["control"], subset)
        v = stability.official_summary(official, rows["candidate"], subset)
        groups[key] = {"control": c, "candidate": v, "delta": stability.summary_delta(c, v)}
    if len(names) == 64 and not math.isclose(
        float(groups["all"]["control"]["score"]), 0.9014331472, rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError("Official E025 control score no longer reproduces S157")
    affected = stability.paired_stats([r for r in records if r["affected"]])
    gates = {
        "full_corpus": len(names) == 64,
        "pooled_gain_002": groups["all"]["delta"]["score"] >= 0.002,
        "adjusted_edge_not_regressed": groups["all"]["delta"]["adj_edge_jaccard"] >= 0,
        "both_embryos_positive": all(groups[g]["delta"]["score"] > 0 for g in ("44b6", "6bba")),
        "both_halves_positive": len(names) == 64 and all(
            groups[g]["delta"]["score"] > 0 for g in ("half0", "half1")),
        "affected_wins_exceed_losses": affected["wins"] > affected["losses"],
        "affected_median_positive": (affected["median_delta"] or 0) > 0,
    }
    report = {"experiment": "E029 frozen public geometric reference", "groups": groups,
              "gates": gates, "promotion_passed": all(gates.values()), "paired": affected,
              "per_movie": records, "graph_signatures": signatures, "official_rows": rows,
              "evidence_boundary": "Two-embryo-stratified development evaluation; not training-disjoint CV"}
    (args.output_dir / "stability.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "per_movie.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return report


def run(args) -> None:
    names = selected_names(args.control_dir, args.mode)
    sources = read_reference(args.reference_notebook)
    deepcenter = args.data_dir / "deepcenter-v1-full/weights/full_frame_center/best.pt"
    if stability.file_sha256(deepcenter) != DEEPCENTER_SHA256:
        raise ValueError("DeepCenter epoch-2 checkpoint bytes changed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    input_dir = args.output_dir / "input"
    image_dir = input_dir / "test"
    image_dir.mkdir(parents=True)
    for name in names:
        source = args.data_dir / "train" / f"{name}.zarr"
        if not source.is_dir():
            raise FileNotFoundError(source)
        (image_dir / source.name).symlink_to(source, target_is_directory=True)
    manifest = {"experiment": "E029", "mode": args.mode, "datasets": names,
                "reference_sha256": REFERENCE_SHA256, "frozen_overrides": FROZEN_OVERRIDES,
                "deepcenter_sha256": DEEPCENTER_SHA256, "runtime_parameter_search": False,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sys.path[:0] = [str(args.runtime_dir), str(args.data_dir / "support-pack/repo/src")]
    os.environ["PYTHONPATH"] = os.pathsep.join([
        str(args.runtime_dir), str(args.scorer_dir / "src"), str(args.scorer_dir / "scripts"),
        str(args.data_dir / "support-pack/repo/src")])
    for key in list(os.environ):
        if key.startswith("BIOHUB_"):
            del os.environ[key]
    os.environ.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                       "OPENBLAS_NUM_THREADS": "4", "POLARS_MAX_THREADS": "4"})
    os.chdir(args.output_dir)
    namespace = {"__name__": "biohub_e029_reference"}
    for i in (0, 1):
        exec(compile(sources[i], f"reference:cell-{i}", "exec"), namespace)
    os.environ.update({"BIOHUB_" + key: str(value) for key, value in FROZEN_OVERRIDES.items()})
    os.environ.update({
        "BIOHUB_PRIMARY_ARTIFACT_MANIFEST": str(args.data_dir / "support-pack/ARTIFACT_MANIFEST.json"),
        "BIOHUB_SECONDARY_ARTIFACT_MANIFEST": str(args.data_dir / "secondary-seed-v1/ARTIFACT_MANIFEST.json"),
        "BIOHUB_DEEPCENTER_ARTIFACT_MANIFEST": str(args.data_dir / "deepcenter-v1-full/ARTIFACT_MANIFEST.json"),
        "BIOHUB_DEEPCENTER_CHECKPOINT": str(deepcenter), "BIOHUB_VALIDATOR_ENABLE": "0",
    })
    # The server runtime has no Jupyter display package; this is only used
    # to display configuration tables, not by prediction or postprocessing.
    display_import = "from IPython.display import display"
    if sources[2].count(display_import) != 1:
        raise ValueError("Reference display import changed")
    sources[2] = sources[2].replace(display_import, "display = print", 1)
    sources[2] = adapt_cell(sources[2], {
        "COMP_DIR": f"Path({str(input_dir)!r})", "WORKING_DIR": f"Path({str(args.output_dir)!r})"})
    sources[3] = adapt_cell(sources[3], {
        "ARTIFACTS": f"Path({str(args.data_dir / 'support-pack')!r})"}, ("ensure_dependencies",))
    sources[4] = adapt_worker_paths(sources[4])
    # Cell 6 has an obsolete hard-coded configuration receipt. Use the
    # independent audit below instead; cells 7 onward select on labels.
    for i in range(2, 6):
        source = sources[i].replace("/kaggle/working", str(args.output_dir))
        print(f"EXECUTING REFERENCE CELL {i}", flush=True)
        exec(compile(source, f"reference:cell-{i}", "exec"), namespace)
    audit = validate_submission(args.output_dir / "submission.csv", image_dir, names)
    (args.output_dir / "topology_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    manifest["effective_environment"] = {k: v for k, v in os.environ.items() if k.startswith("BIOHUB_")}
    manifest["patched_predictor_sha256"] = stability.file_sha256(
        args.output_dir / "tracking_repo/scripts/predict_unet_transformer.py")
    manifest["inference_and_topology_passed"] = True
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = evaluate(args, names)
    print(json.dumps({"groups": result["groups"], "gates": result["gates"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-notebook", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    run(args)


if __name__ == "__main__":
    main()
