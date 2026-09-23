#!/usr/bin/env python3
"""Package the fixed E035 comparison after complete development/output checks."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

import build_flow_submission as checks
import run_motion_ema_experiment as ema

reference = ema.reference
flow = ema.flow
geometric = checks.geometric


def require_promotion(run_dir, function):
    load = lambda name: json.loads((run_dir / name).read_text())
    manifest = load("run_manifest.json")
    function_hash = hashlib.sha256(function.encode()).hexdigest()
    expected = {
        "experiment": "E035", "mode": "full", "ema_alpha": ema.EMA_ALPHA,
        "velocity_weight": reference.FROZEN_OVERRIDES["MOTION_RELINK_VELOCITY_WEIGHT"],
        "reference_sha256": reference.REFERENCE_SHA256, "motion_function_sha256": function_hash,
        "e029_overrides": reference.FROZEN_OVERRIDES, "deepcenter_sha256": reference.DEEPCENTER_SHA256,
        "raw_graph_tree_sha256": ema.RAW_GRAPH_SHA256, "raw_graphs_unchanged": True,
        "control_csv_sha256": flow.E029_CSV_SHA256, "control_replay_sha256": flow.E029_CSV_SHA256,
        "runtime_parameter_search": False, "minimum_pooled_delta": 0.001,
        "output_bounds_policy": reference.OUTPUT_BOUNDS_POLICY, "inference_and_topology_passed": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Full E035 provenance is missing or changed")
    if not 0 < manifest.get("ema_changed_velocities", 0) <= manifest.get("ema_updates", 0):
        raise ValueError("The evaluated EMA was not active")
    names = manifest["datasets"]
    if len(names) != 64 or reference.stability.movie_names_sha256(names) != reference.CORPUS_SHA256:
        raise ValueError("E035 promotion corpus changed")
    if reference.stability.file_sha256(Path(manifest["control_replay_source"])) != flow.E029_CSV_SHA256:
        raise ValueError("Exact E029 replay bytes changed")
    proof = checks.check_report(load("stability.json"), "E035")
    audit = load("topology_audit.json")
    digest = reference.stability.file_sha256(run_dir / "submission.csv")
    if (audit.get("passed") is not True or sorted(audit["datasets"]) != sorted(names)
            or audit.get("submission_sha256") != digest):
        raise ValueError("Complete E035 topology audit or evaluated bytes changed")
    export = load("export_boundary_audit.json")
    if export.get("policy") != reference.OUTPUT_BOUNDS_POLICY or export.get("submission_sha256") != digest:
        raise ValueError("E035 spatial export evidence changed")
    return {**proof, "evaluation_commit": manifest["git_commit"],
            "evaluation_submission_sha256": digest,
            "stability_sha256": reference.stability.file_sha256(run_dir / "stability.json"),
            "manifest_sha256": reference.stability.file_sha256(run_dir / "run_manifest.json"),
            "control_replay_sha256": flow.E029_CSV_SHA256, "raw_graph_tree_sha256": ema.RAW_GRAPH_SHA256,
            "motion_function_sha256": function_hash, "ema_alpha": ema.EMA_ALPHA,
            "velocity_weight": manifest["velocity_weight"], "ema_updates": manifest["ema_updates"],
            "ema_changed_velocities": manifest["ema_changed_velocities"]}


def make_notebook(sources, function, proof):
    adapted = list(sources)
    tree = ast.parse(adapted[5])
    original = flow.function_node(adapted[5])
    replacement = flow.function_node(function)
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == original.name]
    if len(matches) != 1:
        raise ValueError("Expected one top-level motion function")
    tree.body = [replacement if node is matches[0] else node for node in tree.body]
    adapted[5] = ast.unparse(ast.fix_missing_locations(tree))
    notebook = geometric.make_notebook(adapted, proof)
    notebook["cells"][0]["source"] = (
        "# E035: Frozen per-track motion EMA\n\n"
        "Retains [Aman Atar's Geometric Fusion](https://www.kaggle.com/code/amanatar/biohub-geometric-fusion), "
        "its [Harmonic Fusion](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion) foundation, "
        "and original model artifacts by Pilkwang Kim. The sole change is our implementation of "
        "per-track velocity EMA with alpha 0.4 and unchanged motion weight 0.25. The direction was "
        "suggested by [JunhaoLiXD's experiment report](https://github.com/JunhaoLiXD/Biohub_Cell_Tracking/blob/main/docs/experiments.md); "
        "this is an independent adaptation, not its full method reproduction.\n\n"
        "The unchanged candidate passed all frozen 64-movie development gates after a negative two-movie "
        "runtime check. Packaging additionally checks the 59 movies outside the geometric author's "
        "parameter-selection set. These are adaptive development results, not training-disjoint validation "
        "or a public score. No training-label validator or parameter search runs here.\n"
    )
    audit = notebook["cells"][-1]["source"].replace("e029_", "e035_")
    insertion = (
        f"e035_audit['motion_function_sha256'] = {hashlib.sha256(function.encode()).hexdigest()!r}\n"
        f"e035_audit['ema_alpha'] = {ema.EMA_ALPHA!r}\n"
        f"e035_audit['velocity_weight'] = {reference.FROZEN_OVERRIDES['MOTION_RELINK_VELOCITY_WEIGHT']!r}\n"
        "e035_stats = pd.read_csv(RUN_STATS_PATH)\n"
        "e035_updates = int(e035_stats['motion_ema_updates'].fillna(0).sum())\n"
        "e035_changed = int(e035_stats['motion_ema_changed_velocities'].fillna(0).sum())\n"
        "if not 0 < e035_changed <= e035_updates:\n    raise RuntimeError('Frozen EMA was not active')\n"
        "e035_audit['ema_updates'] = e035_updates\n"
        "e035_audit['ema_changed_velocities'] = e035_changed\n"
    )
    anchor = "(WORKING_DIR / 'e035_output_audit.json')"
    if audit.count(anchor) != 1:
        raise ValueError("Output audit anchor changed")
    notebook["cells"][-1]["source"] = audit.replace(anchor, insertion + anchor)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(cell["source"], f"E035-cell-{index}", "exec")
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-notebook", "run-dir", "runtime-dir", "scorer-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--kernel", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+", args.kernel):
        raise ValueError("Invalid Kaggle Kernel identifier")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    sources = reference.read_reference(args.reference_notebook)
    function = ema.motion_function(sources[5])
    proof = require_promotion(args.run_dir, function)
    proof["outside_author_selection"] = checks.check_author_exclusion(
        args.run_dir, args.runtime_dir, args.scorer_dir, "E035")
    notebook = make_notebook(sources, function, proof)
    metadata = json.loads((Path(__file__).parent / "kernel-metadata.json").read_text())
    metadata.update({"id": args.kernel, "title": "Biohub E035 | Frozen Motion EMA",
                     "code_file": "biohub_e035_motion_ema.ipynb", "is_private": True})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in ((metadata["code_file"], notebook), ("kernel-metadata.json", metadata),
                          ("development_proof.json", proof)):
        (args.output_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "run_summary.md").write_text(
        "# E035 private package\n\nFrozen comparison and packaging checks passed. "
        "This builder does not upload or submit.\n\n"
        f"Notebook SHA256: `{reference.stability.file_sha256(args.output_dir / metadata['code_file'])}`\n")
    print(json.dumps({"kernel": args.kernel, "proof": proof}, indent=2))


if __name__ == "__main__":
    main()
