#!/usr/bin/env python3
"""Package E031 only after its complete, frozen comparison passes.

This builder has no Kaggle upload or submission action. It retains the E029
inference pipeline and changes only the checksum-pinned association function.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import build_geometric_submission as geometric
import run_flow_relink_experiment as flow

reference = flow.reference
REQUIRED_GATES = (geometric.REQUIRED_GATES - {"pooled_gain_002"}) | {"pooled_gain_minimum"}


def check_report(report: dict) -> dict:
    gates = report.get("gates", {})
    if (set(gates) != REQUIRED_GATES or any(value is not True for value in gates.values())
            or report.get("promotion_passed") is not True
            or report.get("minimum_pooled_delta") != 0.001):
        raise ValueError("Every frozen E031 gate must pass")
    groups = report["groups"]
    overall = groups["all"]
    if not math.isclose(overall["control"]["score"], flow.E029_SCORE, rel_tol=0, abs_tol=1e-9):
        raise ValueError("E029 official control parity is missing")
    if (not overall["candidate"]["score"] - overall["control"]["score"] >= 0.001
            or not overall["delta"]["score"] >= 0.001
            or not overall["delta"]["adj_edge_jaccard"] >= 0
            or any(not groups[key]["delta"]["score"] > 0
                   for key in ("44b6", "6bba", "half0", "half1"))
            or report["paired"]["wins"] <= report["paired"]["losses"]
            or (report["paired"]["median_delta"] or 0) <= 0):
        raise ValueError("E031 numerical results do not satisfy its frozen gates")
    return {"selection_policy": "all_frozen_e031_gates", "minimum_pooled_delta": 0.001,
            "official_control_score": overall["control"]["score"],
            "official_candidate_score": overall["candidate"]["score"],
            "official_component_deltas": overall["delta"]}


def require_promotion(run_dir: Path, flow_function: str) -> dict:
    load = lambda name: json.loads((run_dir / name).read_text())
    manifest = load("run_manifest.json")
    function_hash = hashlib.sha256(flow_function.encode()).hexdigest()
    expected = {
        "experiment": "E031", "mode": "full",
        "reference_sha256": reference.REFERENCE_SHA256,
        "flow_reference_sha256": flow.FLOW_REFERENCE_SHA256,
        "flow_function_sha256": function_hash,
        "flow_config": flow.FLOW_CONFIG, "e029_overrides": reference.FROZEN_OVERRIDES,
        "deepcenter_sha256": reference.DEEPCENTER_SHA256,
        "control_csv_sha256": flow.E029_CSV_SHA256,
        "control_replay_sha256": flow.E029_CSV_SHA256,
        "raw_graphs_unchanged": True, "runtime_parameter_search": False,
        "minimum_pooled_delta": 0.001, "output_bounds_policy": reference.OUTPUT_BOUNDS_POLICY,
        "inference_and_topology_passed": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Full E031 provenance is missing or changed")
    if (manifest.get("flow_frames", 0) <= 0
            or not re.fullmatch(r"[a-f0-9]{64}", manifest.get("raw_graph_tree_sha256", ""))):
        raise ValueError("Active flow and immutable raw graphs must be recorded")
    names = manifest["datasets"]
    if len(names) != 64 or reference.stability.movie_names_sha256(names) != reference.CORPUS_SHA256:
        raise ValueError("E031 promotion corpus changed")
    if reference.stability.file_sha256(run_dir / "control/submission.csv") != flow.E029_CSV_SHA256:
        raise ValueError("Exact E029 replay bytes changed")
    proof = check_report(load("stability.json"))
    audit = load("topology_audit.json")
    digest = reference.stability.file_sha256(run_dir / "submission.csv")
    if (audit.get("passed") is not True or sorted(audit["datasets"]) != sorted(names)
            or audit.get("submission_sha256") != digest):
        raise ValueError("Complete E031 topology audit or evaluated bytes changed")
    export = load("export_boundary_audit.json")
    if export.get("policy") != reference.OUTPUT_BOUNDS_POLICY or export.get("submission_sha256") != digest:
        raise ValueError("E031 spatial export evidence changed")
    return {**proof, "evaluation_commit": manifest["git_commit"],
            "evaluation_submission_sha256": digest,
            "stability_sha256": reference.stability.file_sha256(run_dir / "stability.json"),
            "manifest_sha256": reference.stability.file_sha256(run_dir / "run_manifest.json"),
            "control_replay_sha256": flow.E029_CSV_SHA256,
            "flow_function_sha256": function_hash, "flow_frames": manifest["flow_frames"],
            "raw_graph_tree_sha256": manifest["raw_graph_tree_sha256"]}


def check_author_exclusion(run_dir: Path, runtime_dir: Path, scorer_dir: Path) -> dict:
    report = json.loads((run_dir / "stability.json").read_text())
    rows = report["official_rows"]
    names = sorted(rows["control"])
    if (names != sorted(rows["candidate"])
            or reference.stability.movie_names_sha256(names) != reference.CORPUS_SHA256):
        raise ValueError("Complete E031 per-movie evidence is missing")
    remaining = sorted(set(names) - geometric.AUTHOR_SELECTION_NAMES)
    if len(remaining) != 59:
        raise ValueError("Author-selection overlap changed")
    official = reference.stability.load_official_scorer(runtime_dir, scorer_dir)
    groups = {}
    for key, subset in {"all": remaining, **{
            prefix: [name for name in remaining if name.startswith(prefix + "_")]
            for prefix in ("44b6", "6bba")}}.items():
        control = reference.stability.official_summary(official, rows["control"], subset)
        candidate = reference.stability.official_summary(official, rows["candidate"], subset)
        groups[key] = {"n": len(subset), "control": control, "candidate": candidate,
                       "delta": reference.stability.summary_delta(control, candidate)}
    if any(not group["delta"]["score"] > 0 for group in groups.values()):
        raise ValueError("E031 failed the frozen author-exclusion check")
    return {"excluded": sorted(set(names) & geometric.AUTHOR_SELECTION_NAMES),
            "groups": groups, "passed": True,
            "evidence_boundary": "Outside author-selection movies; still adaptive development, not training-disjoint CV"}


def make_notebook(sources: list[str], flow_function: str, proof: dict) -> dict:
    adapted = list(sources)
    adapted[5] = flow.patch_postprocessing(adapted[5], flow_function)
    notebook = geometric.make_notebook(adapted, proof)
    notebook["cells"][0]["source"] = (
        "# E031: Frozen neighborhood-flow association\n\n"
        "Uses [Aman Atar's Geometric Fusion](https://www.kaggle.com/code/amanatar/biohub-geometric-fusion) "
        "and its [Harmonic Fusion](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion) foundation, "
        "with original models by Pilkwang Kim. Only the motion association function is replaced "
        "with [Anvith Pothula's x138 neighborhood flow](https://www.kaggle.com/code/anvithpothula/biohub-x138). "
        "The x138 coordinate head, readmission and additional gap filling are excluded.\n\n"
        "The frozen change passed the complete 64-movie development comparison, exact E029 control "
        "replay, and the 59-movie author-selection exclusion check before packaging. This is "
        "adaptive development evidence, not training-disjoint validation or a public score. "
        "No label-dependent validation or parameter selection runs in this Notebook. "
        "Test predictions are generated at runtime and independently audited after the same "
        "one-voxel spatial boundary normalization used by E029.\n"
    )
    audit = notebook["cells"][-1]["source"].replace("e029_", "e031_")
    insertion = (
        f"e031_audit['flow_reference_sha256'] = {flow.FLOW_REFERENCE_SHA256!r}\n"
        f"e031_audit['flow_function_sha256'] = {hashlib.sha256(flow_function.encode()).hexdigest()!r}\n"
        f"e031_audit['flow_config'] = {flow.FLOW_CONFIG!r}\n"
        "e031_flow_stats = pd.read_csv(RUN_STATS_PATH)\n"
        "e031_flow_frames = int(e031_flow_stats['motion_relink_flow_frames'].fillna(0).sum())\n"
        "if e031_flow_frames <= 0:\n    raise RuntimeError('Frozen neighborhood flow did not activate')\n"
        "e031_audit['flow_frames'] = e031_flow_frames\n"
    )
    anchor = "(WORKING_DIR / 'e031_output_audit.json')"
    if audit.count(anchor) != 1:
        raise ValueError("Output audit anchor changed")
    notebook["cells"][-1]["source"] = audit.replace(anchor, insertion + anchor)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(cell["source"], f"E031-cell-{index}", "exec")
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-notebook", "flow-reference", "run-dir", "runtime-dir", "scorer-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--kernel", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+", args.kernel):
        raise ValueError("Invalid Kaggle Kernel identifier")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    function = flow.read_flow_function(args.flow_reference)
    proof = require_promotion(args.run_dir, function)
    proof["outside_author_selection"] = check_author_exclusion(args.run_dir, args.runtime_dir, args.scorer_dir)
    notebook = make_notebook(reference.read_reference(args.reference_notebook), function, proof)
    metadata = json.loads((Path(__file__).parent / "kernel-metadata.json").read_text())
    metadata.update({"id": args.kernel, "title": "Biohub E031 | Frozen Neighborhood Flow",
                     "code_file": "biohub_e031_flow.ipynb", "is_private": True})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in ((metadata["code_file"], notebook), ("kernel-metadata.json", metadata),
                          ("development_proof.json", proof)):
        (args.output_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "run_summary.md").write_text(
        "# E031 private package\n\nAll frozen development checks passed. "
        "The package is built but has not been uploaded or submitted by this builder.\n\n"
        f"Notebook SHA256: `{reference.stability.file_sha256(args.output_dir / metadata['code_file'])}`\n")
    print(json.dumps({"kernel": args.kernel, "proof": proof}, indent=2))


if __name__ == "__main__":
    main()
