#!/usr/bin/env python3
"""Build a private E029 package from a complete, explicitly classified evaluation.

The reference Notebook is acquired separately from its public author. This
builder preserves the verified inference cells and freezes the published
settings; no training-label validator or parameter sweep is shipped.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import re
from pathlib import Path

import run_geometric_reference as reference


REQUIRED_GATES = {
    "full_corpus", "pooled_gain_002", "adjusted_edge_not_regressed",
    "both_embryos_positive", "both_halves_positive",
    "affected_wins_exceed_losses", "affected_median_positive",
}
AUTHOR_SELECTION_NAMES = {
    "44b6_12dfb391", "44b6_267148e4", "44b6_2a2eff9f", "44b6_341df25f",
    "6bba_062c8d37", "6bba_07e24132", "6bba_085bf656", "6bba_09961292",
}


def classify_selection(report: dict, allow_component_tradeoff: bool = False) -> dict:
    gates = report.get("gates", {})
    if set(gates) != REQUIRED_GATES or any(value is not True and value is not False for value in gates.values()):
        raise ValueError("Complete frozen gate results are required")
    required = REQUIRED_GATES - {"adjusted_edge_not_regressed"} if allow_component_tradeoff else REQUIRED_GATES
    if any(gates[key] is not True for key in required):
        raise ValueError("Required selection gates did not pass")
    if not allow_component_tradeoff and report.get("promotion_passed") is not True:
        raise ValueError("Every frozen promotion gate must pass")
    if allow_component_tradeoff and report["groups"]["all"]["delta"]["division_jaccard"] <= 0:
        raise ValueError("A component tradeoff requires a division-score gain")
    return {"selection_policy": "primary_metric_challenger" if allow_component_tradeoff else "all_frozen_gates",
            "original_frozen_gate_passed": report.get("promotion_passed") is True,
            "failed_frozen_gates": sorted(key for key, passed in gates.items() if not passed),
            "official_component_deltas": report["groups"]["all"]["delta"]}


def require_promotion(run_dir: Path, allow_component_tradeoff: bool = False) -> dict:
    load = lambda name: json.loads((run_dir / name).read_text())
    manifest = load("run_manifest.json")
    if (manifest.get("mode") != "full" or manifest.get("reference_sha256") != reference.REFERENCE_SHA256
            or manifest.get("frozen_overrides") != reference.FROZEN_OVERRIDES
            or manifest.get("deepcenter_sha256") != reference.DEEPCENTER_SHA256
            or manifest.get("output_bounds_policy") != reference.OUTPUT_BOUNDS_POLICY
            or manifest.get("inference_and_topology_passed") is not True):
        raise ValueError("Full evaluation provenance is missing or changed")
    names = manifest["datasets"]
    if len(names) != 64 or reference.stability.movie_names_sha256(names) != reference.CORPUS_SHA256:
        raise ValueError("Promotion corpus changed")
    report = load("stability.json")
    selection = classify_selection(report, allow_component_tradeoff)
    if not math.isclose(report["groups"]["all"]["control"]["score"],
                        0.9014331472, abs_tol=1e-9, rel_tol=0):
        raise ValueError("E025 control parity was not established")
    audit = load("topology_audit.json")
    if audit.get("passed") is not True or sorted(audit["datasets"]) != sorted(names):
        raise ValueError("Complete topology audit is missing")
    if audit["submission_sha256"] != reference.stability.file_sha256(run_dir / "submission.csv"):
        raise ValueError("Evaluated submission bytes changed")
    export = load("export_boundary_audit.json")
    if (export.get("policy") != reference.OUTPUT_BOUNDS_POLICY
            or export.get("submission_sha256") != audit["submission_sha256"]):
        raise ValueError("Validated spatial export policy is missing")
    return {**selection, "evaluation_commit": manifest["git_commit"],
            "evaluation_submission_sha256": audit["submission_sha256"],
            "official_control_score": report["groups"]["all"]["control"]["score"],
            "official_candidate_score": report["groups"]["all"]["candidate"]["score"],
            "output_bounds_policy": reference.OUTPUT_BOUNDS_POLICY,
            "stability_sha256": reference.stability.file_sha256(run_dir / "stability.json")}


def remove_status_prints(source: str) -> str:
    # Reference introductory prints mention historical scores/settings. Retain
    # every assignment and assertion while omitting those presentation lines.
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == "print")]
    return ast.unparse(tree)


def check_author_selection_overlap(run_dir: Path, runtime_dir: Path, scorer_dir: Path) -> dict:
    """Score the movies outside the author's eight-movie parameter selection."""
    report = json.loads((run_dir / "stability.json").read_text())
    rows = report["official_rows"]
    names = sorted(rows["control"])
    if names != sorted(rows["candidate"]) or len(names) != 64:
        raise ValueError("Per-movie scoring evidence is incomplete")
    remaining = sorted(set(names) - AUTHOR_SELECTION_NAMES)
    if len(remaining) != 59:
        raise ValueError("Published parameter-selection overlap changed")
    official = reference.stability.load_official_scorer(runtime_dir, scorer_dir)
    groups = {}
    for group, subset in {"all": remaining, **{
            prefix: [n for n in remaining if n.startswith(prefix + "_")]
            for prefix in ("44b6", "6bba")}}.items():
        control = reference.stability.official_summary(official, rows["control"], subset)
        candidate = reference.stability.official_summary(official, rows["candidate"], subset)
        groups[group] = {"n": len(subset), "control": control, "candidate": candidate,
                         "delta": reference.stability.summary_delta(control, candidate)}
    gates = {"pooled_gain_002": groups["all"]["delta"]["score"] >= 0.002,
             "both_embryos_positive": all(groups[g]["delta"]["score"] > 0 for g in ("44b6", "6bba"))}
    audit = {"excluded": sorted(set(names) & AUTHOR_SELECTION_NAMES), "groups": groups,
             "gates": gates, "passed": all(gates.values()),
             "evidence_boundary": "Outside the author's parameter-selection movies; still not training-disjoint CV"}
    (run_dir / "author_selection_overlap.json").write_text(json.dumps(audit, indent=2) + "\n")
    if not audit["passed"]:
        raise ValueError("Gains do not generalize beyond the author's selection movies")
    return audit


def make_notebook(sources: list[str], proof: dict) -> dict:
    audit_source = inspect.getsource(reference.validate_submission).replace(
        "stability.file_sha256(csv_path)", "hashlib.sha256(csv_path.read_bytes()).hexdigest()")
    export_source = inspect.getsource(reference.normalize_export_boundary)
    for variable in ("source_path", "output_path"):
        export_source = export_source.replace(f"stability.file_sha256({variable})",
                                              f"hashlib.sha256({variable}.read_bytes()).hexdigest()")
    overrides = repr({"BIOHUB_" + k: str(v) for k, v in reference.FROZEN_OVERRIDES.items()})
    configuration = (
        "# Frozen author-published postprocessing overrides; no runtime tuning.\n"
        f"os.environ.update({overrides})\n"
        "os.environ['BIOHUB_VALIDATOR_ENABLE'] = '0'\n"
    )
    audit = (
        "import hashlib\n" + export_source + "\n" + audit_source + "\n"
        "e029_raw_csv = WORKING_DIR / 'raw_submission.csv'\n"
        "SUBMISSION_PATH.rename(e029_raw_csv)\n"
        "e029_export = normalize_export_boundary(e029_raw_csv, SUBMISSION_PATH, TEST_DIR, sorted(test_stems))\n"
        "(WORKING_DIR / 'export_boundary_audit.json').write_text(json.dumps(e029_export, indent=2) + '\\n')\n"
        "e029_audit = validate_submission(SUBMISSION_PATH, TEST_DIR, sorted(test_stems))\n"
        f"e029_audit['external_reference_sha256'] = {reference.REFERENCE_SHA256!r}\n"
        f"e029_audit['frozen_overrides'] = {reference.FROZEN_OVERRIDES!r}\n"
        f"e029_audit['offline_promotion'] = {proof!r}\n"
        "e029_audit['status'] = 'kernel_output_audited_public_score_pending'\n"
        "e029_audit['effective_environment'] = {k:v for k,v in os.environ.items() if k.startswith('BIOHUB_')}\n"
        "(WORKING_DIR / 'e029_output_audit.json').write_text(json.dumps(e029_audit, indent=2) + '\\n')\n"
        "print(json.dumps(e029_audit, indent=2))\n"
    )
    code = [remove_status_prints(sources[0]), remove_status_prints(sources[1]),
            configuration, *sources[2:6], audit]
    for i, source in enumerate(code):
        compile(source, f"E029-cell-{i}", "exec")
    introduction = (
        "# E029: Frozen geometric reference reproduction\n\n"
        "Reproduces [Aman Atar's Geometric Fusion](https://www.kaggle.com/code/amanatar/biohub-geometric-fusion), "
        "which extends [Igor Zharov's Harmonic Fusion](https://www.kaggle.com/code/flexonafft/biohub-harmonic-fusion). "
        "Original model artifacts are by Pilkwang Kim. This is an attributed external-method comparison.\n\n"
        "The author's published postprocessing selection is frozen. No validator, training-label "
        "selection, or parameter sweep executes during this Notebook. Offline promotion used "
        "the pinned official scorer on 64 development movies; this is not training-disjoint CV "
        "and does not establish a leaderboard score. The raw reference output is preserved; "
        "a spatial coordinate rounded exactly one voxel past an image boundary is constrained "
        "to the last voxel before auditing and submission. Larger excursions fail.\n"
    )
    if proof.get("selection_policy") == "primary_metric_challenger":
        introduction += (
            "\nThis exploratory candidate improves the official overall score and the registered "
            "embryo/half stability checks, but loses adjusted-edge score. It failed the original "
            "all-component promotion rule. A subsequent primary-metric selection decision "
            "retains that failed result and permits one public-score test without retuning.\n"
        )
    cells = [{"cell_type": "markdown", "metadata": {}, "source": introduction}]
    cells.extend({"cell_type": "code", "metadata": {}, "source": source,
                  "execution_count": None, "outputs": []} for source in code)
    return {"nbformat": 4, "nbformat_minor": 4, "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-notebook", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--allow-component-tradeoff", action="store_true",
                        help="Explicit exploratory primary-score selection; retain failed component gates")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+", args.kernel):
        raise ValueError("Invalid Kaggle Kernel identifier")
    proof = require_promotion(args.run_dir, args.allow_component_tradeoff)
    proof["outside_author_selection"] = check_author_selection_overlap(
        args.run_dir, args.runtime_dir, args.scorer_dir)
    notebook = make_notebook(reference.read_reference(args.reference_notebook), proof)
    metadata = json.loads((Path(__file__).parent / "kernel-metadata.json").read_text())
    metadata.update({"id": args.kernel, "title": "Biohub E029 | Frozen Geometric Reference",
                     "code_file": "biohub_e029_geometric.ipynb", "is_private": True})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in ((metadata["code_file"], notebook), ("kernel-metadata.json", metadata)):
        (args.output_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"kernel": args.kernel, "notebook_sha256": reference.stability.file_sha256(
        args.output_dir / metadata["code_file"]), "proof": proof}, indent=2))


if __name__ == "__main__":
    main()
