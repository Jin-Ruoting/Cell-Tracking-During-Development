#!/usr/bin/env python3
"""Build a private E029 Kaggle package only from a passed full evaluation.

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


def require_promotion(run_dir: Path) -> dict:
    load = lambda name: json.loads((run_dir / name).read_text())
    manifest = load("run_manifest.json")
    if (manifest.get("mode") != "full" or manifest.get("reference_sha256") != reference.REFERENCE_SHA256
            or manifest.get("frozen_overrides") != reference.FROZEN_OVERRIDES
            or manifest.get("deepcenter_sha256") != reference.DEEPCENTER_SHA256
            or manifest.get("inference_and_topology_passed") is not True):
        raise ValueError("Full evaluation provenance is missing or changed")
    names = manifest["datasets"]
    if len(names) != 64 or reference.stability.movie_names_sha256(names) != reference.CORPUS_SHA256:
        raise ValueError("Promotion corpus changed")
    report = load("stability.json")
    gates = report.get("gates", {})
    if (report.get("promotion_passed") is not True or set(gates) != REQUIRED_GATES
            or any(value is not True for value in gates.values())):
        raise ValueError("Every frozen promotion gate must pass")
    if not math.isclose(report["groups"]["all"]["control"]["score"],
                        0.9014331472, abs_tol=1e-9, rel_tol=0):
        raise ValueError("E025 control parity was not established")
    audit = load("topology_audit.json")
    if audit.get("passed") is not True or sorted(audit["datasets"]) != sorted(names):
        raise ValueError("Complete topology audit is missing")
    if audit["submission_sha256"] != reference.stability.file_sha256(run_dir / "submission.csv"):
        raise ValueError("Evaluated submission bytes changed")
    return {"evaluation_commit": manifest["git_commit"],
            "evaluation_submission_sha256": audit["submission_sha256"],
            "official_control_score": report["groups"]["all"]["control"]["score"],
            "official_candidate_score": report["groups"]["all"]["candidate"]["score"],
            "stability_sha256": reference.stability.file_sha256(run_dir / "stability.json")}


def remove_status_prints(source: str) -> str:
    # Reference introductory prints mention historical scores/settings. Retain
    # every assignment and assertion while omitting those presentation lines.
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == "print")]
    return ast.unparse(tree)


def make_notebook(sources: list[str], proof: dict) -> dict:
    audit_source = inspect.getsource(reference.validate_submission).replace(
        "stability.file_sha256(csv_path)", "hashlib.sha256(csv_path.read_bytes()).hexdigest()")
    overrides = repr({"BIOHUB_" + k: str(v) for k, v in reference.FROZEN_OVERRIDES.items()})
    configuration = (
        "# Frozen author-published postprocessing overrides; no runtime tuning.\n"
        f"os.environ.update({overrides})\n"
        "os.environ['BIOHUB_VALIDATOR_ENABLE'] = '0'\n"
    )
    audit = (
        "import hashlib\n" + audit_source + "\n"
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
        "and does not establish a leaderboard score.\n"
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
    parser.add_argument("--kernel", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+", args.kernel):
        raise ValueError("Invalid Kaggle Kernel identifier")
    proof = require_promotion(args.run_dir)
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
