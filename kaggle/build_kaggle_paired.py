#!/usr/bin/env python3
"""Package E039 only after a verified independent Kaggle control repeat.

This local source-packaging step does not run model/data experiments or upload.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

import build_kaggle_development as packaging
import run_kaggle_paired as paired


def require_paired_smoke(path: Path | None, protocol: str, baseline: dict):
    if path is None:
        raise ValueError("E039 full requires a completed same-protocol paired smoke receipt")
    receipt = json.loads(path.read_text())
    if (receipt.get("experiment") != "E039" or receipt.get("mode") != "smoke"
            or receipt.get("technical_check_passed") is not True
            or receipt.get("protocol_sha256") != protocol or receipt.get("cloud_baseline") != baseline
            or receipt["control"].get("paired_control_frozen_before_candidate") is not True
            or receipt["control"].get("csv_sha256") != baseline["csv_sha256"]):
        raise ValueError("E039 smoke does not validate these frozen sources and cloud control")


def build(args):
    root = Path(__file__).resolve().parents[1]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.owner):
        raise ValueError("Invalid Kaggle owner")
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=root, check=True, stdout=subprocess.DEVNULL)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != subprocess.check_output(["git", "rev-parse", "@{upstream}"], cwd=root, text=True).strip():
        raise ValueError("Push local source before packaging E039")
    packaging.reference.read_reference(args.reference_notebook)
    files = packaging.source_closure(root, ["kaggle/run_kaggle_paired.py", "tests/test_point_gap_bridge.py"])
    for name, source in files.items():
        if source != subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root).decode():
            raise ValueError("Uncommitted E039 source: " + name)
    protocol = packaging.protocol_digest(files)
    directory = args.repeat_dir
    paths = {"control.csv": directory / "control/control.csv",
             "control_diagnostic.json": directory / "control_diagnostic.json",
             "run_manifest.json": directory / "run_manifest.json"}
    baseline = paired.verify_repeat_evidence(
        json.loads(paths["control_diagnostic.json"].read_text()), json.loads(paths["run_manifest.json"].read_text()),
        packaging.reference.stability.file_sha256(paths["control.csv"]))
    if args.mode == "full":
        require_paired_smoke(args.smoke_receipt, protocol, baseline)
    for name, path in paths.items():
        files["cloud-control/" + name] = path.read_bytes().decode("utf-8")
    files.update(packaging.scoring_files(args.scorer_archive))
    files["reference.ipynb"] = args.reference_notebook.read_text()
    manifest = {"experiment": "E039", "git_commit": revision, "mode": args.mode,
                "protocol_sha256": protocol, "cloud_baseline": baseline,
                "reference_sha256": packaging.reference.REFERENCE_SHA256,
                "scorer_archive_sha256": packaging.SCORER_ARCHIVE_SHA256,
                "completed_control": None,
                "files": {name: packaging.sha256(source.encode()) for name, source in files.items()}}
    files["bundle_manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    metadata = {"id": f"{args.owner}/biohub-e039-kaggle-paired-{args.mode}",
                "title": "Biohub E039 Kaggle Paired " + args.mode.title(),
                "code_file": "e039_paired.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1",
                                    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
                                    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
                                    "hengck23/hengck23-cell-point-detector-demo"],
                "competition_sources": ["biohub-cell-tracking-during-development"], "kernel_sources": [],
                "model_sources": [], "docker_image": packaging.DOCKER, "machine_shape": "NvidiaTeslaT4"}
    source = packaging.bootstrap_source(files, runner_name="run_kaggle_paired.py", work_name="e039-development",
                                        phases=("control", "peaks", "bridges", "score"))
    compile(source, "e039-bootstrap", "exec")
    intro = ("# E039 fixed Kaggle paired development\n\nPrivate training-movie evaluation. "
             "No competition submission. One immutable control graph is shared by both arms. "
             "This does not certify historical E038 server parity.\n\n"
             "References: [Geometric Fusion](https://www.kaggle.com/code/amanatar/biohub-geometric-fusion), "
             "[v5 detector](https://www.kaggle.com/datasets/hengck23/hengck23-cell-point-detector-demo), "
             "[official scorer](https://github.com/royerlab/kaggle-cell-tracking-competition/tree/075fc5f). "
             "Original licenses and sources are retained privately.\n\n"
             f"Source: `{revision}`. Mode: `{args.mode}`. Protocol: `{protocol}`.\n")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [{"cell_type": "markdown", "metadata": {}, "id": "protocol", "source": intro},
                          {"cell_type": "code", "metadata": {}, "id": "run", "source": source,
                           "outputs": [], "execution_count": None}]}
    size = len((json.dumps(notebook, indent=2) + "\n").encode())
    if size >= 1_000_000:
        raise ValueError("E039 package exceeds Kaggle's one-megabyte source limit")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("kernel-metadata.json", metadata), ("e039_paired.ipynb", notebook),
                        ("bundle_manifest.json", manifest)):
        (args.output_dir / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"id": metadata["id"], "is_private": True, "mode": args.mode, "git_commit": revision,
                      "notebook_bytes": size, "bundled_files": len(files)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--reference-notebook", type=Path, required=True)
    parser.add_argument("--scorer-archive", type=Path, required=True)
    parser.add_argument("--repeat-dir", type=Path, required=True,
                        help="Completed independent control repeat outputs, including diagnostic, CSV and manifest")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
