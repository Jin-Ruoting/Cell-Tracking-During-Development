#!/usr/bin/env python3
"""Package reviewed code for a private, offline Kaggle E038 development run.

This builder does no model/data computation and uploads nothing. Downloaded
third-party sources and the generated Notebook stay outside the public repo.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import zlib

import run_geometric_reference as reference

SCORER_ARCHIVE_SHA256 = "0e31329953304331d83f32b5d6ab40f1424ed6391ea2b4e6ad5b56eba62ab367"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_closure(root: Path, entrypoints: list[str]) -> dict[str, str]:
    pending, files = list(entrypoints), {}
    while pending:
        relative = pending.pop()
        if relative in files:
            continue
        text = (root / relative).read_text()
        tree = ast.parse(text, filename=relative)
        files[relative] = text
        for node in ast.walk(tree):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                       [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                dependency = "kaggle/" + module.split(".")[0] + ".py"
                if (root / dependency).is_file():
                    pending.append(dependency)
    return files


def scoring_files(archive: Path) -> dict[str, str]:
    if reference.stability.file_sha256(archive) != SCORER_ARCHIVE_SHA256:
        raise ValueError("Official scorer archive checksum changed")
    files = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            parts = PurePosixPath(member.name).parts
            relative = PurePosixPath(*parts[1:])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe official archive member")
            include = (relative.as_posix() in {"LICENSE", "scripts/evaluate.py", "scripts/csv_to_geffs.py", "scripts/dataspec.py"}
                       or relative.as_posix().startswith("src/") and relative.suffix == ".py")
            if include and member.isfile():
                files["official/" + relative.as_posix()] = handle.extractfile(member).read().decode()
    evaluator = files["official/scripts/evaluate.py"]
    if sha256(evaluator.encode()) != reference.stability.EXPECTED_EVALUATOR_SHA256:
        raise ValueError("Official evaluator checksum changed")
    return files


def protocol_digest(files: dict[str, str]) -> str:
    core = {name: sha256(text.encode()) for name, text in files.items()
            if name.startswith("kaggle/")}
    return sha256(json.dumps(core, sort_keys=True).encode())


def require_smoke_receipt(path: Path | None, protocol: str):
    if path is None:
        raise ValueError("Full run requires the completed Kaggle smoke receipt")
    receipt = json.loads(path.read_text())
    if (receipt.get("mode") != "smoke" or receipt.get("technical_check_passed") is not True
            or receipt.get("control_byte_parity") is not True or receipt.get("control_score_reproduced") is not True
            or receipt.get("protocol_sha256") != protocol):
        raise ValueError("Smoke receipt does not validate the current frozen sources")


def bootstrap_source(files: dict[str, str]) -> str:
    compressed = zlib.compress(json.dumps(files, ensure_ascii=False, sort_keys=True).encode(), level=9)
    encoded = base64.b64encode(compressed).decode()
    return f'''import base64, hashlib, json, os, subprocess, sys, zlib
from pathlib import Path

payload = base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest() == {sha256(compressed)!r}
bundle = Path("/kaggle/working/e038-bundle")
bundle.mkdir(exist_ok=False)
for name, text in json.loads(zlib.decompress(payload)).items():
    target = (bundle / name).resolve()
    if bundle not in target.parents:
        raise ValueError("Invalid bundled path")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
work = Path("/kaggle/working/logs/e038-development")
work.mkdir(parents=True, exist_ok=False)
env = {{**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
       "OPENBLAS_NUM_THREADS": "4", "POLARS_MAX_THREADS": "4"}}

def run_logged(label, command):
    with (work / (label + ".log")).open("x") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   env=env, cwd=bundle)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait() != 0:
            if not (work / "run_summary.md").exists():
                (work / "run_summary.md").write_text("# Kaggle development failed\\n\\nStage: " + label + "\\n")
            raise RuntimeError(label + " failed; inspect its retained log")

run_logged("semantic-tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_point_gap_bridge.py", "-v"])
runner = [sys.executable, str(bundle / "kaggle/run_kaggle_development.py"), "--bundle", str(bundle), "--phase"]
run_logged("control", runner + ["control"])
if not json.loads((work / "control_parity.json").read_text())["byte_parity"]:
    run_logged("diagnose-control", runner + ["diagnose-control"])
    (work / "run_summary.md").write_text("# Control migration mismatch\\n\\nE029 CSV did not reproduce its historical checksum. Diagnostic score retained; E038 not run.\\n")
    raise RuntimeError("E029 migration parity failed; no candidate or parameter adjustment")
for phase in ("peaks", "bridges", "score"):
    run_logged(phase, runner + [phase])
print((work / "run_summary.md").read_text())
'''


def build(args):
    root = Path(__file__).resolve().parents[1]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.owner):
        raise ValueError("Invalid Kaggle owner")
    # A pushed source revision is required before any cloud run is packaged.
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=root, check=True, stdout=subprocess.DEVNULL)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    remote = subprocess.check_output(["git", "rev-parse", "@{upstream}"], cwd=root, text=True).strip()
    if revision != remote:
        raise ValueError("Push the local source commit before building the cloud run")
    reference.read_reference(args.reference_notebook)
    files = source_closure(root, ["kaggle/run_kaggle_development.py", "tests/test_point_gap_bridge.py"])
    for name, text in files.items():
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root).decode()
        if committed != text:
            raise ValueError("Uncommitted bundled source: " + name)
    protocol = protocol_digest(files)
    if args.mode == "full":
        require_smoke_receipt(args.smoke_receipt, protocol)
    files.update(scoring_files(args.scorer_archive))
    files["reference.ipynb"] = args.reference_notebook.read_text()
    manifest = {"git_commit": revision, "mode": args.mode, "protocol_sha256": protocol,
                "reference_sha256": reference.REFERENCE_SHA256, "scorer_archive_sha256": SCORER_ARCHIVE_SHA256,
                "files": {name: sha256(text.encode()) for name, text in files.items()}}
    files["bundle_manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    metadata = {"id": f"{args.owner}/biohub-e038-development-{args.mode}",
                "title": "Biohub E038 Development | Frozen " + args.mode.title(),
                "code_file": "e038_development.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1",
                                    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
                                    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
                                    "hengck23/hengck23-cell-point-detector-demo"],
                "competition_sources": ["biohub-cell-tracking-during-development"], "kernel_sources": [],
                "model_sources": [], "docker_image": DOCKER, "machine_shape": "NvidiaTeslaT4"}
    intro = ("# E038 frozen development comparison\n\nPrivate compute fallback; training movies only. "
             "No submission is made. A smoke result is a technical check, not selection evidence.\n\n"
             "References: [Geometric Fusion](https://www.kaggle.com/code/amanatar/biohub-geometric-fusion), "
             "[v5 detector](https://www.kaggle.com/datasets/hengck23/hengck23-cell-point-detector-demo), "
             "[official scorer](https://github.com/royerlab/kaggle-cell-tracking-competition/tree/075fc5f). "
             "Original licenses and pinned source bytes are retained in the private runtime bundle.\n\n"
             f"Source commit: `{revision}`. Mode: `{args.mode}`. Protocol: `{protocol}`.\n")
    source = bootstrap_source(files)
    compile(source, "kaggle-bootstrap", "exec")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [{"cell_type": "markdown", "metadata": {}, "id": "protocol", "source": intro},
                          {"cell_type": "code", "metadata": {}, "id": "run", "source": source,
                           "outputs": [], "execution_count": None}]}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("kernel-metadata.json", metadata), ("e038_development.ipynb", notebook),
                        ("bundle_manifest.json", manifest)):
        (args.output_dir / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"id": metadata["id"], "is_private": True, "mode": args.mode, "git_commit": revision,
                      "bundled_files": len(files), "output_dir": str(args.output_dir)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--reference-notebook", type=Path, required=True)
    parser.add_argument("--scorer-archive", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
