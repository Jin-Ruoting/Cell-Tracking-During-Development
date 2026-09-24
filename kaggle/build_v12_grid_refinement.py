#!/usr/bin/env python3
"""Package the frozen E044 GPU experiment from pushed local sources only."""
from __future__ import annotations

import argparse
import base64
import json
import lzma
from pathlib import Path
import re
import subprocess

import build_kaggle_development as packaging
import check_v12_fixed_nodes as cpu
import v12_grid_refinement as policy


def build(args):
    root = Path(__file__).resolve().parents[1]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.owner):
        raise ValueError("Invalid Kaggle owner")
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=root, check=True, stdout=subprocess.DEVNULL)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != subprocess.check_output(["git", "rev-parse", "@{upstream}"], cwd=root, text=True).strip():
        raise ValueError("Push local source before packaging E044")
    files = packaging.source_closure(root, ["kaggle/run_v12_grid_refinement.py", "kaggle/check_v12_refinement.py", "tests/test_v12_grid_refinement.py"])
    for name, source in files.items():
        if source != subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root).decode():
            raise ValueError("Uncommitted E044 source: " + name)
    protocol = packaging.protocol_digest(files)
    config = policy.POLICY
    receipt = json.loads(args.cpu_receipt.read_text())
    checks = ("zero_residual_identity", "rounded_grid_sampling", "fractional_input_preserved", "physical_residual_units")
    if (receipt.get("passed") is not True or receipt.get("ground_truth_accessed") is not False
            or receipt.get("competition_test_accessed") is not False or receipt.get("quality_score") is not None
            or receipt.get("strict_model_load") is not True
            or not all(receipt.get("synthetic_contract", {}).get(k) is True for k in checks)
            or receipt["control"]["csv_sha256"] != cpu.CONTROL_SHA256
            or receipt["checkpoint_sha256"] != cpu.image_audit.checkpoint_audit.WEIGHT_SHA256
            or receipt["source_manifest"]["files"]["check_v12_refinement.py"] != packaging.sha256(files["kaggle/check_v12_refinement.py"].encode())):
        raise ValueError("Completed S230 CPU interface receipt required; it is not quality evidence")
    if args.mode == "smoke":
        names = list(packaging.reference.SMOKE_NAMES)
        control = {"kernel": cpu.CONTROL_KERNEL, "directory": "logs/e038-development",
                   "csv_sha256": cpu.CONTROL_SHA256, "source_git_commit": cpu.CONTROL_SOURCE,
                   "score": 0.9638391788805566, "control_regenerated_in_this_run": False}
    else:
        if args.full_control_dir is None or args.smoke_receipt is None:
            raise ValueError("Full E044 requires completed S213 control and same-protocol GPU smoke")
        directory = args.full_control_dir
        prior = json.loads((directory / "paired_development_receipt.json").read_text())
        frozen = json.loads((directory / "frozen_control.json").read_text())
        run = json.loads((directory / "run_manifest.json").read_text())
        cohort = json.loads((directory / "cohort.json").read_text())
        names = cohort["datasets"]
        if (prior.get("experiment") != "E039" or prior.get("mode") != "full"
                or prior.get("technical_check_passed") is not True or prior["control"] != frozen
                or prior["protocol_sha256"] != "d21433e05ba4ed54f6ca564244650927d5dd692e4fd8d0a2b593e385605d0ebd"
                or run["git_commit"] != "e2e689e21b7bbedf485e4855a62d4edc5cab79e7"
                or run["datasets"] != names or len(names) != 64 or names != sorted(set(names))
                or packaging.reference.stability.movie_names_sha256(names) != packaging.reference.CORPUS_SHA256
                or frozen["paired_control_frozen_before_candidate"] is not True):
            raise ValueError("Full cloud control provenance/coverage failed")
        smoke = json.loads(args.smoke_receipt.read_text())
        if (smoke.get("experiment") != "E044" or smoke.get("mode") != "smoke"
                or smoke.get("technical_check_passed") is not True or smoke["protocol_sha256"] != protocol
                or smoke["control"]["csv_sha256"] != cpu.CONTROL_SHA256
                or smoke["all_edges_ids_times_preserved"] is not True):
            raise ValueError("Full run requires the same frozen complete-movie GPU smoke")
        control = {"kernel": args.owner + "/biohub-e039-kaggle-paired-full", "directory": "logs/e039-development",
                   "source_git_commit": run["git_commit"], "csv_sha256": frozen["csv_sha256"],
                   "score": prior["groups"]["all"]["control"]["score"], "frozen_receipt": frozen,
                   "completion_receipt": prior, "control_regenerated_in_this_run": False}
    files.update(packaging.scoring_files(args.scorer_archive))
    files["cpu_receipt.json"] = args.cpu_receipt.read_text()
    manifest = {"experiment": "E044", "git_commit": revision, "mode": args.mode, "datasets": names,
                "protocol_sha256": protocol, "config": config, "control": control,
                "files": {n: packaging.sha256(s.encode()) for n, s in files.items()}}
    files["bundle_manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    payload = lzma.compress(json.dumps(files, sort_keys=True).encode(), preset=6)
    code = f'''import base64, concurrent.futures, hashlib, json, lzma, os, subprocess, sys, threading
from pathlib import Path
payload = base64.b85decode({base64.b85encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {packaging.sha256(payload)!r}
bundle = Path("/kaggle/working/e044-bundle")
bundle.mkdir(exist_ok=False)
for name, text in json.loads(lzma.decompress(payload)).items():
    target = (bundle / name).resolve()
    if bundle not in target.parents:
        raise ValueError("Invalid source path")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
logs = Path("/kaggle/working/logs/e044-launch")
logs.mkdir(parents=True, exist_ok=False)
env = {{**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "POLARS_MAX_THREADS": "4", "POLARS_PREFER_PKG": "32"}}
processes, lock = [], threading.Lock()
def run(label, command, gpu=None):
    current_env = dict(env)
    if gpu is not None:
        current_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with (logs / (label + ".log")).open("x") as output:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=bundle, env=current_env)
        with lock:
            processes.append(process)
        for line in process.stdout:
            print(label + ": " + line, end="", flush=True)
            output.write(line)
            output.flush()
        if process.wait() != 0:
            raise RuntimeError(label + " failed; inspect retained log")
runner = [sys.executable, str(bundle / "kaggle/run_v12_grid_refinement.py"), "--bundle", str(bundle), "--phase"]
run("semantic-tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_v12_grid_refinement.py", "-v"])
run("setup", runner + ["setup"])
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(run, "predict-" + str(i), runner + ["predict", "--shard", str(i)], i) for i in range(2)]
    try:
        for future in concurrent.futures.as_completed(futures):
            future.result()
    except BaseException:
        with lock:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
        raise
run("merge", runner + ["merge"])
run("score", runner + ["score"])
print(Path("/kaggle/working/logs/e044-grid-refinement/run_summary.md").read_text())
'''
    compile(code, "e044-gpu-bootstrap", "exec")
    metadata = {"id": f"{args.owner}/biohub-e044-grid-refinement-{args.mode}",
                "title": "Biohub E044 Grid Refinement " + args.mode.title(), "code_file": "e044.ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1", "hengck23/hengck23-cell-point-detector-demo"],
                "competition_sources": ["biohub-cell-tracking-during-development"],
                "kernel_sources": [control["kernel"]], "model_sources": [], "docker_image": packaging.DOCKER,
                "machine_shape": "NvidiaTeslaT4"}
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [{"cell_type": "markdown", "metadata": {}, "id": "protocol", "source":
                           "# E044 integer-anchor refinement\n\nPrivate paired development; frozen E029 control reused. "
                           "Node IDs, times and every edge stay fixed. Compare anchor-only and learned refinement against E029. No parameter search or submission.\n\n"
                           "Model: https://www.kaggle.com/code/hengck23/end2end-cell-linker-raw-edge-ja-0-9-no-ilp\n\n"
                           + f"Source: {revision}. Mode: {args.mode}. Protocol: {protocol}."},
                          {"cell_type": "code", "metadata": {}, "id": "run", "source": code,
                           "outputs": [], "execution_count": None}]}
    size = len((json.dumps(notebook, indent=2) + "\n").encode())
    if size >= 1_000_000:
        raise ValueError("Notebook exceeds Kaggle source limit")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("kernel-metadata.json", metadata), ("e044.ipynb", notebook), ("bundle_manifest.json", manifest)):
        (args.output_dir / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"id": metadata["id"], "git_commit": revision, "protocol": protocol, "notebook_bytes": size}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--cpu-receipt", type=Path, required=True)
    parser.add_argument("--scorer-archive", type=Path, required=True)
    parser.add_argument("--full-control-dir", type=Path)
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
