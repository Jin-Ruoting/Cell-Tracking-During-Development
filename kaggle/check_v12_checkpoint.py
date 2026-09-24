#!/usr/bin/env python3
"""Package a private Kaggle CPU-only compatibility audit of the public v12 model.

Local mode packages committed sources only. Runtime mode inspects checkpoint
metadata, optionally checks synthetic tensors, and never accesses competition data.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile
import zlib

import check_point_detector_runtime as reviewed

WORK = Path("/kaggle/working/logs/v12-cpu-compatibility")
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
WEIGHT_SHA256 = "ecd8869de9cf405c1a93a563b4810faad5fd2378f57341e18c4e6b5a87201552"


def package(owner: str, output: Path, *, metadata_only: bool = False):
    root = Path(__file__).resolve().parents[1]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", owner):
        raise ValueError("Invalid Kaggle owner")
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != subprocess.check_output(["git", "rev-parse", "@{upstream}"], cwd=root, text=True).strip():
        raise ValueError("Commit and push before cloud packaging")
    files = {}
    for path in (Path(__file__).resolve(), Path(reviewed.__file__).resolve()):
        source = path.read_text()
        if source != subprocess.check_output(["git", "show", f"HEAD:{path.relative_to(root)}"], cwd=root).decode():
            raise ValueError("Uncommitted compatibility-audit source")
        files[path.name] = source
    payload = zlib.compress(json.dumps(files).encode(), level=9)
    runtime_args = ["--metadata-only"] if metadata_only else []
    code = f'''import base64, hashlib, json, os, subprocess, sys, zlib
from pathlib import Path
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {hashlib.sha256(payload).hexdigest()!r}
root = Path("/kaggle/working/v12-audit-source")
root.mkdir(exist_ok=False)
for name, text in json.loads(zlib.decompress(payload)).items():
    if Path(name).name != name:
        raise ValueError("Invalid source filename")
    (root / name).write_text(text)
env = {{**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}}
subprocess.run([sys.executable, str(root / "check_v12_checkpoint.py"), "run", "--source-commit", {commit!r}] + {runtime_args!r}, env=env, check=True, timeout=600)
'''
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [{"cell_type": "markdown", "metadata": {}, "id": "scope", "source":
                           "# V12 CPU compatibility\n\nCheckpoint metadata audit"
                           + (" only. " if metadata_only else " and synthetic tensor check. ")
                           + "No images, labels, quality score or submission. "
                           "Reference: https://www.kaggle.com/datasets/hengck23/hengck23-cell-point-detector-demo\n\n"
                           + "Source commit: " + commit},
                          {"cell_type": "code", "metadata": {}, "id": "audit", "source": code,
                           "outputs": [], "execution_count": None}]}
    compile(code, "v12-cpu-bootstrap", "exec")
    metadata = {"id": owner + "/biohub-v12-cpu-compatibility", "title": "Biohub V12 CPU Compatibility",
                "code_file": "v12_cpu.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": ["hengck23/hengck23-cell-point-detector-demo"],
                "competition_sources": [], "kernel_sources": [], "model_sources": [], "docker_image": DOCKER}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("kernel-metadata.json", metadata), ("v12_cpu.ipynb", notebook)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"git_commit": commit, "id": metadata["id"], "is_private": True,
                      "enable_gpu": False, "metadata_only": metadata_only}, indent=2))


def load_checkpoint(root: Path, report: dict, *, cpu_only: bool = True):
    """Load only the pinned, reviewed checkpoint using restricted deserialization."""
    weight = root / "00000008.pth"
    if weight.stat().st_size != 43_076_221:
        raise ValueError("V12 checkpoint is not the complete publicly listed file")
    report["checkpoint_bytes"] = weight.stat().st_size
    report["checkpoint_sha256"] = hashlib.sha256(weight.read_bytes()).hexdigest()
    if report["checkpoint_sha256"] != WEIGHT_SHA256:
        raise ValueError("V12 checkpoint differs from the completed S212 acquisition")
    with zipfile.ZipFile(weight) as archive:
        if archive.testzip() is not None:
            raise ValueError("Checkpoint archive CRC failed")
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    report["torch_version"] = str(torch.__version__)
    external = torch.serialization.get_unsafe_globals_in_checkpoint(weight)
    report["checkpoint_extra_globals"] = external
    allowed = []
    for name in external:
        if name not in ("model_v12.DotDict", "loss_and_metric_v12.DotDict"):
            raise ValueError("Unreviewed checkpoint type: " + name)
        filename = name.rsplit(".", 1)[0] + ".py"
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != reviewed.PINNED[filename]:
            raise ValueError("Reviewed metadata dictionary source changed")
        # These reviewed classes only add attribute access to dict. Their
        # metadata is not used to configure the model. Loading it as the
        # built-in container avoids PyTorch's SETITEMS subclass restriction
        # without enabling arbitrary checkpoint code or altering tensors.
        allowed.append((dict, name))
    report["metadata_dotdict_loaded_as_builtin_dict"] = external
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
    return checkpoint


def run(commit: str, *, metadata_only: bool = False):
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("This runtime audit may execute only on Kaggle compute")
    WORK.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "git_commit": commit, "device": "cpu", "real_image_inference": False,
              "ground_truth_accessed": False, "tracking_run": False, "quality_score": None,
              "audit_mode": "checkpoint_metadata_only" if metadata_only else "synthetic_compatibility",
              "synthetic_input_shape": None if metadata_only else [1, 1, 64, 64, 64],
              "model_constructed": False, "synthetic_forward_executed": False,
              "weight_sha256_previously_pinned": True}
    started = time.monotonic()
    try:
        candidates = [Path("/kaggle/input/hengck23-cell-point-detector-demo"),
                      Path("/kaggle/input/datasets/hengck23/hengck23-cell-point-detector-demo")]
        roots = {p.resolve() for p in candidates if p.is_dir()}
        if len(roots) != 1:
            raise ValueError("Expected one mounted author dataset")
        root = roots.pop()
        checkpoint = load_checkpoint(root, report)
        import torch
        report["checkpoint_keys"] = sorted(checkpoint)
        # Export only plain JSON configuration/epoch metadata, never state tensors.
        # Fail on unsupported values or nonfinite numbers rather than stringify them.
        report["checkpoint_metadata"] = json.loads(json.dumps(
            {key: checkpoint[key] for key in ("CFG", "epoch")}, allow_nan=False))
        if metadata_only:
            report["passed"] = True
            return
        source = reviewed.load_reviewed(root, "model_v12")
        report["source_sha256"] = reviewed.PINNED["model_v12.py"]
        cfg = source.DotDict(
            unet_cfg=source.DotDict(channel=(64, 128, 256), dropout=0.1, gradient_checkpointing=True,
                                   node_peak_kernel=3, node_peak_threshold=0.2, max_detected_nodes=800),
            tx_cfg=source.DotDict(feat_dims=(64, 128, 256), hidden_dim=256, embed_dim=128,
                                 n_heads=4, n_layers=4, dropout=0.1))
        model = source.End2EndCellLinker(CFG=cfg).eval()
        report["model_constructed"] = True
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        report["strict_model_load"] = True
        with torch.inference_mode():
            out0, out1, links = model(torch.zeros(1, 1, 64, 64, 64), torch.zeros(1, 1, 64, 64, 64))
        report["synthetic_forward_executed"] = True
        for output in (out0, out1, links):
            for value in output.values():
                tensors = value if isinstance(value, list) else [value]
                for tensor in tensors:
                    if torch.is_tensor(tensor) and (tensor.device.type != "cpu" or not torch.isfinite(tensor).all()):
                        raise ValueError("Nonfinite or non-CPU output")
        n0, n1 = out0.refine_zyx.shape[1], out1.refine_zyx.shape[1]
        if tuple(links.edge_logit.shape) != (1, n0, n1) or not (1 <= n0 <= 800 and 1 <= n1 <= 800):
            raise ValueError("Linker output shape contract failed")
        report.update({"passed": True, "parameters": sum(p.numel() for p in model.parameters()),
                       "node_counts": [n0, n1], "edge_logit_shape": list(links.edge_logit.shape),
                       "refined_coordinates_outside_synthetic_volume": [int(((out.refine_zyx < 0) | (out.refine_zyx >= 64)).any(-1).sum()) for out in (out0, out1)]})
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (WORK / "runtime_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        (WORK / "run_summary.md").write_text("# V12 CPU compatibility audit\n\n```json\n" + json.dumps(report, indent=2) +
                                            "\n```\n\nMetadata/interface audit only; no tracking-quality evidence.\n")
        print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    package_parser = modes.add_parser("package")
    package_parser.add_argument("--owner", required=True)
    package_parser.add_argument("--output-dir", type=Path, required=True)
    package_parser.add_argument("--metadata-only", action="store_true")
    run_parser = modes.add_parser("run")
    run_parser.add_argument("--source-commit", required=True)
    run_parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.mode == "package":
        package(args.owner, args.output_dir, metadata_only=args.metadata_only)
    else:
        run(args.source_commit, metadata_only=args.metadata_only)


if __name__ == "__main__":
    main()
