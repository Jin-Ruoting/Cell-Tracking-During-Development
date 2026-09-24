#!/usr/bin/env python3
"""E040 fixed-node CPU plumbing check on two prerecorded frame pairs.

Runs only on Kaggle. Uses the completed independent cloud E029 control, keeps
every node coordinate, and never evaluates labels or selects model parameters.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import zlib

import check_v12_real_frames as image_audit

CONTROL_KERNEL = "buaaauto/biohub-e038-development-frozen-smoke"
CONTROL_SHA256 = "4cae8278063f1ec44932750f9585b45805efb944540ccea670e2d3fa5687d68f"
CONTROL_SOURCE = "4f13ce072428c89a5d2f199e4dd731190025be1a"
WORK = Path("/kaggle/working/logs/e040-fixed-node-cpu")


def package(owner: str, output: Path):
    root = Path(__file__).resolve().parents[1]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", owner):
        raise ValueError("Invalid Kaggle owner")
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != subprocess.check_output(["git", "rev-parse", "@{upstream}"], cwd=root, text=True).strip():
        raise ValueError("Commit and push before cloud packaging")
    files = {}
    for relative in ("kaggle/check_v12_fixed_nodes.py", "kaggle/check_v12_real_frames.py",
                     "kaggle/check_v12_checkpoint.py", "kaggle/check_point_detector_runtime.py",
                     "kaggle/v12_edge_swap.py", "tests/test_v12_edge_swap.py"):
        path = root / relative
        source = path.read_text()
        if source != subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=root).decode():
            raise ValueError("Uncommitted source: " + relative)
        files[path.name] = source
    manifest = {"git_commit": commit, "experiment": "E040 fixed-node CPU technical check",
                "files": {n: hashlib.sha256(s.encode()).hexdigest() for n, s in files.items()}}
    files["bundle_manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    payload = zlib.compress(json.dumps(files).encode(), level=9)
    code = f'''import base64, hashlib, json, os, subprocess, sys, zlib
from pathlib import Path
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {hashlib.sha256(payload).hexdigest()!r}
root = Path("/kaggle/working/e040-cpu-source")
root.mkdir(exist_ok=False)
for name, text in json.loads(zlib.decompress(payload)).items():
    if Path(name).name != name:
        raise ValueError("Invalid source filename")
    (root / name).write_text(text)
env = {{**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}}
subprocess.run([sys.executable, "-m", "unittest", "test_v12_edge_swap", "-v"], cwd=root, env=env, check=True, timeout=120)
subprocess.run([sys.executable, str(root / "check_v12_fixed_nodes.py"), "run", "--bundle", str(root)], env=env, check=True, timeout=1200)
'''
    compile(code, "e040-cpu-bootstrap", "exec")
    metadata = {"id": owner + "/biohub-e040-fixed-node-cpu", "title": "Biohub E040 Fixed Node CPU",
                "code_file": "e040_cpu.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": ["hengck23/hengck23-cell-point-detector-demo",
                                    "pilkwang/biohub-tracking-support-pack-50ep-v1"],
                "competition_sources": ["biohub-cell-tracking-during-development"],
                "kernel_sources": [CONTROL_KERNEL], "model_sources": [], "docker_image": image_audit.checkpoint_audit.DOCKER}
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [{"cell_type": "markdown", "metadata": {}, "id": "scope", "source":
                           "# E040 fixed-node CPU check\n\nFixed E029 control nodes, frames 0 and 1 of two movies. "
                           "Only sample v12 features and check reciprocal edge swaps. No labels, quality score or submission.\n\n"
                           "Model: https://www.kaggle.com/code/hengck23/end2end-cell-linker-raw-edge-ja-0-9-no-ilp\n\n"
                           + "Source commit: " + commit},
                          {"cell_type": "code", "metadata": {}, "id": "audit", "source": code,
                           "outputs": [], "execution_count": None}]}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("kernel-metadata.json", metadata), ("e040_cpu.ipynb", notebook), ("bundle_manifest.json", manifest)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"git_commit": commit, "id": metadata["id"], "is_private": True, "enable_gpu": False}, indent=2))


def frozen_control() -> tuple[Path, dict]:
    owner, slug = CONTROL_KERNEL.split("/")
    roots = {p.resolve() for p in (Path("/kaggle/input") / slug,
                                  Path("/kaggle/input/notebooks") / owner / slug) if p.is_dir()}
    if len(roots) != 1:
        raise ValueError("Expected one mounted completed independent control Notebook")
    work = roots.pop() / "logs/e038-development"
    path = work / "control/control.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != CONTROL_SHA256:
        raise ValueError("Independent cloud control bytes changed")
    run = json.loads((work / "run_manifest.json").read_text())
    diagnostic = json.loads((work / "control_diagnostic.json").read_text())
    repeated = diagnostic["cloud_repeatability"]
    if (run["git_commit"] != CONTROL_SOURCE or run["datasets"] != list(image_audit.NAMES)
            or run["control_prediction_complete"] is not True or run["control_predicted_in_this_run"] is not True
            or run["predictions_read_ground_truth"] is not False
            or repeated["csv_bytes_reproduced"] is not True or repeated["score_reproduced"] is not True
            or diagnostic["control_parity"]["actual_csv_sha256"] != CONTROL_SHA256):
        raise ValueError("Independent cloud control provenance differs from S210")
    return path, {"source_kernel": CONTROL_KERNEL, "source_git_commit": CONTROL_SOURCE,
                  "csv_sha256": CONTROL_SHA256, "reused_completed_control": True,
                  "control_regenerated_in_this_run": False, "historical_server_parity": False}


def read_pair(path: Path, name: str):
    """Read original fields; only two frame images will be inferred."""
    nodes, edges = {}, []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != name:
                continue
            if row["row_type"] == "node":
                node = int(row["node_id"])
                point = [float(row[k]) for k in ("t", "z", "y", "x")]
                if node in nodes or point[0] != int(point[0]):
                    raise ValueError("Duplicate node or noninteger frame")
                nodes[node] = point
            elif row["row_type"] == "edge":
                edges.append((int(row["source_id"]), int(row["target_id"])))
            else:
                raise ValueError("Invalid control row type")
    if not nodes or len(edges) != len(set(edges)):
        raise ValueError("Incomplete/duplicate control graph")
    frame_edges = []
    for a, b in edges:
        if a not in nodes or b not in nodes or nodes[b][0] != nodes[a][0] + 1:
            raise ValueError("Dangling or nonconsecutive control edge")
        if nodes[a][0] == 0:
            frame_edges.append((a, b))
    ids = [sorted(n for n, pt in nodes.items() if pt[0] == t) for t in (0, 1)]
    if not all(0 < len(v) <= 1024 for v in ids):
        raise ValueError("Fixed-node count outside the reviewed model's diagnostic range; never subsample")
    return ids, [[nodes[n][1:] for n in row] for row in ids], frame_edges


def run(bundle: Path):
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Fixed-node image inference may execute only on Kaggle compute")
    WORK.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "experiment": "E040", "mode": "fixed_node_cpu_technical",
              "device": "cpu", "dtype": "float32", "frames_per_movie": [0, 1],
              "ground_truth_accessed": False, "competition_test_accessed": False,
              "quality_score": None, "submission_created": False, "full_movie_tracking": False,
              "node_detector_and_refinement_heads_executed": False, "movies": {}}
    started = time.monotonic()
    try:
        manifest = json.loads((bundle / "bundle_manifest.json").read_text())
        for name, digest in manifest["files"].items():
            if Path(name).name != name or hashlib.sha256((bundle / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Bundled source mismatch: " + name)
        report.update(git_commit=manifest["git_commit"], source_manifest=manifest)
        control_path, report["control"] = frozen_control()
        support = image_audit.mounted("pilkwang", "biohub-tracking-support-pack-50ep-v1")
        runtime = WORK / "runtime"
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-deps",
                        "--find-links", str(support / "wheels"), "--target", str(runtime),
                        "zarr==3.2.1", "numcodecs==0.15.1", "donfig==0.8.1.post1"], check=True)
        sys.path.insert(0, str(runtime))
        import numpy as np
        import torch
        import zarr
        import v12_edge_swap as swap

        report["config"] = swap.CONFIG
        author = image_audit.mounted("hengck23", "hengck23-cell-point-detector-demo")
        checkpoint = image_audit.checkpoint_audit.load_checkpoint(author, report)
        source = image_audit.checkpoint_audit.reviewed.load_reviewed(author, "model_v12")
        cfg = source.DotDict(**{k: source.DotDict(**checkpoint["CFG"][k]) for k in ("unet_cfg", "tx_cfg")})
        model = source.End2EndCellLinker(CFG=cfg).eval()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        report["strict_model_load"] = True
        data = image_audit.mounted("", "biohub-cell-tracking-during-development", competition=True)
        for name in image_audit.NAMES:
            start = time.monotonic()
            ids, points, pairs = read_pair(control_path, name)
            original = json.dumps({"ids": ids, "points": points, "pairs": pairs})
            group = zarr.open_group(str(data / "train" / (name + ".zarr")), mode="r")
            image, attrs = group["0"], dict(group.attrs)
            transform = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
            if (tuple(image.shape) != (100, 64, 256, 256) or image.dtype != np.uint16
                    or transform["type"] != "scale" or tuple(transform["scale"][-3:]) != tuple(swap.SCALE)):
                raise ValueError("Unexpected image format or physical scale")
            low, high = (float(attrs["image_statistics"]["quantiles"][k]) for k in ("0.001", "0.999"))
            if not np.isfinite([low, high]).all() or high <= low:
                raise ValueError("Invalid image-only quantiles")
            features, coordinates, masks = [], [], []
            with torch.inference_mode():
                for t in (0, 1):
                    volume = image[t, ::1, ::4, ::4].astype(np.float32)
                    volume = np.clip((volume - low) / (high - low + 1e-6), 0.0, None).astype(np.float32)
                    if not np.isfinite(volume).all():
                        raise ValueError("Nonfinite normalized image")
                    coord = np.asarray(points[t], dtype=np.float32) / np.asarray([1, 4, 4], dtype=np.float32)
                    if not np.isfinite(coord).all() or (coord < 0).any() or (coord >= 64).any():
                        raise ValueError("Fixed E029 coordinate invalid; no clipping permitted")
                    coord = torch.from_numpy(coord)[None]
                    mask = torch.ones((1, len(ids[t])), dtype=torch.bool)
                    layers, _ = model.unet.make_feature(torch.from_numpy(np.ascontiguousarray(volume))[None, None])
                    pyramid = [layers[3], layers[4], layers[2]]  # Author UNet.forward uses [d0,d1,e2].
                    sampled = source.sample_pyr_feature_at_zyx(pyramid, coord, (64, 64, 64), mask)
                    if [v.shape[-1] for v in sampled] != [64, 128, 256] or any(not torch.isfinite(v).all() for v in sampled):
                        raise ValueError("Invalid fixed-node multiscale features")
                    features.append(sampled)
                    coordinates.append(coord)
                    masks.append(mask)
                    del layers, pyramid
                links = model.linker(features[0], features[1], coordinates[0], coordinates[1], masks[0], masks[1], (64, 64, 64))
                if links.edge_logit.shape != (1, len(ids[0]), len(ids[1])):
                    raise ValueError("Incomplete fixed-node logit matrix")
                logits = links.edge_logit[0].numpy()
            changed, decisions, counts = swap.swap_frame(ids[0], ids[1], points[0], points[1], pairs, logits)
            if json.dumps({"ids": ids, "points": points, "pairs": pairs}) != original:
                raise ValueError("Frozen original graph objects mutated")
            report["movies"][name] = {"node_counts": list(map(len, ids)), "logit_shape": list(logits.shape),
                                      "logit_range": [float(logits.min()), float(logits.max())],
                                      "counts": counts, "decisions": decisions,
                                      "original_objects_unchanged": True, "seconds": time.monotonic() - start}
            print(json.dumps({"dataset": name, **report["movies"][name]}), flush=True)
        frozen_control()
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (WORK / "runtime_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        (WORK / "run_summary.md").write_text("# E040 fixed-node CPU check\n\n```json\n" + json.dumps(report, indent=2) +
                                            "\n```\n\nFour frames only; no label score, full graph or promotion.\n")
        print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    package_parser = modes.add_parser("package")
    package_parser.add_argument("--owner", required=True)
    package_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser = modes.add_parser("run")
    run_parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "package":
        package(args.owner, args.output_dir)
    else:
        run(args.bundle)


if __name__ == "__main__":
    main()
