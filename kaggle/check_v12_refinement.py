#!/usr/bin/env python3
"""Image-only CPU audit of the published v12 refiner on fixed E029 nodes."""
from __future__ import annotations

import argparse
import base64
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

import check_v12_fixed_nodes as fixed

image_audit = fixed.image_audit
WORK = Path("/kaggle/working/logs/v12-fixed-refinement")
SOURCES = ("check_v12_refinement.py", "check_v12_fixed_nodes.py", "check_v12_real_frames.py",
           "check_v12_checkpoint.py", "check_point_detector_runtime.py", "v12_edge_swap.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(owner, output):
    root = Path(__file__).resolve().parents[1]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", owner):
        raise ValueError("Invalid owner")
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=root, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != subprocess.check_output(["git", "rev-parse", "@{upstream}"], cwd=root, text=True).strip():
        raise ValueError("Commit and push before packaging")
    files = {}
    for name in SOURCES:
        relative = "kaggle/" + name
        source = (root / relative).read_text()
        if source != subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=root).decode():
            raise ValueError("Source differs from committed version")
        files[name] = source
    manifest = {"git_commit": commit, "audit": "fixed E029 node refinement interface",
                "files": {n: hashlib.sha256(s.encode()).hexdigest() for n, s in files.items()}}
    files["bundle_manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    payload = zlib.compress(json.dumps(files).encode(), level=9)
    code = f'''import base64, hashlib, json, os, subprocess, sys, zlib
from pathlib import Path
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {hashlib.sha256(payload).hexdigest()!r}
root = Path("/kaggle/working/v12-refinement-source")
root.mkdir(exist_ok=False)
for name, text in json.loads(zlib.decompress(payload)).items():
    if Path(name).name != name:
        raise ValueError("Invalid bundled filename")
    (root / name).write_text(text)
env = {{**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}}
subprocess.run([sys.executable, str(root / "check_v12_refinement.py"), "run", "--bundle", str(root)], env=env, check=True, timeout=1200)
'''
    compile(code, "v12-refinement-bootstrap", "exec")
    metadata = {"id": owner + "/biohub-v12-fixed-refinement-cpu", "title": "Biohub V12 Fixed Refinement CPU",
                "code_file": "refinement.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": ["hengck23/hengck23-cell-point-detector-demo", "pilkwang/biohub-tracking-support-pack-50ep-v1"],
                "competition_sources": ["biohub-cell-tracking-during-development"],
                "kernel_sources": [fixed.CONTROL_KERNEL], "model_sources": [],
                "docker_image": image_audit.checkpoint_audit.DOCKER}
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [{"cell_type": "markdown", "metadata": {}, "id": "scope", "source":
                           "# Fixed-node v12 refinement interface\n\nFour training-image frames on CPU. "
                           "No labels, graph export, parameter selection or quality score.\n\n"
                           "Reference: https://www.kaggle.com/code/hengck23/end2end-cell-linker-raw-edge-ja-0-9-no-ilp\n\n"
                           + "Source commit: " + commit},
                          {"cell_type": "code", "metadata": {}, "id": "audit", "source": code,
                           "outputs": [], "execution_count": None}]}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("kernel-metadata.json", metadata), ("refinement.ipynb", notebook), ("bundle_manifest.json", manifest)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"git_commit": commit, "id": metadata["id"], "gpu": False, "private": True}, indent=2))


def check_refinement_contract(source, torch):
    """Check nearest-grid sampling and residual units with known spatial fields."""
    coords = torch.tensor([[[1.25, 2.25, 3.25], [2.75, 2.75, 2.75]]])
    logits = torch.zeros((1, 2))
    field = torch.zeros((1, 4, 8, 8, 8))
    refined, updated = source.refine_node_peak(coords, logits, field)
    if not torch.equal(refined, coords) or not torch.equal(updated, logits):
        raise ValueError("Zero field is not an identity")
    field[:, 0] = torch.arange(8).view(8, 1, 1)
    refined, _ = source.refine_node_peak(coords, logits, field)
    expected = coords.clone()
    expected[..., 0] += torch.tensor([[1.0, 3.0]])
    if not torch.equal(refined, expected):
        raise ValueError("Author nearest-grid sampling contract changed")
    field.zero_()
    field[:, :3] = torch.tensor([0.125, -0.25, 0.375]).view(1, 3, 1, 1, 1)
    refined, _ = source.refine_node_peak(coords, logits, field)
    movement = (refined - coords) * torch.tensor([1.0, 4.0, 4.0]) * torch.tensor([1.625, 0.40625, 0.40625])
    expected = torch.tensor([0.203125, -0.40625, 0.609375]).expand_as(movement)
    if not torch.equal(movement, expected):
        raise ValueError("Raw voxel to physical displacement conversion changed")
    return {"zero_residual_identity": True, "rounded_grid_sampling": True,
            "fractional_input_preserved": True, "physical_residual_units": True}


def run(bundle):
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Real-image audit executes only on Kaggle")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    WORK.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "device": "cpu", "dtype": "float32", "ground_truth_accessed": False,
              "competition_test_accessed": False, "quality_score": None, "submission_created": False,
              "full_movie_tracking": False, "frames_per_movie": [0, 1], "movies": {},
              "detector_and_linker_heads_executed": False, "output_coordinates_clipped": False}
    started = time.monotonic()
    try:
        manifest = json.loads((bundle / "bundle_manifest.json").read_text())
        if set(manifest["files"]) != set(SOURCES):
            raise ValueError("Unexpected source closure")
        for name, digest in manifest["files"].items():
            if Path(name).name != name or sha(bundle / name) != digest:
                raise ValueError("Bundled source checksum mismatch")
        report.update(git_commit=manifest["git_commit"], source_manifest=manifest)
        control_path, report["control"] = fixed.frozen_control()
        support = image_audit.mounted("pilkwang", "biohub-tracking-support-pack-50ep-v1")
        runtime = WORK / "runtime"
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-deps",
                        "--find-links", str(support / "wheels"), "--target", str(runtime),
                        "zarr==3.2.1", "numcodecs==0.15.1", "donfig==0.8.1.post1"], check=True)
        sys.path.insert(0, str(runtime))
        import numpy as np
        import torch
        import zarr

        author = image_audit.mounted("hengck23", "hengck23-cell-point-detector-demo")
        checkpoint = image_audit.checkpoint_audit.load_checkpoint(author, report)
        source = image_audit.checkpoint_audit.reviewed.load_reviewed(author, "model_v12")
        report["source_sha256"] = image_audit.checkpoint_audit.reviewed.PINNED["model_v12.py"]
        report["synthetic_contract"] = check_refinement_contract(source, torch)
        cfg = source.DotDict(**{k: source.DotDict(**checkpoint["CFG"][k]) for k in ("unet_cfg", "tx_cfg")})
        model = source.End2EndCellLinker(CFG=cfg).eval()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        if any(p.device.type != "cpu" for p in model.parameters()):
            raise ValueError("CPU-only audit required")
        report["strict_model_load"] = True
        data = image_audit.mounted("", "biohub-cell-tracking-during-development", competition=True)
        scale = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float64)
        downsample = np.asarray([1, 4, 4], dtype=np.float64)
        for name in image_audit.NAMES:
            movie_start = time.monotonic()
            ids, points, pairs = fixed.read_pair(control_path, name)
            original = json.dumps({"ids": ids, "points": points, "pairs": pairs})
            group = zarr.open_group(str(data / "train" / (name + ".zarr")), mode="r")
            image, attrs = group["0"], dict(group.attrs)
            transform = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
            if (tuple(image.shape) != (100, 64, 256, 256) or image.dtype != np.uint16
                    or transform["type"] != "scale" or tuple(transform["scale"][-3:]) != tuple(scale)):
                raise ValueError("Unexpected image shape or physical scale")
            low, high = [float(attrs["image_statistics"]["quantiles"][k]) for k in ("0.001", "0.999")]
            if not np.isfinite([low, high]).all() or high <= low:
                raise ValueError("Invalid image quantiles")
            records = []
            with torch.inference_mode():
                for t in (0, 1):
                    volume = image[t, ::1, ::4, ::4].astype(np.float32)
                    volume = np.clip((volume - low) / (high - low + 1e-6), 0.0, None).astype(np.float32)
                    raw_coords = np.asarray(points[t], dtype=np.float64)
                    coord = torch.from_numpy((raw_coords / downsample).astype(np.float32))[None]
                    if (not np.isfinite(volume).all() or not torch.isfinite(coord).all()
                            or (coord < 0).any() or (coord >= 64).any()):
                        raise ValueError("Invalid image or original coordinate")
                    layers, feature = model.unet.make_feature(torch.from_numpy(np.ascontiguousarray(volume))[None, None])
                    field = model.unet.refine_head(feature)
                    refined, delta_logit = source.refine_node_peak(coord, torch.zeros((1, len(ids[t]))), field)
                    if (tuple(field.shape) != (1, 4, 64, 64, 64) or refined.shape != coord.shape
                            or not torch.isfinite(field).all() or not torch.isfinite(refined).all()
                            or not torch.isfinite(delta_logit).all() or refined.device.type != "cpu"):
                        raise ValueError("Invalid refinement field or output")
                    delta_raw = (refined - coord)[0].numpy().astype(np.float64) * downsample
                    proposed = raw_coords + delta_raw
                    outside = np.any((proposed < 0) | (proposed >= np.asarray(image.shape[1:])), axis=1)
                    displacement = np.linalg.norm(delta_raw * scale, axis=1)
                    fractional = np.any(np.abs(coord[0].numpy() - coord[0].numpy().round()) > 1e-6, axis=1)
                    records.append({"frame": t, "nodes": len(ids[t]), "fractional_input_nodes": int(fractional.sum()),
                                    "out_of_bounds": int(outside.sum()), "coordinates_in_bounds": not bool(outside.any()),
                                    "displacement_um_quantiles": dict(zip(["min", "q25", "median", "q75", "q95", "max"],
                                                                         map(float, np.quantile(displacement, [0, .25, .5, .75, .95, 1])))),
                                    "raw_coordinate_min": proposed.min(axis=0).tolist(), "raw_coordinate_max": proposed.max(axis=0).tolist(),
                                    "refinement_sha256": hashlib.sha256(field.numpy().tobytes()).hexdigest()})
                    del layers, feature, field, refined, delta_logit
            if json.dumps({"ids": ids, "points": points, "pairs": pairs}) != original:
                raise ValueError("Original graph objects changed")
            report["movies"][name] = {"frames": records, "original_objects_unchanged": True,
                                      "seconds": time.monotonic() - movie_start}
            print(json.dumps({"dataset": name, **report["movies"][name]}), flush=True)
        fixed.frozen_control()
        report["all_coordinates_in_bounds"] = all(r["coordinates_in_bounds"] for m in report["movies"].values() for r in m["frames"])
        report["passed"] = True  # Interface completion; bounds are independently reported.
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (WORK / "runtime_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        (WORK / "run_summary.md").write_text("# V12 fixed-node refinement CPU audit\n\n```json\n" + json.dumps(report, indent=2)
                                            + "\n```\n\nFour image frames only; no graph export, quality score or promotion.\n")
        print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("package")
    p.add_argument("--owner", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "package":
        package(args.owner, args.output_dir)
    else:
        run(args.bundle)


if __name__ == "__main__":
    main()
