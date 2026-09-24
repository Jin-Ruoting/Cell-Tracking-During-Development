#!/usr/bin/env python3
"""Bounded Kaggle CPU audit of four fixed real frames; no labels or scoring.

Package locally only after commit/push. The reviewed author postprocessing is
bundled privately from a checksum-pinned Notebook, never copied into Git.
"""
from __future__ import annotations

import argparse
import ast
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

import check_v12_checkpoint as checkpoint_audit

DEMO_SHA256 = "b0b2da4e467311a70723e14a8c33be716da54b62001a8a0d0536a36b328cd724"
NAMES = ("44b6_eb2880fc", "6bba_969618f6")
FUNCTIONS = ("cluster_predicted_nodes", "edge_top_cumulative_bidirectional", "select_top1_with_dst_correction")
WORK = Path("/kaggle/working/logs/v12-real-frames")


def reviewed_functions(path: Path) -> str:
    if hashlib.sha256(path.read_bytes()).hexdigest() != DEMO_SHA256:
        raise ValueError("Author Notebook differs from the reviewed version")
    notebook = json.loads(path.read_text())
    source = "".join(notebook["cells"][2]["source"])
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    if tuple(n.name for n in nodes) != FUNCTIONS:
        raise ValueError("Reviewed postprocessing function anchors changed")
    tree = ast.Module(body=nodes, type_ignores=[])
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def package(owner: str, output: Path, demo: Path):
    root = Path(__file__).resolve().parents[1]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", owner):
        raise ValueError("Invalid Kaggle owner")
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=root, check=True, stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != subprocess.check_output(["git", "rev-parse", "@{upstream}"], cwd=root, text=True).strip():
        raise ValueError("Commit and push before cloud packaging")
    files = {}
    for module in (sys.modules[__name__], checkpoint_audit, checkpoint_audit.reviewed):
        path = Path(module.__file__).resolve()
        source = path.read_text()
        if source != subprocess.check_output(["git", "show", f"HEAD:{path.relative_to(root)}"], cwd=root).decode():
            raise ValueError("Uncommitted source: " + path.name)
        files[path.name] = source
    files["reviewed_postprocessing.py"] = reviewed_functions(demo)
    manifest = {"git_commit": commit, "author_notebook_sha256": DEMO_SHA256,
                "files": {n: hashlib.sha256(s.encode()).hexdigest() for n, s in files.items()}}
    files["bundle_manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    payload = zlib.compress(json.dumps(files).encode(), level=9)
    code = f'''import base64, hashlib, json, os, subprocess, sys, zlib
from pathlib import Path
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {hashlib.sha256(payload).hexdigest()!r}
root = Path("/kaggle/working/v12-real-source")
root.mkdir(exist_ok=False)
for name, text in json.loads(zlib.decompress(payload)).items():
    if Path(name).name != name:
        raise ValueError("Invalid source filename")
    (root / name).write_text(text)
env = {{**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}}
subprocess.run([sys.executable, str(root / "check_v12_real_frames.py"), "run", "--bundle", str(root)], env=env, check=True, timeout=1200)
'''
    compile(code, "v12-real-bootstrap", "exec")
    metadata = {"id": owner + "/biohub-v12-real-frame-audit", "title": "Biohub V12 Real Frame Audit",
                "code_file": "v12_real.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": ["hengck23/hengck23-cell-point-detector-demo",
                                    "pilkwang/biohub-tracking-support-pack-50ep-v1"],
                "competition_sources": ["biohub-cell-tracking-during-development"],
                "kernel_sources": [], "model_sources": [], "docker_image": checkpoint_audit.DOCKER}
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [{"cell_type": "markdown", "metadata": {}, "id": "scope", "source":
                           "# V12 real-frame CPU audit\n\nFrames 0 and 1 of two frozen development movies. "
                           "No labels, quality score, full movie or submission. CPU FP32 differs from author GPU FP16.\n\n"
                           "Author: https://www.kaggle.com/code/hengck23/end2end-cell-linker-raw-edge-ja-0-9-no-ilp\n\n"
                           + "Source commit: " + commit},
                          {"cell_type": "code", "metadata": {}, "id": "audit", "source": code,
                           "outputs": [], "execution_count": None}]}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("kernel-metadata.json", metadata), ("v12_real.ipynb", notebook), ("bundle_manifest.json", manifest)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"git_commit": commit, "id": metadata["id"], "is_private": True, "enable_gpu": False}, indent=2))


def mounted(owner: str, slug: str, competition=False) -> Path:
    choices = [Path("/kaggle/input") / slug,
               Path("/kaggle/input/competitions" if competition else "/kaggle/input/datasets") /
               (slug if competition else owner + "/" + slug)]
    roots = {p.resolve() for p in choices if p.is_dir()}
    if len(roots) != 1:
        raise ValueError("Expected exactly one mounted source for " + slug)
    return roots.pop()


def run(bundle: Path):
    if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Real-image model audits may execute only on Kaggle compute")
    WORK.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "device": "cpu", "dtype": "float32", "ground_truth_accessed": False,
              "competition_test_accessed": False, "quality_score": None, "submission_created": False,
              "full_movie_tracking": False, "datasets": list(NAMES), "frames_per_movie": [0, 1],
              "downsample": [1, 4, 4], "quantiles": ["0.001", "0.999"], "clip": "minimum_zero_only",
              "author_postprocessing": {"cluster_radius_grid_voxels": 3.0, "no_link_logit": 0, "cumulative": 0.99},
              "movies": {}}
    started = time.monotonic()
    try:
        manifest = json.loads((bundle / "bundle_manifest.json").read_text())
        for name, digest in manifest["files"].items():
            if Path(name).name != name or hashlib.sha256((bundle / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Bundled source mismatch: " + name)
        if manifest["author_notebook_sha256"] != DEMO_SHA256:
            raise ValueError("Unexpected author source")
        report.update({"git_commit": manifest["git_commit"], "source_manifest": manifest})
        support = mounted("pilkwang", "biohub-tracking-support-pack-50ep-v1")
        runtime = WORK / "runtime"
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-deps",
                        "--find-links", str(support / "wheels"), "--target", str(runtime),
                        "zarr==3.2.1", "numcodecs==0.15.1", "donfig==0.8.1.post1"], check=True)
        sys.path.insert(0, str(runtime))
        import numpy as np
        import pandas as pd
        import torch
        import zarr

        author = mounted("hengck23", "hengck23-cell-point-detector-demo")
        checkpoint = checkpoint_audit.load_checkpoint(author, report)
        source = checkpoint_audit.reviewed.load_reviewed(author, "model_v12")
        cfg = source.DotDict(**{k: source.DotDict(**checkpoint["CFG"][k]) for k in ("unet_cfg", "tx_cfg")})
        if (cfg.unet_cfg.max_detected_nodes != 1024 or cfg.unet_cfg.node_peak_threshold != 0.2
                or cfg.unet_cfg.node_peak_kernel != 3 or checkpoint["epoch"] != 8):
            raise ValueError("Checkpoint configuration differs from S214")
        report["inference_cfg"] = cfg
        model = source.End2EndCellLinker(CFG=cfg).eval()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        report["strict_model_load"] = True
        scope = {"torch": torch, "np": np, "pd": pd}
        exec(compile((bundle / "reviewed_postprocessing.py").read_text(), "reviewed_postprocessing.py", "exec"), scope)
        data = mounted("", "biohub-cell-tracking-during-development", competition=True)
        for name in NAMES:
            start = time.monotonic()
            image_path = data / "train" / (name + ".zarr")
            group = zarr.open_group(str(image_path), mode="r")
            image = group["0"]
            attrs = dict(group.attrs)
            scale = (1.625, 0.40625, 0.40625)  # Same explicit fallback as the reviewed support loader.
            scale_source = "reviewed_support_loader_default"
            if "multiscales" in attrs:
                transform = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
                if transform["type"] != "scale":
                    raise ValueError("Unexpected coordinate transform")
                scale = tuple(transform["scale"][-3:])
                scale_source = "image_multiscales_metadata"
            if (scale != (1.625, 0.40625, 0.40625)
                    or tuple(image.shape) != (100, 64, 256, 256) or image.dtype != np.uint16):
                raise ValueError("Unexpected image dimensions, voxel scale or dtype")
            quantiles = attrs["image_statistics"]["quantiles"]
            low, high = (float(quantiles[k]) for k in ("0.001", "0.999"))
            if not np.isfinite([low, high]).all() or high <= low:
                raise ValueError("Invalid image-only quantiles")
            # Same strided selection and normalization as the author; no GEFF open.
            volume = image[0:2, ::1, ::4, ::4].astype(np.float32)
            volume = np.clip((volume - low) / (high - low + 1e-6), 0.0, None).astype(np.float32)
            if not np.isfinite(volume).all():
                raise ValueError("Nonfinite normalized image")
            with torch.inference_mode():
                out = model.unet(torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(1))
                before = [int(v) for v in out.count]
                out = scope["cluster_predicted_nodes"](out, radius=3.0)
                links = model.linker([f[:-1] for f in out.node_feature], [f[1:] for f in out.node_feature],
                                     out.refine_zyx[:-1], out.refine_zyx[1:], out.mask[:-1], out.mask[1:], (64, 64, 64))
                for value in (out.refine_zyx, out.refine_logit, links.edge_logit):
                    if value.device.type != "cpu" or not torch.isfinite(value).all():
                        raise ValueError("Nonfinite or non-CPU prediction")
                rows = scope["edge_top_cumulative_bidirectional"](links.edge_logit, out.mask[:-1], out.mask[1:],
                                                                  t_start=0, no_link_logit=0, cumulative=0.99)
                candidates = pd.DataFrame(rows)
                selected = scope["select_top1_with_dst_correction"](candidates)
            if selected.duplicated(["t", "src_id"]).any() or selected.duplicated(["t", "dst_id"]).any():
                raise ValueError("Source/destination conflict remains")
            for edge in selected.itertuples(index=False):
                if edge.t != 0 or not bool(out.mask[0, edge.src_id]) or not bool(out.mask[1, edge.dst_id]):
                    raise ValueError("Invalid selected edge endpoint")
            coords = [out.refine_zyx[b, out.mask[b]].numpy() for b in (0, 1)]
            outside = [int(((v < 0) | (v >= 64)).any(axis=1).sum()) for v in coords]
            record = {"frames": [0, 1], "quantile_low": low, "quantile_high": high,
                      "scale": list(scale), "scale_source": scale_source,
                      "input_range": [float(volume.min()), float(volume.max())], "nodes_before_cluster": before,
                      "nodes_after_cluster": [int(v) for v in out.count], "selected_edges": len(selected),
                      "top1_no_link": int(((candidates["rank"] == 0) & (candidates["dst_id"] == -1)).sum()),
                      "refined_coordinates_outside_volume": outside, "coordinates_in_bounds": not any(outside),
                      "coordinate_ranges": [[v.min(axis=0).tolist(), v.max(axis=0).tolist()] for v in coords],
                      "edge_logit_shape": list(links.edge_logit.shape), "seconds": time.monotonic() - start}
            report["movies"][name] = record
            print(json.dumps({"dataset": name, **record}), flush=True)
        report["passed"] = True
        report["all_coordinates_in_bounds"] = all(r["coordinates_in_bounds"] for r in report["movies"].values())
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (WORK / "runtime_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        (WORK / "run_summary.md").write_text("# V12 fixed real-frame CPU audit\n\n```json\n" + json.dumps(report, indent=2) +
                                            "\n```\n\nFour real frames only; no labels, official score or quality promotion.\n")
        print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    package_parser = modes.add_parser("package")
    package_parser.add_argument("--owner", required=True)
    package_parser.add_argument("--output-dir", type=Path, required=True)
    package_parser.add_argument("--reference-notebook", type=Path, required=True)
    run_parser = modes.add_parser("run")
    run_parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "package":
        package(args.owner, args.output_dir, args.reference_notebook)
    else:
        run(args.bundle)


if __name__ == "__main__":
    main()
