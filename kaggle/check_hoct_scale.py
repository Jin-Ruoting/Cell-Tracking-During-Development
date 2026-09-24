#!/usr/bin/env python3
"""Package/run a data-free Kaggle CPU audit of HOCT physical scaling and v1 assets.

The old wheel and three upstream scale-fix files are privately bundled by hash.
No competition input, real-image tracking, label access, solver or score is used.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time
import zipfile
import zlib

WORK = Path("/kaggle/working/logs/hoct-scale-cpu")
OLD_SHA = "c6194e81a05d272913dd0945ede2484d9a7d89f8d751c4eb1e30c43031e9093c"
FIX_SHA = "733d9c70e9e51103156cfc987b216bb73ea6f5138d74403c4d207ba277ad72f6"
FIX_COMMIT = "8709ee9d3c4d7aae1f022b259d48dc6584237b02"
FIXED_FILES = {"_api.py", "data/_transforms.py", "features/graph.py"}
V1_SHA = "5bd836dfcb15ad796ea79a9595841a3e73b650a71c4acba3fc66aac65d745b33"
SCALE = (1.0, 1.625, 0.40625, 0.40625)
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sources(old_wheel, fixed_archive):
    old_raw, fixed_raw = old_wheel.read_bytes(), fixed_archive.read_bytes()
    if sha(old_raw) != OLD_SHA or sha(fixed_raw) != FIX_SHA:
        raise ValueError("Upstream reviewed source archive changed")
    with zipfile.ZipFile(io.BytesIO(old_raw)) as archive:
        old = {name[5:]: archive.read(name).decode() for name in archive.namelist()
               if name.startswith("hoct/") and name.endswith(".py")}
    with tarfile.open(fileobj=io.BytesIO(fixed_raw)) as archive:
        fixed = {entry.name.split("/src/hoct/", 1)[1]: archive.extractfile(entry).read().decode()
                 for entry in archive.getmembers() if "/src/hoct/" in entry.name and entry.name.endswith(".py")}
    changed = {name for name in old if name in fixed and old[name] != fixed[name]}
    if changed != FIXED_FILES or any(name not in fixed for name in old if name != "__about__.py"):
        raise ValueError("Upstream difference is not limited to the reviewed three scale files")
    # Retain wheel-generated version metadata; provenance separately identifies the fix.
    new = {name: fixed[name] if name in FIXED_FILES else value for name, value in old.items()}
    return {f"upstream/{flavour}/hoct/{name}": value
            for flavour, content in (("old", old), ("fixed", new)) for name, value in content.items()}


def package(args):
    root = Path(__file__).resolve().parents[1]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.owner):
        raise ValueError("Invalid owner")
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=root, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != subprocess.check_output(["git", "rev-parse", "@{upstream}"], cwd=root, text=True).strip():
        raise ValueError("Commit and push before packaging")
    files = sources(args.old_wheel, args.fixed_archive)
    for path in (Path(__file__).resolve(), Path(__file__).resolve().with_name("fetch_hoct_wheels.py")):
        content = path.read_text()
        if content != subprocess.check_output(["git", "show", f"HEAD:{path.relative_to(root)}"], cwd=root).decode():
            raise ValueError("Uncommitted project source")
        files[path.name] = content
    manifest = {"git_commit": revision, "upstream_fix_commit": FIX_COMMIT,
                "upstream_old_wheel_sha256": OLD_SHA, "upstream_fixed_archive_sha256": FIX_SHA,
                "files": {name: sha(content.encode()) for name, content in files.items()}}
    files["source_manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    payload = zlib.compress(json.dumps(files, sort_keys=True).encode(), 9)
    code = f'''import base64, hashlib, json, os, subprocess, sys, zlib
from pathlib import Path
raw = base64.b85decode({base64.b85encode(payload).decode()!r})
assert hashlib.sha256(raw).hexdigest() == {sha(payload)!r}
bundle = Path("/kaggle/working/hoct-scale-source")
bundle.mkdir(exist_ok=False)
for name, source in json.loads(zlib.decompress(raw)).items():
    path = (bundle / name).resolve()
    if bundle not in path.parents:
        raise ValueError("Invalid source path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
env = {{**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "POLARS_MAX_THREADS": "2", "POLARS_PREFER_PKG": "32", "PYTHONUNBUFFERED": "1"}}
subprocess.run([sys.executable, str(bundle / "check_hoct_scale.py"), "run", "--bundle", str(bundle)], env=env, check=True, timeout=840)
'''
    compile(code, "hoct-cpu-bootstrap", "exec")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                "cells": [{"cell_type": "markdown", "metadata": {}, "id": "scope", "source":
                           "# HOCT CPU scale/asset audit\n\nSynthetic labels only; no competition data, model inference, "
                           "solver, quality score or submission. Network retrieves pinned official dependencies and v1 weight.\n\n"
                           f"Project source: {revision}. Upstream fix: {FIX_COMMIT}.\n\n"
                           "https://github.com/royerlab/hoct/pull/6"},
                          {"cell_type": "code", "metadata": {}, "id": "audit", "source": code,
                           "outputs": [], "execution_count": None}]}
    metadata = {"id": args.owner + "/biohub-hoct-scale-cpu-audit", "title": "Biohub HOCT Scale CPU Audit",
                "code_file": "hoct_scale.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
                "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1"],
                "competition_sources": [], "kernel_sources": [], "model_sources": [], "docker_image": DOCKER}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in (("kernel-metadata.json", metadata), ("hoct_scale.ipynb", notebook), ("source_manifest.json", manifest)):
        save(args.output_dir / name, data)
    print(json.dumps({"git_commit": revision, "id": metadata["id"], "source_files": len(manifest["files"]),
                      "notebook_bytes": (args.output_dir / "hoct_scale.ipynb").stat().st_size}, indent=2))


def case(bundle, flavour):
    sys.path.insert(0, str(bundle / "upstream" / flavour))
    import importlib.metadata
    import numpy as np
    import polars as pl
    import torch
    import tracksdata as td
    import hoct._api as api
    from hoct.features.graph import create_graph
    from tracksdata.functional import TilingScheme, apply_tiled

    if Path(api.__file__).resolve() != bundle / "upstream" / flavour / "hoct/_api.py":
        raise ValueError("Wrong HOCT source imported")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    labels = np.zeros((2, 24, 48, 48), dtype=np.int16)
    centres = ((0, 8, 24, 24), (1, 14, 24, 24), (1, 8, 36, 24))
    for label, (t, z, y, x) in enumerate(centres, 1):
        labels[t, z-1:z+2, y-1:y+2, x-1:x+2] = label
    with td.options.Options(n_workers=1):
        graph = create_graph(labels, distance_threshold=8.0, n_neighbors=5, delta_t=1, scale=SCALE)
    nodes = graph.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    points = {row["node_id"]: tuple(round(row[k]) for k in ("t", "z", "y", "x")) for row in nodes.to_dicts()}
    edges = graph.edge_attrs(attr_keys=[])
    pairs = sorted((points[a], points[b]) for a, b in zip(edges["source_id"], edges["target_id"]))
    expected = [(centres[0], centres[1 if flavour == "old" else 2])]
    if pairs != expected:
        raise ValueError(f"Synthetic anisotropic candidate contract differs: {pairs}")
    details = {"flavour": flavour, "candidate_pairs_tzyx": pairs, "datasets": {}, "passed": False}
    save(WORK / (flavour + "_details.json"), details)
    kwargs = {"scale": SCALE} if flavour == "fixed" else {}
    reports = {}
    for kind, tiling in (("frame", None), ("tiled", TilingScheme(tile_shape=labels.shape, overlap_shape=(0, 0, 0, 0)))):
        dataset = api._create_dataset(graph, tiling_scheme=tiling, window_size=2, test_time_augs=0, **kwargs)
        items = [dataset[0]] if kind == "frame" else list(dataset)
        if len(items) != 1 or items[0] is None:
            raise ValueError("Unexpected synthetic dataset coverage")
        item = items[0]
        positions = item["node_pos"].numpy()
        original = np.asarray([points[n][1:] for n in item["node_id"].tolist()])
        origin = np.zeros(3)
        if kind == "tiled":
            tiles = [tile for tile in apply_tiled(graph=graph, tiling_scheme=tiling, func=lambda value: value)
                     if tile.graph_filter.num_edges() > 0]
            if len(tiles) != 1:
                raise ValueError("Unexpected number of nonempty tiles")
            origin = np.asarray([part.start for part in tiles[0].slicing[1:]])
        intended = (original - origin) * (np.asarray(SCALE[1:]) if flavour == "fixed" else 1)
        details["datasets"][kind] = {"positions": positions.tolist(), "expected": intended.tolist(),
                                    "tile_origin_voxels": origin.tolist()}
        save(WORK / (flavour + "_details.json"), details)
        if not np.allclose(positions, intended, rtol=0, atol=1e-6):
            raise ValueError(f"{kind}: dataset positions do not follow the expected units")
        if (not torch.isfinite(item["node_feats"]).all() or item["edge_targets"] is not None
                or item["gt_graph"] is not None):
            raise ValueError("Nonfinite features or unexpected ground-truth content")
        raw = pl.DataFrame({"z": [1.0, 2.0], "y": [2.0, 3.0], "x": [3.0, 4.0], "area": [27.0, 54.0]})
        transformed, repeat = raw, raw
        for transform in dataset._df_transforms:
            transformed, repeat = transform(transformed), transform(repeat)
        expected_area = np.asarray(raw["area"]) * (np.prod(SCALE[1:]) if flavour == "fixed" else 1)
        if not transformed.equals(repeat) or not np.allclose(transformed["area"], expected_area):
            raise ValueError("Physical area scaling is missing or nondeterministic")
        reports[kind] = {"node_positions": positions.tolist(), "feature_shape": list(item["node_feats"].shape),
                         "tile_origin_voxels": origin.tolist(),
                         "area_after_transform": transformed["area"].to_list(), "finite": True,
                         "deterministic_scale": True, "gt_fields_absent": True}
    result = {"flavour": flavour, "passed": True, "candidate_pairs_tzyx": pairs,
              "scale_tzyx": SCALE, "distance_threshold": 8.0,
              "physical_displacements_um": {"z_case": 9.75, "y_case": 4.875}, "datasets": reports,
              "versions": {n: importlib.metadata.version(n) for n in ("torch", "numpy", "polars", "tracksdata", "spatial-graph")}}
    save(WORK / (flavour + ".json"), result)
    details["passed"] = True
    save(WORK / (flavour + "_details.json"), details)
    print(json.dumps(result), flush=True)


def run(bundle):
    WORK.mkdir(parents=True, exist_ok=False)
    report = {"passed": False, "device": "cpu", "competition_input_mounted": False,
              "real_image_inference": False, "ground_truth_accessed": False,
              "model_forward_executed": False, "solver_run": False, "quality_score": None}
    started = time.monotonic()
    try:
        manifest = json.loads((bundle / "source_manifest.json").read_text())
        for name, expected in manifest["files"].items():
            path = (bundle / name).resolve()
            if bundle not in path.parents or sha(path.read_bytes()) != expected:
                raise ValueError("Bundled source changed")
        report["source_manifest"] = manifest
        support_roots = {p.resolve() for p in (Path("/kaggle/input/biohub-tracking-support-pack-50ep-v1"),
                         Path("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1")) if p.is_dir()}
        if len(support_roots) != 1:
            raise ValueError("Expected one pinned support pack mount")
        support = support_roots.pop()
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-deps",
                        "--find-links", str(support / "wheels"), "tracksdata", "zarr==3.2.1", "numcodecs==0.15.1",
                        "donfig==0.8.1.post1", "geff==1.2.0.1.1", "geff-spec==1.1.1", "pyscipopt==6.2.1",
                        "ilpy==0.6.0", "rustworkx==0.18.0", "polars==1.42.0", "polars-runtime-32==1.42.0",
                        "bidict==0.23.1", "imagecodecs==2026.6.26"], check=True, timeout=180)
        wheels = WORK / "wheels"
        fetch_code = ("import fetch_hoct_wheels as f; "
                      "f.PACKAGES.update({'witty': '0.3.2', 'CT3': '3.4.0.post5'}); f.main()")
        subprocess.run([sys.executable, "-c", fetch_code, "--output-dir", str(wheels)],
                       cwd=bundle, check=True, timeout=360)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-deps",
                        "--find-links", str(wheels), "spatial-graph==0.1.1", "pooch==1.9.0", "gurobipy==12.0.3",
                        "witty==0.3.2", "CT3==3.4.0.post5"], check=True, timeout=90)
        report["download_receipt"] = json.loads((WORK / "download_receipt.json").read_text())
        for flavour in ("old", "fixed"):
            subprocess.run([sys.executable, str(bundle / "check_hoct_scale.py"), "case", "--bundle", str(bundle), "--flavour", flavour], check=True, timeout=120)
            report[flavour] = json.loads((WORK / (flavour + ".json")).read_text())
        weight = WORK / "general_v1.pt"
        with weight.open("xb") as output:
            subprocess.run(["curl", "-4", "--proto", "=https", "--proto-redir", "=https", "--location", "--fail", "--silent", "--show-error",
                            "--connect-timeout", "10", "--max-time", "180", "--max-filesize", "25496698",
                            "https://github.com/royerlab/hoct/releases/download/weights-v1/general_v1.pt"],
                           stdout=output, check=True, timeout=190)
        if weight.stat().st_size != 25_496_698 or sha(weight.read_bytes()) != V1_SHA:
            raise ValueError("Complete official v1 checkpoint hash/size mismatch")
        import torch
        torch.set_num_threads(2)
        model = torch.jit.load(str(weight), map_location="cpu").eval()
        if any(p.device.type != "cpu" or not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Checkpoint parameter contract failed")
        report["v1_asset"] = {"bytes": weight.stat().st_size, "sha256": V1_SHA,
                              "parameters": sum(p.numel() for p in model.parameters()),
                              "input_projection_shape": list(model.input_proj.weight.shape),
                              "forward_schema": str(model.forward.schema), "loaded_on_cpu": True}
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save(WORK / "runtime_receipt.json", report)
        (WORK / "run_summary.md").write_text("# HOCT CPU scale/asset audit\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n\nNo tracking-quality claim.\n")
        print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    local = sub.add_parser("package")
    local.add_argument("--owner", required=True)
    for key in ("old-wheel", "fixed-archive", "output-dir"):
        local.add_argument("--" + key, type=Path, required=True)
    for name in ("run", "case"):
        runtime = sub.add_parser(name)
        runtime.add_argument("--bundle", type=Path, required=True)
        if name == "case":
            runtime.add_argument("--flavour", choices=("old", "fixed"), required=True)
    args = parser.parse_args()
    if args.action == "package":
        package(args)
    else:
        if not Path("/kaggle/input").is_dir() or not Path("/kaggle/working").is_dir():
            raise RuntimeError("This audit executes only on Kaggle CPU")
        if args.action == "run":
            run(args.bundle.resolve())
        else:
            case(args.bundle.resolve(), args.flavour)


if __name__ == "__main__":
    main()
