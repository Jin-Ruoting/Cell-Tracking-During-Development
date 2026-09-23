#!/usr/bin/env python3
"""Capture E029 weak peaks without changing detection or rerunning association.

The exact E029 runtime predictor is required. Only its edge loop is removed;
the two seeds, D4 logits, retention guard and first-seen frame context remain.
Every movie must reproduce its original pre-ILP detection-coordinate hash.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import run_flow_relink_experiment as flow

reference = flow.reference
PREDICTOR_SHA256 = "e5851002ad72730de0169d2c7a67ec21fd458d29cae900ab01ffcc935b8066b8"
LOW_THRESHOLD = 0.3
DETECTION_CONFIG = {
    "det_threshold": 0.965, "det_tta": True, "pool_kernel_um": 3.0,
    "secondary_detection_weight": 0.8,
    "minimum_candidate_retention": 0.9,
    "window_size": 2, "downsample": [1, 4, 4],
}


def instrument_peaks(source: str) -> str:
    """Add read-only peak collection at the same first-seen detection site."""
    tree = ast.parse(source)
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "predict_video"]
    if len(found) != 1:
        raise ValueError("Expected one predict_video function")
    function = found[0]
    initial = ast.parse("_weak_coords = []\n_weak_scores = []").body
    # Keep the function docstring first.
    function.body[1:1] = initial

    class Capture(ast.NodeTransformer):
        count = 0

        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "arr" and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "_detect_cells_pooled"):
                self.count += 1
                return [node, *ast.parse(
                    "_low = _detect_cells_pooled(det_logits[f_idx][0], t, 0.3, pool_k)\n"
                    "_lg = det_logits[f_idx][0][0]\n"
                    "_ix = torch.as_tensor(_low[:, 1:].astype(np.int64), device=_lg.device)\n"
                    "_score = torch.sigmoid(_lg[_ix[:, 0], _ix[:, 1], _ix[:, 2]]).float().cpu().numpy()\n"
                    "_weak_coords.append(_low.astype(np.float32))\n"
                    "_weak_scores.append(_score.astype(np.float32))\n"
                ).body]
            return self.generic_visit(node)

    capture = Capture()
    capture.visit(function)
    if capture.count != 1:
        raise ValueError("First-seen detection anchor changed")
    return ast.unparse(ast.fix_missing_locations(tree))


def capture_source(source: str, original_dir: Path, output_dir: Path) -> str:
    tree = ast.parse(instrument_peaks(source))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "predict_video")

    class RemoveEdges(ast.NodeTransformer):
        count = 0

        def visit_For(self, node):
            if (isinstance(node.target, ast.Name) and node.target.id == "f_idx"
                    and ast.dump(node.iter) == ast.dump(ast.parse("range(W - 1)", mode="eval").body)):
                self.count += 1
                return None
            return self.generic_visit(node)

    remover = RemoveEdges()
    remover.visit(function)
    if remover.count != 1 or not isinstance(function.body[-1], ast.Return):
        raise ValueError("Association loop or return anchor changed")
    function.body[-1:] = ast.parse(
        "_lc = np.concatenate(_weak_coords) if _weak_coords else np.empty((0, 4), dtype=np.float32)\n"
        "_ls = np.concatenate(_weak_scores) if _weak_scores else np.empty(0, dtype=np.float32)\n"
        "_lc[:, 1:] *= ds_arr\n"
        "return coords, (_lc.astype(np.int16), _ls, sorted(seen_frames))\n"
    ).body
    # The original runtime has two absolute log paths. Redirect both so the
    # frozen control directory cannot be modified by this new run.
    redirected = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Constant) and node.value == str(original_dir):
            node.value = str(output_dir)
            redirected += 1
    if redirected != 2:
        raise ValueError("Expected exactly two runtime log paths")
    return ast.unparse(ast.fix_missing_locations(tree))


def original_coordinates(raw_run_dir: Path) -> dict:
    rows = {}
    paths = sorted(raw_run_dir.glob("detector_coordinates_harmonic_association_production_*.jsonl"))
    if len(paths) != 2:
        raise ValueError("Original detector receipts missing")
    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            name = record["dataset"]
            if name in rows or record["stage"] != "post_detection_pre_graph_pre_ilp":
                raise ValueError("Ambiguous original detection receipt")
            rows[name] = record
    if reference.stability.movie_names_sha256(sorted(rows)) != reference.CORPUS_SHA256:
        raise ValueError("Original detector coverage changed")
    return rows


def validate_arrays(low, scores, frames, shape):
    import numpy as np

    if (low.ndim != 2 or low.shape[1] != 4 or scores.shape != (len(low),)
            or not np.isfinite(low).all() or not np.isfinite(scores).all()
            or not (low == np.rint(low)).all() or (low < 0).any()
            or (low >= np.asarray(shape)).any()
            or (scores <= LOW_THRESHOLD).any() or (scores > 1).any()
            or list(frames) != list(range(shape[0]))):
        raise ValueError("Invalid peak values, scales, bounds or frame coverage")
    if len(np.unique(low, axis=0)) != len(low):
        raise ValueError("Duplicate first-seen peaks")


def run(args):
    names = reference.selected_names(args.control_dir, args.mode)[args.shard::2]
    original = original_coordinates(args.raw_run_dir)
    source_path = args.raw_run_dir / "tracking_repo/scripts/predict_unet_transformer.py"
    if reference.stability.file_sha256(source_path) != PREDICTOR_SHA256:
        raise ValueError("Frozen E029 predictor changed")
    primary = args.data_dir / "support-pack/weights/unet_transformer/split_0/edge_predictor_best.pth"
    secondary = args.data_dir / "secondary-seed-v1/weights/unet_transformer/split_0/edge_predictor_best.pth"
    for path, digest in ((primary, reference.stability.EXPECTED_PRIMARY_WEIGHT_SHA256),
                         (secondary, reference.stability.EXPECTED_SECONDARY_WEIGHT_SHA256)):
        if reference.stability.file_sha256(path) != digest or not (path.parent / "config.json").is_file():
            raise ValueError("Frozen model or configuration missing/changed")
        original_config = (args.raw_run_dir / ("tracking_repo/weights" if path == primary else "secondary_seed_weights")
                           / "unet_transformer/split_0/config.json")
        if (path.parent / "config.json").read_bytes() != original_config.read_bytes():
            raise ValueError("Model configuration differs from E029")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "peaks").mkdir()
    source = capture_source(source_path.read_text(), args.raw_run_dir, args.output_dir)
    (args.output_dir / "capture_predictor.py").write_text(source + "\n")
    manifest = {
        "purpose": "frozen_e029_detection_peaks", "mode": args.mode, "shard": args.shard,
        "datasets": names, "predictor_sha256": PREDICTOR_SHA256,
        "capture_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "detection_config": DETECTION_CONFIG, "low_threshold": LOW_THRESHOLD,
        "primary_sha256": reference.stability.file_sha256(primary),
        "secondary_sha256": reference.stability.file_sha256(secondary),
        "ground_truth_accessed": False, "complete": False, "movies": {},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    repo_dir = args.raw_run_dir / "tracking_repo"
    sys.path[:0] = [str(args.runtime_dir), str(repo_dir / "src"), str(repo_dir / "scripts")]
    for key in list(os.environ):
        if key.startswith("BIOHUB_"):
            del os.environ[key]
    os.environ.update({"BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION": "0.90",
                       "BIOHUB_EDGE_FEATURE_TTA": "1", "BIOHUB_SECONDARY_EDGE_FEATURE_TTA": "1",
                       "BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT": "0.75",
                       "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
    module = types.ModuleType("biohub_frozen_peak_capture")
    module.__file__ = str(source_path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(args.output_dir / "capture_predictor.py"), "exec"), module.__dict__)
    import numpy as np
    import torch
    import zarr

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    first, window, downsample = module.load_model(primary, device)
    second, second_window, second_downsample = module.load_model(secondary, device)
    if window != 2 or second_window != window or tuple(downsample) != (1, 4, 4) or downsample != second_downsample:
        raise ValueError("Frozen temporal/spatial configuration changed")
    cfg = module.PredictConfig(det_threshold=0.965, det_tta=True, pool_kernel_um=3.0)
    for name in names:
        started = time.monotonic()
        image = args.data_dir / "train" / f"{name}.zarr"
        high, (low, scores, frames) = module.predict_video(
            first, image, device, cfg, window_size=window, downsample=downsample,
            secondary_model=second, secondary_detection_weight=0.8)
        digest = hashlib.sha256(np.ascontiguousarray(high.astype("<i2")).tobytes()).hexdigest()
        if digest != original[name]["coordinate_sha256"] or len(high) != original[name]["rows"]:
            raise ValueError(f"E029 detector coordinate parity failed: {name}")
        shape = list(zarr.open_group(str(image), mode="r")["0"].shape)
        validate_arrays(low, scores, frames, shape)
        path = args.output_dir / "peaks" / f"{name}.npz"
        np.savez_compressed(path, low_coords=low, low_score=scores,
                            processed_frames=np.asarray(frames), image_shape=np.asarray(shape))
        record = {"high_coordinate_sha256": digest, "high_nodes": len(high), "peaks": len(low),
                  "frames": len(frames), "shape": shape, "cache_sha256": reference.stability.file_sha256(path),
                  "detector_parity_passed": True, "seconds": time.monotonic() - started}
        manifest["movies"][name] = record
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"dataset": name, **record}), flush=True)
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "run_summary.md").write_text(
        f"# Frozen detector peak capture\n\n{len(names)} movies completed; all original E029 "
        "detection-coordinate hashes reproduced exactly. No ground-truth labels or graph score were used.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("raw-run-dir", "data-dir", "runtime-dir", "control-dir", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    try:
        run(args)
    except Exception as exc:
        if args.output_dir.exists():
            (args.output_dir / "run_summary.md").write_text(
                f"# Peak capture failed\n\n{type(exc).__name__}: {exc}\n\nCache cannot be promoted.\n")
        raise


if __name__ == "__main__":
    main()
