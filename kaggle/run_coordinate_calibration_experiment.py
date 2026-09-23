#!/usr/bin/env python3
"""Cross-fit one bounded coordinate regressor on frozen E029 graphs.

Smoke extracts features/targets only. Full mode fits two deterministic folds
and scores only out-of-fold predictions; it never fits a deployment model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import audit_e000_error_budget as graphs
import coordinate_calibration as calibration
import run_flow_relink_experiment as flow

reference = flow.reference
E029_GEFF_TREE_SHA256 = "50ac680ef53c7456ea00fd1358ea706b33c61d62fb6e5f4b3ed7748cf6581aff"


def fold_assignment(names):
    assignment = {}
    for embryo in ("44b6", "6bba"):
        for index, name in enumerate(sorted(n for n in names if n.startswith(embryo + "_"))):
            assignment[name] = index % 2
    if set(assignment) != set(names):
        raise ValueError("Unexpected embryo names")
    return assignment


def identity_sha256(path):
    """Fingerprint every CSV field except node Z/Y/X, including row order."""
    digest = hashlib.sha256()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["row_type"] == "node":
                for key in ("z", "y", "x"):
                    del row[key]
            digest.update((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return digest.hexdigest()


def geff_to_csv_index(rows, times, positions):
    """Official csv_to_geffs assigns new IDs in the CSV node-row order."""
    if sorted(int(row["node_id"]) for row in rows) != list(range(len(times))):
        raise ValueError("Official GEFF node reindexing changed")
    for row in rows:
        index = int(row["node_id"])
        if (int(row["t"]) != int(times[index])
                or tuple(float(row[key]) for key in ("z", "y", "x"))
                != tuple(float(value) for value in positions[index])):
            raise ValueError("GEFF and original CSV coordinates do not align")
    return {int(row["node_id"]): int(row["node_id"]) for row in rows}


def load_predictor(repo_dir):
    path = repo_dir / "scripts/predict_unet_transformer.py"
    if reference.stability.file_sha256(path) != reference.stability.EXPECTED_SUPPORT_PREDICTOR_SHA256:
        raise ValueError("Frozen primary predictor source changed")
    sys.path[:0] = [str(repo_dir / "src"), str(repo_dir / "scripts")]
    spec = importlib.util.spec_from_file_location("e032_frozen_primary", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def capture_movie(args, name, nodes, model, predictor, downsample, device):
    import numpy as np
    import torch
    import tracksdata as td
    import zarr

    image_path = args.data_dir / "train" / f"{name}.zarr"
    dataset = predictor.open_dataset(image_path, normalize=False, load_image=False, downsample=downsample)
    image = zarr.open_group(str(image_path), mode="r")["0"]
    low, high = (float(dataset.quantiles[key]) for key in ("0.001", "0.999"))
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("Invalid frozen input normalization")
    ids = nodes.node_id.to_numpy(dtype=np.int64)
    times = nodes.t.to_numpy(dtype=np.int64)
    positions = nodes[["z", "y", "x"]].to_numpy(dtype=np.float64)
    features = np.empty((len(nodes), 224), dtype=np.float16)
    captured = np.zeros(len(nodes), dtype=bool)
    with torch.inference_mode():
        for start in range(image.shape[0] - 1):
            indexes = (0, 1) if start == 0 else (1,)
            needed = {index: np.flatnonzero(times == start + index) for index in indexes}
            if not any(len(rows) for rows in needed.values()):
                continue
            frames = torch.stack([predictor._load_frame(image, t, list(dataset.image_shape[1:]), downsample)
                                  for t in (start, start + 1)])
            frames = ((frames - low) / (high - low + 1e-6)).clamp(0.0)[None].to(device)
            encoded, _ = model.encode(frames)
            for index, rows in needed.items():
                values = calibration.sample_features(encoded[0, index], positions[rows] / downsample)
                features[rows] = values.cpu().numpy().astype(np.float16)
                captured[rows] = True
            del encoded, frames
    if not captured.all() or not np.isfinite(features).all():
        raise ValueError(f"Incomplete or invalid feature capture: {name}")
    pred = graphs.load_graph(args.control_dir / f"{name}.geff", td)
    gt = graphs.load_graph(args.data_dir / "train" / f"{name}.geff", td)
    scale = np.asarray(dataset.scale, dtype=np.float64)
    graphs.match_graph(pred, gt, tuple(scale), 7.0)
    truth = {int(row["node_id"]): row for row in gt.node_attrs().iter_rows(named=True)}
    pred_rows = list(pred.node_attrs().iter_rows(named=True))
    index_of = geff_to_csv_index(pred_rows, times, positions)
    target = np.full((len(ids), 3), np.nan, dtype=np.float32)
    for row in pred_rows:
        matched = row.get(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID)
        if matched is None or int(matched) < 0:
            continue
        other = truth[int(matched)]
        index = index_of[int(row["node_id"])]
        if int(other["t"]) != times[index]:
            raise ValueError("Ground-truth target crossed frames")
        delta = (np.asarray([other[key] for key in ("z", "y", "x")]) - positions[index]) * scale
        if np.isfinite(delta).all() and np.linalg.norm(delta) <= calibration.MAX_TRAIN_RESIDUAL_UM:
            target[index] = delta
    matched_count = int(np.isfinite(target).all(axis=1).sum())
    if matched_count == 0:
        raise ValueError(f"No calibration training pairs: {name}")
    path = args.output_dir / "features" / f"{name}.npz"
    np.savez_compressed(path, features=features, target=target, node_id=ids, t=times,
                        positions=positions.astype(np.int64), scale=scale,
                        shape=np.asarray(image.shape[1:], dtype=np.int64))
    return {"nodes": len(ids), "matched_training_pairs": matched_count,
            "cache_sha256": reference.stability.file_sha256(path)}


def train_fold(args, names, assignment, held_out):
    import numpy as np

    training = sorted(name for name in names if assignment[name] != held_out)
    prediction = sorted(name for name in names if assignment[name] == held_out)
    if len(training) != 32 or len(prediction) != 32 or set(training) & set(prediction):
        raise ValueError("Cross-fit movie separation changed")
    blocks = []
    for name in training:
        with np.load(args.output_dir / "features" / f"{name}.npz") as data:
            mask = np.isfinite(data["target"]).all(axis=1)
            blocks.append((data["features"][mask], data["target"][mask]))
    model = calibration.fit_ridge(blocks)
    path = args.output_dir / f"fold_{held_out}_head.npz"
    np.savez(path, **model)
    return model, {"training_movies": training, "prediction_movies": prediction,
                   "training_pairs": sum(len(x) for x, _ in blocks),
                   "model_sha256": reference.stability.file_sha256(path)}


def write_candidate(args, names, assignment, models):
    import numpy as np

    lookup, stats = {}, {}
    for name in names:
        with np.load(args.output_dir / "features" / f"{name}.npz") as data:
            result = calibration.calibrated_positions(models[assignment[name]], data["features"],
                data["positions"], data["scale"], data["shape"])
            lookup[name] = dict(zip(data["node_id"].tolist(), result.tolist()))
            displacement = np.linalg.norm((result - data["positions"]) * data["scale"], axis=1)
            stats[name] = {"changed_nodes": int((displacement > 0).sum()),
                           "max_displacement_um": float(displacement.max())}
    path = args.output_dir / "raw_submission.csv"
    with args.control_csv.open(newline="") as source, path.open("x", newline="") as output:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["row_type"] == "node":
                xyz = lookup[row["dataset"]].pop(int(row["node_id"]))
                row.update(dict(zip(("z", "y", "x"), xyz)))
            writer.writerow(row)
    if any(lookup.values()) or identity_sha256(path) != identity_sha256(args.control_csv):
        raise ValueError("Coordinate calibration changed graph identity or coverage")
    if sum(value["changed_nodes"] for value in stats.values()) == 0:
        raise ValueError("Calibration made no coordinate changes")
    return stats


def run(args):
    os.environ.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                       "OPENBLAS_NUM_THREADS": "4", "POLARS_MAX_THREADS": "4"})
    graphs.prepare_imports(args.runtime_dir, args.scorer_dir)
    import numpy as np
    import pandas as pd
    import torch

    names = reference.selected_names(args.control_dir, args.mode)
    if reference.stability.file_sha256(args.control_csv) != flow.E029_CSV_SHA256:
        raise ValueError("Frozen E029 control CSV changed")
    if flow.graph_tree_sha256(sorted(args.control_dir.glob("*.geff"))) != E029_GEFF_TREE_SHA256:
        raise ValueError("Frozen E029 prediction graph bytes changed")
    weights = args.data_dir / "support-pack/weights/unet_transformer/split_0/edge_predictor_best.pth"
    if reference.stability.file_sha256(weights) != reference.stability.EXPECTED_PRIMARY_WEIGHT_SHA256:
        raise ValueError("Frozen primary model changed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "features").mkdir()
    image_dir = args.output_dir / "input/test"
    image_dir.mkdir(parents=True)
    for name in names:
        (image_dir / f"{name}.zarr").symlink_to(args.data_dir / "train" / f"{name}.zarr", target_is_directory=True)
    assignment = fold_assignment(names)
    manifest = {"experiment": "E032", "mode": args.mode, "datasets": names,
                "control_csv_sha256": flow.E029_CSV_SHA256,
                "control_graph_tree_sha256": E029_GEFF_TREE_SHA256,
                "primary_weight_sha256": reference.stability.EXPECTED_PRIMARY_WEIGHT_SHA256,
                "fold_assignment": assignment, "ridge_mean_loss_l2": calibration.RIDGE_L2,
                "maximum_shift_um": calibration.MAX_SHIFT_UM,
                "maximum_training_residual_um": calibration.MAX_TRAIN_RESIDUAL_UM,
                "feature_definition": "identity first-seen temporal features; trilinear center plus six axis differences; float16 capture",
                "feature_code_sha256": reference.stability.file_sha256(Path(calibration.__file__)),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "runtime_parameter_search": False, "capture": {}}
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    predictor = load_predictor(args.data_dir / "support-pack/repo")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    model, window, downsample = predictor.load_model(weights, device)
    if window != 2 or downsample != (1, 4, 4):
        raise ValueError("Frozen temporal context or feature grid changed")
    frame = pd.read_csv(args.control_csv)
    nodes = frame[frame.row_type == "node"]
    for index, name in enumerate(names, 1):
        manifest["capture"][name] = capture_movie(args, name, nodes[nodes.dataset == name],
                                                 model, predictor, downsample, device)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"CAPTURED {index}/{len(names)} {name}: {manifest['capture'][name]}", flush=True)
    del model, frame, nodes
    torch.cuda.empty_cache()
    if args.mode == "smoke":
        (args.output_dir / "run_summary.md").write_text(
            "# E032 extraction smoke\n\nBoth movies passed complete feature/target capture. "
            "No regressor was fitted and no candidate score or promotion is established.\n")
        return
    models, folds = {}, {}
    for held_out in (0, 1):
        models[held_out], folds[held_out] = train_fold(args, names, assignment, held_out)
    manifest["folds"] = folds
    manifest["coordinate_changes"] = write_candidate(args, names, assignment, models)
    if reference.stability.file_sha256(args.control_csv) != flow.E029_CSV_SHA256:
        raise ValueError("Control bytes changed during the experiment")
    args.expected_control_score = flow.E029_SCORE
    args.minimum_pooled_delta = 0.001
    args.experiment = "E032 out-of-fold coordinate calibration on E029"
    os.environ["PYTHONPATH"] = os.pathsep.join([str(args.runtime_dir), str(args.scorer_dir / "src"),
                                               str(args.scorer_dir / "scripts")])
    reference.complete_export(args, names, manifest, args.output_dir / "raw_submission.csv")
    if identity_sha256(args.output_dir / "submission.csv") != identity_sha256(args.control_csv):
        raise ValueError("Export changed graph identity")
    result = json.loads((args.output_dir / "stability.json").read_text())
    lines = ["# E032 out-of-fold coordinate calibration", "",
             "Each regressor is fitted only on the other 32 movies. Nodes and edges are fixed.", "",
             "| Group | E029 | E032 | Delta |", "|---|---:|---:|---:|"]
    for key, group in result["groups"].items():
        lines.append(f"| {key} | {group['control']['score']:.9f} | {group['candidate']['score']:.9f} | {group['delta']['score']:+.9f} |")
    lines += ["", "```json", json.dumps(result["gates"], indent=2), "```", "",
              "This cross-fits only the new regressor. The pretrained detector and earlier method selection used public training data.", ""]
    (args.output_dir / "run_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-dir", "control-dir", "control-csv", "runtime-dir", "scorer-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
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
                f"# E032 execution failed\n\n{type(exc).__name__}: {exc}\n\nNo score improvement or promotion is established.\n")
        raise


if __name__ == "__main__":
    main()
