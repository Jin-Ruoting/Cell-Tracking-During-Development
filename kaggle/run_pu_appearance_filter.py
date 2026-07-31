#!/usr/bin/env python3
"""Train and apply the frozen E028 cross-embryo PU appearance filter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from audit_deepcenter_rescue import (
    frame_nodes,
    maximum_radius_matching,
    points_um,
)
from validate_e006_postprocess import (
    CSV_COLUMNS,
    E025_NOTEBOOK_SHA256,
    file_sha256,
    graph_rows,
    load_notebook_namespace,
)


SCHEMA_VERSION = 1
BASE_SEED = 20260731
MATCH_RADIUS_UM = 7.0
POSITIVE_PRIOR = 0.98
POSITIVE_QUANTILE = 0.01
OOF_FOLDS = 4
EPOCHS = 30
POSITIVE_BATCH_SIZE = 512
UNLABELED_BATCH_SIZE = 4096
MAX_UNLABELED_PER_POSITIVE = 8
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
COMPONENT_MIN_NODES = 6
COMPONENT_MAX_NODES = 10
MIN_BELOW_THRESHOLD_FRACTION = 0.75
MAX_REMOVED_FRACTION = 0.0025
MAX_REMOVED_NODES = 120
FEATURE_NAMES = (
    "center",
    "inner_mean",
    "inner_std",
    "inner_min",
    "inner_max",
    "inner_center_contrast",
    "inner_range",
    "outer_mean",
    "outer_std",
    "outer_min",
    "outer_max",
    "center_outer_contrast",
    "inner_outer_contrast",
    "peak_prominence",
    "outer_range",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--support-src", type=Path, required=True)
    parser.add_argument("--scorer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=64)
    parser.add_argument("--expected-names-sha256", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def movie_names_sha256(names: list[str]) -> str:
    payload = "".join(f"{name}.geff\n" for name in sorted(names)).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_args(args: argparse.Namespace) -> list[Path]:
    for path in (
        args.baseline_dir,
        args.image_dir,
        args.ground_truth_dir,
        args.runtime_dir,
        args.support_src,
        args.scorer_dir,
    ):
        if not path.is_dir():
            raise NotADirectoryError(path)
    if not args.notebook.is_file():
        raise FileNotFoundError(args.notebook)
    if file_sha256(args.notebook) != E025_NOTEBOOK_SHA256:
        raise RuntimeError("E028 requires the pinned E025 notebook")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.expected_count <= 0:
        raise ValueError("--expected-count must be positive")
    baseline_paths = sorted(args.baseline_dir.glob("*.geff"))
    if len(baseline_paths) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} baseline movies, "
            f"found {len(baseline_paths)}"
        )
    names = [path.stem for path in baseline_paths]
    if movie_names_sha256(names) != args.expected_names_sha256:
        raise RuntimeError("E028 movie-name SHA256 changed")
    groups = {
        prefix: [name for name in names if name.startswith(f"{prefix}_")]
        for prefix in ("44b6", "6bba")
    }
    if set(groups["44b6"]) | set(groups["6bba"]) != set(names):
        raise RuntimeError("Unexpected embryo prefix in E028 corpus")
    if len(groups["44b6"]) != len(groups["6bba"]):
        raise RuntimeError("E028 embryo groups are not balanced")
    return baseline_paths


def sample_offsets(scale: np.ndarray, radius_um: float) -> np.ndarray:
    steps = np.maximum(
        1,
        np.rint(float(radius_um) / scale).astype(np.int64),
    )
    offsets = [[0, 0, 0]]
    for axis, step in enumerate(steps):
        for sign in (-1, 1):
            row = [0, 0, 0]
            row[axis] = int(sign * step)
            offsets.append(row)
    for y_sign in (-1, 1):
        for x_sign in (-1, 1):
            offsets.append(
                [
                    0,
                    int(y_sign * steps[1]),
                    int(x_sign * steps[2]),
                ]
            )
    return np.asarray(offsets, dtype=np.int64)


def sampled_values(
    volume: np.ndarray,
    points: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    centers = np.rint(points).astype(np.int64)
    shape = np.asarray(volume.shape, dtype=np.int64)
    result = np.empty((len(points), len(offsets)), dtype=np.float32)
    for index, offset in enumerate(offsets):
        coordinates = np.clip(centers + offset[None, :], 0, shape - 1)
        result[:, index] = np.asarray(
            volume[
                coordinates[:, 0],
                coordinates[:, 1],
                coordinates[:, 2],
            ],
            dtype=np.float32,
        )
    return result


def robust_limits(volume: np.ndarray) -> tuple[float, float]:
    sample = np.asarray(volume)[::1, ::8, ::8].astype(
        np.float32,
        copy=False,
    )
    lower = float(np.quantile(sample, 0.01))
    upper = float(np.quantile(sample, 0.999))
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise RuntimeError("Image normalization limits are not finite")
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def normalize_samples(
    values: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    return np.clip(
        (values.astype(np.float32, copy=False) - lower) / (upper - lower),
        0.0,
        1.0,
    )


def appearance_features(
    volume: np.ndarray,
    points: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    inner = sampled_values(
        volume,
        points,
        sample_offsets(scale, 1.0),
    )
    outer = sampled_values(
        volume,
        points,
        sample_offsets(scale, 3.0),
    )
    lower, upper = robust_limits(volume)
    inner = normalize_samples(inner, lower, upper)
    outer = normalize_samples(outer, lower, upper)
    center = inner[:, 0]
    inner_mean = inner.mean(axis=1)
    inner_std = inner.std(axis=1)
    inner_min = inner.min(axis=1)
    inner_max = inner.max(axis=1)
    outer_mean = outer.mean(axis=1)
    outer_std = outer.std(axis=1)
    outer_min = outer.min(axis=1)
    outer_max = outer.max(axis=1)
    features = np.column_stack(
        [
            center,
            inner_mean,
            inner_std,
            inner_min,
            inner_max,
            center - inner_mean,
            inner_max - inner_min,
            outer_mean,
            outer_std,
            outer_min,
            outer_max,
            center - outer_mean,
            inner_mean - outer_mean,
            inner_max - outer_mean,
            outer_max - outer_min,
        ]
    ).astype(np.float32, copy=False)
    if features.shape[1] != len(FEATURE_NAMES):
        raise AssertionError("Appearance feature width changed")
    if not np.isfinite(features).all():
        raise RuntimeError("Appearance features contain non-finite values")
    return features


def matched_prediction_ids(
    nodes: dict[int, dict[str, object]],
    gt_nodes: dict[int, dict[str, object]],
    scale: np.ndarray,
) -> set[int]:
    prediction_by_frame = frame_nodes(nodes)
    gt_by_frame = frame_nodes(gt_nodes)
    matched: set[int] = set()
    for frame, gt_ids in sorted(gt_by_frame.items()):
        prediction_ids = prediction_by_frame.get(frame, [])
        matching = maximum_radius_matching(
            points_um(prediction_ids, nodes, scale),
            points_um(gt_ids, gt_nodes, scale),
            MATCH_RADIUS_UM,
        )
        for prediction_index, gt_index in enumerate(matching):
            if gt_index >= 0:
                matched.add(int(prediction_ids[prediction_index]))
    return matched


def extract_dataset_features(
    namespace: dict[str, object],
    baseline_path: Path,
    ground_truth_dir: Path,
    cache_path: Path,
    scale: np.ndarray,
) -> dict[str, object]:
    dataset = baseline_path.stem
    _, nodes, edges = graph_rows(namespace, baseline_path)
    gt_path = ground_truth_dir / f"{dataset}.geff"
    if not gt_path.is_dir():
        raise FileNotFoundError(gt_path)
    _, gt_nodes, _ = graph_rows(namespace, gt_path)
    positive_ids = matched_prediction_ids(nodes, gt_nodes, scale)
    node_ids = np.asarray(sorted(nodes), dtype=np.int64)
    index_by_id = {
        int(node_id): index for index, node_id in enumerate(node_ids)
    }
    features = np.empty(
        (len(node_ids), len(FEATURE_NAMES)),
        dtype=np.float32,
    )
    by_frame = frame_nodes(nodes)
    for frame_index, (frame, frame_ids) in enumerate(
        sorted(by_frame.items()),
        start=1,
    ):
        volume = np.asarray(
            namespace["read_test_frame"](dataset, int(frame), {})
        )
        points = np.asarray(
            [
                [
                    float(nodes[node_id]["z"]),
                    float(nodes[node_id]["y"]),
                    float(nodes[node_id]["x"]),
                ]
                for node_id in frame_ids
            ],
            dtype=np.float64,
        )
        rows = np.asarray(
            [index_by_id[node_id] for node_id in frame_ids],
            dtype=np.int64,
        )
        features[rows] = appearance_features(volume, points, scale)
        if frame_index % 20 == 0:
            print(
                f"{dataset}: features {frame_index}/{len(by_frame)}",
                flush=True,
            )
    labels = np.asarray(
        [int(node_id) in positive_ids for node_id in node_ids],
        dtype=np.bool_,
    )
    np.savez(
        cache_path,
        node_ids=node_ids,
        features=features,
        labels=labels,
    )
    return {
        "dataset": dataset,
        "nodes": len(nodes),
        "edges": len(edges),
        "labeled_gt_nodes": len(gt_nodes),
        "matched_positive_nodes": int(labels.sum()),
        "unlabeled_nodes": int((~labels).sum()),
        "feature_cache": str(cache_path.resolve()),
        "feature_cache_sha256": file_sha256(cache_path),
    }


def load_feature_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        node_ids = np.asarray(data["node_ids"], dtype=np.int64)
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.bool_)
    if len(node_ids) != len(features) or len(labels) != len(features):
        raise RuntimeError(f"Feature cache row mismatch: {path}")
    return node_ids, features, labels


def sampled_training_arrays(
    cache_paths: dict[str, Path],
    names: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    positives = []
    unlabeled = []
    rng = np.random.default_rng(seed)
    total_unlabeled = 0
    sampled_unlabeled = 0
    for name in sorted(names):
        _, features, labels = load_feature_cache(cache_paths[name])
        positive = features[labels]
        unknown = features[~labels]
        if len(positive) == 0 or len(unknown) == 0:
            raise RuntimeError(f"{name}: PU class is empty")
        limit = min(
            len(unknown),
            MAX_UNLABELED_PER_POSITIVE * len(positive),
        )
        indices = rng.choice(len(unknown), size=limit, replace=False)
        positives.append(positive)
        unlabeled.append(unknown[indices])
        total_unlabeled += len(unknown)
        sampled_unlabeled += limit
    positive_array = np.concatenate(positives, axis=0)
    unlabeled_array = np.concatenate(unlabeled, axis=0)
    return positive_array, unlabeled_array, {
        "positive_samples": len(positive_array),
        "unlabeled_population": total_unlabeled,
        "sampled_unlabeled": sampled_unlabeled,
    }


def build_linear_model(torch, feature_count: int):
    return torch.nn.Linear(feature_count, 1)


def train_pu_model(
    cache_paths: dict[str, Path],
    names: list[str],
    seed: int,
    device_name: str,
) -> dict[str, object]:
    import torch
    import torch.nn.functional as functional

    positive, unlabeled, counts = sampled_training_arrays(
        cache_paths,
        names,
        seed,
    )
    fit_population = np.concatenate([positive, unlabeled], axis=0)
    mean = fit_population.mean(axis=0).astype(np.float32)
    std = fit_population.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    positive = (positive - mean) / std
    unlabeled = (unlabeled - mean) / std

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    model = build_linear_model(torch, positive.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    rng = np.random.default_rng(seed + 1)
    steps = max(
        1,
        math.ceil(len(positive) / POSITIVE_BATCH_SIZE),
        math.ceil(len(unlabeled) / UNLABELED_BATCH_SIZE),
    )
    epoch_losses = []
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for _ in range(steps):
            positive_indices = rng.integers(
                0,
                len(positive),
                size=min(POSITIVE_BATCH_SIZE, len(positive)),
            )
            unlabeled_indices = rng.integers(
                0,
                len(unlabeled),
                size=min(UNLABELED_BATCH_SIZE, len(unlabeled)),
            )
            positive_batch = torch.as_tensor(
                positive[positive_indices],
                dtype=torch.float32,
                device=device,
            )
            unlabeled_batch = torch.as_tensor(
                unlabeled[unlabeled_indices],
                dtype=torch.float32,
                device=device,
            )
            positive_logits = model(positive_batch).squeeze(-1)
            unlabeled_logits = model(unlabeled_batch).squeeze(-1)
            positive_risk = POSITIVE_PRIOR * functional.softplus(
                -positive_logits
            ).mean()
            negative_risk = functional.softplus(
                unlabeled_logits
            ).mean() - POSITIVE_PRIOR * functional.softplus(
                positive_logits
            ).mean()
            loss = (
                -negative_risk
                if float(negative_risk.detach().cpu()) < 0.0
                else positive_risk + negative_risk
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        epoch_losses.append(total_loss / steps)
    return {
        "model": model,
        "mean": mean,
        "std": std,
        "counts": counts,
        "epoch_losses": epoch_losses,
        "seed": seed,
        "training_movies": sorted(names),
        "device": str(device),
    }


def predict_probabilities(
    model_info: dict[str, object],
    features: np.ndarray,
    device_name: str,
) -> np.ndarray:
    import torch

    model = model_info["model"]
    mean = np.asarray(model_info["mean"], dtype=np.float32)
    std = np.asarray(model_info["std"], dtype=np.float32)
    standardized = (features - mean) / std
    device = torch.device(device_name)
    output = np.empty(len(features), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), 65536):
            stop = min(len(features), start + 65536)
            batch = torch.as_tensor(
                standardized[start:stop],
                dtype=torch.float32,
                device=device,
            )
            output[start:stop] = (
                torch.sigmoid(model(batch).squeeze(-1))
                .detach()
                .cpu()
                .numpy()
            )
    if not np.isfinite(output).all():
        raise RuntimeError("PU model produced non-finite probabilities")
    return output


def calibrate_source_embryo(
    source: str,
    cache_paths: dict[str, Path],
    names: list[str],
    device_name: str,
) -> tuple[float, dict[str, object], dict[str, object]]:
    source_names = sorted(
        name for name in names if name.startswith(f"{source}_")
    )
    if len(source_names) < OOF_FOLDS:
        raise RuntimeError(f"{source}: not enough movies for OOF calibration")
    oof_positive_scores = []
    fold_reports = []
    for fold in range(OOF_FOLDS):
        validation_names = [
            name
            for index, name in enumerate(source_names)
            if index % OOF_FOLDS == fold
        ]
        training_names = [
            name for name in source_names if name not in validation_names
        ]
        model_info = train_pu_model(
            cache_paths,
            training_names,
            BASE_SEED + fold + (0 if source == "44b6" else 100),
            device_name,
        )
        fold_scores = []
        for name in validation_names:
            _, features, labels = load_feature_cache(cache_paths[name])
            scores = predict_probabilities(
                model_info,
                features,
                device_name,
            )
            fold_scores.append(scores[labels])
        positive_scores = np.concatenate(fold_scores)
        oof_positive_scores.append(positive_scores)
        fold_reports.append(
            {
                "fold": fold,
                "training_movies": training_names,
                "validation_movies": validation_names,
                "positive_scores": len(positive_scores),
                "positive_score_minimum": float(positive_scores.min()),
                "positive_score_median": float(
                    np.median(positive_scores)
                ),
                "counts": model_info["counts"],
                "final_loss": float(model_info["epoch_losses"][-1]),
            }
        )
    combined = np.concatenate(oof_positive_scores)
    threshold = float(np.quantile(combined, POSITIVE_QUANTILE))
    final_model = train_pu_model(
        cache_paths,
        source_names,
        BASE_SEED + (1000 if source == "44b6" else 2000),
        device_name,
    )
    calibration = {
        "source_embryo": source,
        "folds": fold_reports,
        "oof_positive_scores": len(combined),
        "positive_quantile": POSITIVE_QUANTILE,
        "threshold": threshold,
        "oof_positive_minimum": float(combined.min()),
        "oof_positive_median": float(np.median(combined)),
        "oof_positive_maximum": float(combined.max()),
    }
    return threshold, final_model, calibration


def graph_components(
    nodes: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
) -> tuple[list[list[int]], dict[int, int]]:
    adjacency = {node_id: [] for node_id in nodes}
    outdegree: dict[int, int] = {}
    for edge in edges:
        source = int(edge["source_id"])
        target = int(edge["target_id"])
        if source not in nodes or target not in nodes:
            raise RuntimeError("Cannot componentize a dangling edge")
        adjacency[source].append(target)
        adjacency[target].append(source)
        outdegree[source] = outdegree.get(source, 0) + 1
    components = []
    unseen = set(nodes)
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component = []
        while stack:
            node_id = stack.pop()
            component.append(node_id)
            for neighbour in adjacency[node_id]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    return components, outdegree


def select_removed_components(
    nodes: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    score_by_node: dict[int, float],
    threshold: float,
) -> tuple[set[int], dict[str, object]]:
    components, outdegree = graph_components(nodes, edges)
    eligible = []
    for component in components:
        if not COMPONENT_MIN_NODES <= len(component) <= COMPONENT_MAX_NODES:
            continue
        if any(outdegree.get(node_id, 0) >= 2 for node_id in component):
            continue
        scores = np.asarray(
            [score_by_node[node_id] for node_id in component],
            dtype=np.float64,
        )
        median = float(np.median(scores))
        below_fraction = float(np.mean(scores < threshold))
        if (
            median < threshold
            and below_fraction >= MIN_BELOW_THRESHOLD_FRACTION
        ):
            eligible.append(
                {
                    "node_ids": component,
                    "nodes": len(component),
                    "median_probability": median,
                    "below_threshold_fraction": below_fraction,
                }
            )
    eligible.sort(
        key=lambda item: (
            float(item["median_probability"]),
            int(item["node_ids"][0]),
        )
    )
    cap = min(
        MAX_REMOVED_NODES,
        int(math.floor(len(nodes) * MAX_REMOVED_FRACTION)),
    )
    selected = []
    removed: set[int] = set()
    for item in eligible:
        node_ids = set(int(value) for value in item["node_ids"])
        if len(removed) + len(node_ids) > cap:
            continue
        removed.update(node_ids)
        selected.append(item)
    return removed, {
        "components": len(components),
        "eligible_components": len(eligible),
        "eligible_nodes": sum(int(item["nodes"]) for item in eligible),
        "selected_components": len(selected),
        "removed_nodes": len(removed),
        "node_cap": cap,
        "selected": selected,
    }


def audit_graph(
    nodes: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
) -> dict[str, int]:
    pairs: set[tuple[int, int]] = set()
    indegree: dict[int, int] = {}
    outdegree: dict[int, int] = {}
    duplicate_edges = 0
    dangling_edges = 0
    nonconsecutive_edges = 0
    for edge in edges:
        source = int(edge["source_id"])
        target = int(edge["target_id"])
        pair = (source, target)
        duplicate_edges += int(pair in pairs)
        pairs.add(pair)
        if source not in nodes or target not in nodes:
            dangling_edges += 1
            continue
        nonconsecutive_edges += int(
            int(nodes[target]["t"]) != int(nodes[source]["t"]) + 1
        )
        indegree[target] = indegree.get(target, 0) + 1
        outdegree[source] = outdegree.get(source, 0) + 1
    result = {
        "nodes": len(nodes),
        "edges": len(edges),
        "duplicate_edges": duplicate_edges,
        "dangling_edges": dangling_edges,
        "nonconsecutive_edges": nonconsecutive_edges,
        "max_indegree": max(indegree.values(), default=0),
        "max_outdegree": max(outdegree.values(), default=0),
        "nonbinary_sources": sum(value > 2 for value in outdegree.values()),
    }
    if (
        duplicate_edges
        or dangling_edges
        or nonconsecutive_edges
        or result["max_indegree"] > 1
        or result["max_outdegree"] > 2
        or result["nonbinary_sources"]
    ):
        raise RuntimeError(f"Filtered graph topology failed: {result}")
    return result


def write_graph_rows(
    writer: csv.DictWriter,
    row_id: int,
    dataset: str,
    nodes: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
) -> int:
    for node_id in sorted(nodes):
        node = nodes[node_id]
        writer.writerow(
            {
                "id": row_id,
                "dataset": dataset,
                "row_type": "node",
                "node_id": node_id,
                "t": int(node["t"]),
                "z": max(0, int(round(float(node["z"])))),
                "y": max(0, int(round(float(node["y"])))),
                "x": max(0, int(round(float(node["x"])))),
                "source_id": -1,
                "target_id": -1,
            }
        )
        row_id += 1
    for edge in edges:
        writer.writerow(
            {
                "id": row_id,
                "dataset": dataset,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": int(edge["source_id"]),
                "target_id": int(edge["target_id"]),
            }
        )
        row_id += 1
    return row_id


def json_model_summary(model_info: dict[str, object]) -> dict[str, object]:
    return {
        "seed": model_info["seed"],
        "training_movies": model_info["training_movies"],
        "device": model_info["device"],
        "counts": model_info["counts"],
        "epoch_losses": [
            float(value) for value in model_info["epoch_losses"]
        ],
        "feature_mean": np.asarray(
            model_info["mean"],
            dtype=np.float32,
        ).tolist(),
        "feature_std": np.asarray(
            model_info["std"],
            dtype=np.float32,
        ).tolist(),
    }


def save_model(
    path: Path,
    model_info: dict[str, object],
    threshold: float,
    source: str,
    target: str,
) -> str:
    import torch

    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "feature_names": FEATURE_NAMES,
            "state_dict": model_info["model"].state_dict(),
            "feature_mean": np.asarray(
                model_info["mean"],
                dtype=np.float32,
            ),
            "feature_std": np.asarray(
                model_info["std"],
                dtype=np.float32,
            ),
            "threshold": threshold,
            "source_embryo": source,
            "target_embryo": target,
        },
        path,
    )
    return file_sha256(path)


def convert_csv_to_geffs(
    args: argparse.Namespace,
    csv_path: Path,
    output_dir: Path,
) -> None:
    python_path = os.pathsep.join(
        [
            str(args.runtime_dir.resolve()),
            str((args.scorer_dir / "src").resolve()),
            str((args.scorer_dir / "scripts").resolve()),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    subprocess.run(
        [
            sys.executable,
            str(args.scorer_dir / "scripts" / "csv_to_geffs.py"),
            "--csv",
            str(csv_path),
            "--out-dir",
            str(output_dir),
            "--no-overwrite",
        ],
        env={**os.environ, "PYTHONPATH": python_path},
        check=True,
    )


def main() -> None:
    args = parse_args()
    baseline_paths = validate_args(args)
    started_at = time.time()
    args.output_dir.mkdir(parents=True)
    feature_dir = args.output_dir / "features"
    model_dir = args.output_dir / "models"
    feature_dir.mkdir()
    model_dir.mkdir()
    args.public_v40_parity = False
    namespace = load_notebook_namespace(args)
    scale = np.asarray(namespace["VOXEL_SCALE_UM"], dtype=np.float64)
    if scale.shape != (3,) or np.any(scale <= 0.0):
        raise RuntimeError(f"Invalid voxel scale: {scale}")

    cache_paths: dict[str, Path] = {}
    extraction = {}
    for index, baseline_path in enumerate(baseline_paths, start=1):
        cache_path = feature_dir / f"{baseline_path.stem}.npz"
        extraction[baseline_path.stem] = extract_dataset_features(
            namespace,
            baseline_path,
            args.ground_truth_dir,
            cache_path,
            scale,
        )
        cache_paths[baseline_path.stem] = cache_path
        print(
            f"[feature {index:02d}/{len(baseline_paths):02d}] "
            f"{baseline_path.stem}",
            flush=True,
        )

    names = sorted(cache_paths)
    target_models = {}
    for target, source in (("44b6", "6bba"), ("6bba", "44b6")):
        threshold, model_info, calibration = calibrate_source_embryo(
            source,
            cache_paths,
            names,
            args.device,
        )
        model_path = model_dir / f"{source}_to_{target}.pt"
        model_sha256 = save_model(
            model_path,
            model_info,
            threshold,
            source,
            target,
        )
        target_models[target] = {
            "source": source,
            "threshold": threshold,
            "model": model_info,
            "model_path": model_path,
            "model_sha256": model_sha256,
            "calibration": calibration,
        }
        print(
            f"Trained {source} -> {target}: threshold={threshold:.8f}",
            flush=True,
        )

    candidate_csv = args.output_dir / "candidate.csv"
    per_movie = {}
    row_id = 0
    total_removed = 0
    with candidate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for index, baseline_path in enumerate(baseline_paths, start=1):
            dataset = baseline_path.stem
            target = dataset.split("_", 1)[0]
            model_record = target_models[target]
            node_ids, features, _ = load_feature_cache(
                cache_paths[dataset]
            )
            scores = predict_probabilities(
                model_record["model"],
                features,
                args.device,
            )
            _, nodes, edges = graph_rows(namespace, baseline_path)
            if set(int(value) for value in node_ids) != set(nodes):
                raise RuntimeError(f"{dataset}: feature/node IDs changed")
            score_by_node = {
                int(node_id): float(score)
                for node_id, score in zip(node_ids, scores)
            }
            removed, filter_stats = select_removed_components(
                nodes,
                edges,
                score_by_node,
                float(model_record["threshold"]),
            )
            filtered_nodes = {
                node_id: node
                for node_id, node in nodes.items()
                if node_id not in removed
            }
            filtered_edges = [
                edge
                for edge in edges
                if (
                    int(edge["source_id"]) not in removed
                    and int(edge["target_id"]) not in removed
                )
            ]
            topology = audit_graph(filtered_nodes, filtered_edges)
            row_id = write_graph_rows(
                writer,
                row_id,
                dataset,
                filtered_nodes,
                filtered_edges,
            )
            total_removed += len(removed)
            per_movie[dataset] = {
                "target_embryo": target,
                "source_embryo": model_record["source"],
                "threshold": float(model_record["threshold"]),
                "raw_nodes": len(nodes),
                "raw_edges": len(edges),
                "score_minimum": float(scores.min()),
                "score_median": float(np.median(scores)),
                "score_maximum": float(scores.max()),
                **filter_stats,
                "topology": topology,
            }
            print(
                f"[filter {index:02d}/{len(baseline_paths):02d}] "
                f"{dataset}: removed={len(removed)}",
                flush=True,
            )

    candidate_dir = args.output_dir / "filtered_geffs"
    convert_csv_to_geffs(args, candidate_csv, candidate_dir)
    candidate_paths = sorted(candidate_dir.glob("*.geff"))
    if [path.stem for path in candidate_paths] != names:
        raise RuntimeError("Filtered GEFF movie set changed")

    model_manifest = {}
    for target, record in target_models.items():
        model_manifest[target] = {
            "target_embryo": target,
            "source_embryo": record["source"],
            "target_labels_used": False,
            "threshold": record["threshold"],
            "model_path": str(record["model_path"].resolve()),
            "model_sha256": record["model_sha256"],
            "calibration": record["calibration"],
            "final_model": json_model_summary(record["model"]),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "E028 cross-embryo PU appearance filter",
        "clean_only": True,
        "metric_exploit_used": False,
        "baseline_dir": str(args.baseline_dir.resolve()),
        "candidate_dir": str(candidate_dir.resolve()),
        "candidate_csv": str(candidate_csv.resolve()),
        "candidate_csv_sha256": file_sha256(candidate_csv),
        "notebook_sha256": file_sha256(args.notebook),
        "datasets": names,
        "movie_names_sha256": movie_names_sha256(names),
        "voxel_scale_um": scale.tolist(),
        "feature_names": FEATURE_NAMES,
        "matching": {
            "method": "maximum_cardinality_radius_matching",
            "radius_um": MATCH_RADIUS_UM,
            "matched_prediction_label": "positive",
            "other_prediction_label": "unlabeled",
            "declared_negative_samples": 0,
        },
        "training": {
            "algorithm": "non_negative_positive_unlabeled_linear",
            "positive_prior": POSITIVE_PRIOR,
            "oof_folds": OOF_FOLDS,
            "epochs": EPOCHS,
            "positive_batch_size": POSITIVE_BATCH_SIZE,
            "unlabeled_batch_size": UNLABELED_BATCH_SIZE,
            "max_unlabeled_per_positive": MAX_UNLABELED_PER_POSITIVE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "base_seed": BASE_SEED,
        },
        "filter_policy": {
            "component_min_nodes": COMPONENT_MIN_NODES,
            "component_max_nodes": COMPONENT_MAX_NODES,
            "division_components_allowed": False,
            "positive_probability_quantile": POSITIVE_QUANTILE,
            "median_below_threshold": True,
            "minimum_below_threshold_fraction": (
                MIN_BELOW_THRESHOLD_FRACTION
            ),
            "maximum_removed_fraction": MAX_REMOVED_FRACTION,
            "maximum_removed_nodes": MAX_REMOVED_NODES,
            "whole_components_only": True,
            "nodes_added": 0,
            "edges_added": 0,
        },
        "feature_extraction": extraction,
        "models": model_manifest,
        "per_movie": per_movie,
        "total_removed_nodes": total_removed,
        "elapsed_seconds": time.time() - started_at,
    }
    manifest_path = args.output_dir / "filter_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate_dir": str(candidate_dir),
                "total_removed_nodes": total_removed,
            }
        )
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
