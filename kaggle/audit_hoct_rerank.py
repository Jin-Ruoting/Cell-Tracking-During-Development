#!/usr/bin/env python3
"""Audit HOCT as a read-only local edge reranker for E000 detections.

The audit keeps E000 nodes and topology unchanged. It builds consecutive-frame
nearest-neighbour candidates around those nodes, extracts deterministic local
intensity features, runs the pinned HOCT ``general_v0`` TorchScript model, and
compares candidate rankings with sparse official lineage labels.

No prediction graph or submission file is written. The output is a JSON report
and a CSV of labeled-edge decisions that can falsify the reranking hypothesis
before any integration work is attempted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.spatial import cKDTree

from audit_deepcenter_rescue import (
    frame_nodes,
    maximum_radius_matching,
    points_um,
)
from validate_e006_postprocess import (
    apply_filter_config,
    file_sha256,
    graph_rows,
    load_notebook_namespace,
)


EXPECTED_MODEL_SHA256 = (
    "024c2e4606275c96667907abfc9e0c27487b543480caf99d9ebd1d267cef8e4a"
)
FEATURE_NAMES = (
    "t",
    "z",
    "y",
    "x",
    "equivalent_diameter_area",
    "intensity_min",
    "intensity_max",
    "intensity_mean",
    "intensity_std",
    "inertia_zz",
    "inertia_zy",
    "inertia_zx",
    "inertia_yz",
    "inertia_yy",
    "inertia_yx",
    "inertia_xz",
    "inertia_xy",
    "inertia_xx",
    "border_dist",
)
HOCT_MEAN = np.asarray(
    [
        4.6326e02,
        2.9380e00,
        3.5649e02,
        3.4491e02,
        1.1521e01,
        2.7600e-01,
        9.6600e-01,
        5.7400e-01,
        1.6200e-01,
        1.6781e02,
        -2.7000e-02,
        5.0000e-02,
        -2.7000e-02,
        8.7012e01,
        -1.4010e00,
        5.0000e-02,
        -1.4010e00,
        8.3695e01,
        9.0000e-03,
    ],
    dtype=np.float32,
)
HOCT_STD = np.asarray(
    [
        5.5578e02,
        7.6000e00,
        1.9588e02,
        2.2610e02,
        8.1990e00,
        2.1600e-01,
        2.8100e-01,
        1.9300e-01,
        6.9000e-02,
        6.7845e02,
        3.1670e00,
        2.8750e00,
        3.1670e00,
        5.1292e02,
        1.8274e02,
        2.8750e00,
        1.8274e02,
        3.0608e02,
        7.8000e-02,
    ],
    dtype=np.float32,
)
SHAPE_FEATURE_MEAN = HOCT_MEAN[4:5].tolist() + HOCT_MEAN[9:18].tolist()
DEFAULT_THRESHOLDS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--support-src", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--max-windows-per-dataset", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--max-distance-um", type=float, default=15.0)
    parser.add_argument("--patch-radius-um", type=float, default=4.0)
    parser.add_argument("--match-radius-um", type=float, default=7.0)
    parser.add_argument(
        "--position-mode",
        choices=("voxel", "physical"),
        default="voxel",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-nodes-per-window", type=int, default=4000)
    parser.add_argument("--max-edges-per-window", type=int, default=10000)
    parser.add_argument(
        "--replacement-thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument("--no-autocast", action="store_true")
    parser.add_argument("--allow-unpinned-model", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[float]:
    for path in (
        args.baseline_dir,
        args.image_dir,
        args.ground_truth_dir,
        args.runtime_dir,
        args.support_src,
    ):
        if not path.is_dir():
            raise NotADirectoryError(path)
    for path in (args.notebook, args.model):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    integer_values = {
        "max_datasets": args.max_datasets,
        "max_windows_per_dataset": args.max_windows_per_dataset,
    }
    if any(value < 0 for value in integer_values.values()):
        raise ValueError(f"Limits cannot be negative: {integer_values}")
    positive_integers = {
        "window_size": args.window_size,
        "neighbors": args.neighbors,
        "max_nodes_per_window": args.max_nodes_per_window,
        "max_edges_per_window": args.max_edges_per_window,
    }
    if any(value < 1 for value in positive_integers.values()):
        raise ValueError(
            f"Integer controls must be positive: {positive_integers}"
        )
    if args.window_size < 2:
        raise ValueError("--window-size must be at least 2")
    positive_floats = {
        "max_distance_um": args.max_distance_um,
        "patch_radius_um": args.patch_radius_um,
        "match_radius_um": args.match_radius_um,
    }
    if any(
        not math.isfinite(value) or value <= 0
        for value in positive_floats.values()
    ):
        raise ValueError(f"Distance controls must be positive: {positive_floats}")
    thresholds = sorted({float(value) for value in args.replacement_thresholds})
    if not thresholds or any(
        not math.isfinite(value) or value < 0 for value in thresholds
    ):
        raise ValueError("Replacement thresholds must be finite and nonnegative")
    model_sha = file_sha256(args.model)
    if not args.allow_unpinned_model and model_sha != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            {
                "model": str(args.model),
                "expected_sha256": EXPECTED_MODEL_SHA256,
                "actual_sha256": model_sha,
            }
        )
    return thresholds


def normalize_frame(volume: np.ndarray) -> np.ndarray:
    image = np.asarray(volume, dtype=np.float32)
    lower = float(image.min())
    upper = float(np.quantile(image, 0.999))
    return (image - lower) / (upper - lower + 1e-7)


def patch_intensity_features(
    image: np.ndarray,
    point: np.ndarray,
    scale: np.ndarray,
    radius_um: float,
) -> tuple[float, float, float, float]:
    radii = np.maximum(1, np.ceil(radius_um / scale).astype(np.int64))
    center = np.rint(point).astype(np.int64)
    starts = np.maximum(0, center - radii)
    stops = np.minimum(np.asarray(image.shape), center + radii + 1)
    slices = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(starts, stops, strict=True)
    )
    patch = image[slices]
    axes = [
        np.arange(int(start), int(stop), dtype=np.float32) - float(coordinate)
        for start, stop, coordinate in zip(
            starts,
            stops,
            point,
            strict=True,
        )
    ]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    mask = (
        (zz * scale[0]) ** 2
        + (yy * scale[1]) ** 2
        + (xx * scale[2]) ** 2
        <= radius_um**2
    )
    values = patch[mask]
    if values.size == 0:
        nearest = np.clip(center, 0, np.asarray(image.shape) - 1)
        value = float(image[tuple(nearest)])
        return value, value, value, 0.0
    return (
        float(values.min()),
        float(values.max()),
        float(values.mean()),
        float(values.std()),
    )


def border_feature(point: np.ndarray, shape: tuple[int, int, int]) -> float:
    distance = float(
        np.minimum(point, np.asarray(shape, dtype=np.float64) - point).min()
    )
    return 1.0 - min(1.0, distance / 5.0)


def extract_node_features(
    namespace: dict[str, object],
    dataset: str,
    nodes: dict[int, dict[str, object]],
    node_ids_by_frame: dict[int, list[int]],
    frames: Iterable[int],
    scale: np.ndarray,
    patch_radius_um: float,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    features: dict[int, np.ndarray] = {}
    raw_rows: list[np.ndarray] = []
    frame_cache: dict[int, np.ndarray] = {}
    selected_frames = sorted(set(frames))
    for frame_index, t in enumerate(selected_frames, start=1):
        node_ids = node_ids_by_frame.get(t, [])
        if not node_ids:
            continue
        volume = np.asarray(
            namespace["read_test_frame"](
                dataset,
                int(t),
                frame_cache,
            )
        )
        normalized = normalize_frame(volume)
        shape = tuple(int(value) for value in normalized.shape)
        for node_id in node_ids:
            node = nodes[node_id]
            point = np.asarray(
                [node["z"], node["y"], node["x"]],
                dtype=np.float64,
            )
            intensity = patch_intensity_features(
                normalized,
                point,
                scale,
                patch_radius_um,
            )
            raw = np.asarray(
                [
                    float(node["t"]),
                    *point.tolist(),
                    SHAPE_FEATURE_MEAN[0],
                    *intensity,
                    *SHAPE_FEATURE_MEAN[1:],
                    border_feature(point, shape),
                ],
                dtype=np.float32,
            )
            if raw.shape != (len(FEATURE_NAMES),):
                raise AssertionError((node_id, raw.shape))
            standardized = (raw - HOCT_MEAN) / np.maximum(HOCT_STD, 1e-7)
            if not np.isfinite(standardized).all():
                raise FloatingPointError(f"{dataset}: nonfinite feature {node_id}")
            features[node_id] = standardized
            raw_rows.append(raw)
        frame_cache.pop(int(t), None)
        if frame_index % 25 == 0:
            print(
                f"{dataset}: extracted features for "
                f"{frame_index}/{len(selected_frames)} frames",
                flush=True,
            )
    raw_matrix = np.stack(raw_rows)
    standardized_matrix = (raw_matrix - HOCT_MEAN) / HOCT_STD
    stats = {
        "nodes": int(len(raw_matrix)),
        "raw_mean": {
            name: float(value)
            for name, value in zip(
                FEATURE_NAMES,
                raw_matrix.mean(axis=0),
                strict=True,
            )
        },
        "standardized_mean": {
            name: float(value)
            for name, value in zip(
                FEATURE_NAMES,
                standardized_matrix.mean(axis=0),
                strict=True,
            )
        },
        "standardized_std": {
            name: float(value)
            for name, value in zip(
                FEATURE_NAMES,
                standardized_matrix.std(axis=0),
                strict=True,
            )
        },
    }
    return features, stats


def match_gt_nodes(
    nodes: dict[int, dict[str, object]],
    gt_nodes: dict[int, dict[str, object]],
    scale: np.ndarray,
    radius_um: float,
) -> tuple[dict[int, int], dict[str, int]]:
    by_frame = frame_nodes(nodes)
    gt_by_frame = frame_nodes(gt_nodes)
    gt_to_pred: dict[int, int] = {}
    labeled_nodes = 0
    for t, gt_ids in sorted(gt_by_frame.items()):
        pred_ids = by_frame.get(t, [])
        matching = maximum_radius_matching(
            points_um(pred_ids, nodes, scale),
            points_um(gt_ids, gt_nodes, scale),
            radius_um,
        )
        labeled_nodes += len(gt_ids)
        for pred_idx, gt_idx in enumerate(matching):
            if gt_idx >= 0:
                gt_to_pred[int(gt_ids[int(gt_idx)])] = int(pred_ids[pred_idx])
    return gt_to_pred, {
        "labeled_nodes": labeled_nodes,
        "matched_labeled_nodes": len(gt_to_pred),
    }


def select_window_starts(
    nodes_by_frame: dict[int, list[int]],
    gt_edges: list[dict[str, object]],
    gt_nodes: dict[int, dict[str, object]],
    window_size: int,
    limit: int,
) -> list[int]:
    time_points = sorted(nodes_by_frame)
    if not time_points:
        return []
    min_time, max_time = time_points[0], time_points[-1]
    possible = list(range(min_time, max_time - window_size + 2))
    if not possible:
        raise RuntimeError(
            f"Time span {min_time}:{max_time} is shorter than window "
            f"size {window_size}"
        )
    labeled_target_times = {
        int(gt_nodes[int(edge["target_id"])]["t"])
        for edge in gt_edges
        if int(edge["target_id"]) in gt_nodes
    }
    relevant = [
        start
        for start in possible
        if any(
            start < target_time < start + window_size
            for target_time in labeled_target_times
        )
    ]
    if not relevant:
        return []
    if not limit or len(relevant) <= limit:
        return relevant
    indices = np.rint(np.linspace(0, len(relevant) - 1, limit)).astype(int)
    return sorted({relevant[int(index)] for index in indices})


def build_candidates(
    nodes: dict[int, dict[str, object]],
    nodes_by_frame: dict[int, list[int]],
    baseline_edges: list[dict[str, object]],
    interval_times: set[int],
    scale: np.ndarray,
    neighbors: int,
    max_distance_um: float,
) -> tuple[
    list[tuple[int, int]],
    dict[int, list[tuple[int, int]]],
    dict[str, int],
]:
    baseline_pairs = {
        (int(edge["source_id"]), int(edge["target_id"]))
        for edge in baseline_edges
    }
    pairs: set[tuple[int, int]] = set()
    nearest_pairs = 0
    retained_baseline_pairs = 0
    for target_time in sorted(interval_times):
        source_ids = nodes_by_frame.get(target_time - 1, [])
        target_ids = nodes_by_frame.get(target_time, [])
        if not source_ids or not target_ids:
            continue
        source_points = points_um(source_ids, nodes, scale)
        target_points = points_um(target_ids, nodes, scale)
        tree = cKDTree(source_points)
        k = min(int(neighbors), len(source_ids))
        distances, indices = tree.query(
            target_points,
            k=k,
            distance_upper_bound=float(max_distance_um),
        )
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        for target_idx, target_id in enumerate(target_ids):
            for distance, source_idx in zip(
                distances[target_idx],
                indices[target_idx],
                strict=True,
            ):
                if not math.isfinite(float(distance)):
                    continue
                pair = (int(source_ids[int(source_idx)]), int(target_id))
                if pair not in pairs:
                    nearest_pairs += 1
                pairs.add(pair)
    for source_id, target_id in baseline_pairs:
        source = nodes.get(source_id)
        target = nodes.get(target_id)
        if source is None or target is None:
            continue
        target_time = int(target["t"])
        if (
            target_time in interval_times
            and int(source["t"]) == target_time - 1
            and (source_id, target_id) not in pairs
        ):
            pairs.add((source_id, target_id))
            retained_baseline_pairs += 1
    ordered = sorted(
        pairs,
        key=lambda pair: (
            int(nodes[pair[1]]["t"]),
            pair[1],
            pair[0],
        ),
    )
    by_target: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pair in ordered:
        by_target[pair[1]].append(pair)
    return ordered, dict(by_target), {
        "candidate_edges": len(ordered),
        "nearest_candidate_edges": nearest_pairs,
        "retained_baseline_edges_outside_nearest_pool": retained_baseline_pairs,
        "candidate_targets": len(by_target),
    }


def node_position(
    node: dict[str, object],
    scale: np.ndarray,
    mode: str,
) -> np.ndarray:
    position = np.asarray(
        [node["z"], node["y"], node["x"]],
        dtype=np.float32,
    )
    if mode == "physical":
        position *= scale.astype(np.float32)
    return position


def infer_candidate_scores(
    model: torch.jit.ScriptModule,
    device: torch.device,
    nodes: dict[int, dict[str, object]],
    nodes_by_frame: dict[int, list[int]],
    features: dict[int, np.ndarray],
    candidate_by_target: dict[int, list[tuple[int, int]]],
    starts: list[int],
    window_size: int,
    scale: np.ndarray,
    position_mode: str,
    max_nodes: int,
    max_edges: int,
    autocast: bool,
) -> tuple[dict[tuple[int, int], float], dict[str, object]]:
    observations: dict[tuple[int, int], list[float]] = defaultdict(list)
    window_rows: list[dict[str, object]] = []
    candidates_by_time: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for target_id, pairs in candidate_by_target.items():
        candidates_by_time[int(nodes[target_id]["t"])].extend(pairs)
    started = time.monotonic()
    for window_index, start in enumerate(starts, start=1):
        stop = start + window_size
        node_ids = [
            node_id
            for t in range(start, stop)
            for node_id in nodes_by_frame.get(t, [])
        ]
        local_index = {
            node_id: index for index, node_id in enumerate(node_ids)
        }
        pairs = [
            pair
            for t in range(start + 1, stop)
            for pair in candidates_by_time.get(t, [])
        ]
        if not pairs:
            continue
        if len(node_ids) > max_nodes or len(pairs) > max_edges:
            raise RuntimeError(
                {
                    "window_start": start,
                    "nodes": len(node_ids),
                    "edges": len(pairs),
                    "max_nodes": max_nodes,
                    "max_edges": max_edges,
                }
            )
        node_features = np.stack([features[node_id] for node_id in node_ids])
        positions = np.stack(
            [
                node_position(nodes[node_id], scale, position_mode)
                for node_id in node_ids
            ]
        )
        edge_indices = np.asarray(
            [
                [local_index[source_id], local_index[target_id]]
                for source_id, target_id in pairs
            ],
            dtype=np.int64,
        )
        edge_positions = (
            positions[edge_indices[:, 0]] + positions[edge_indices[:, 1]]
        ) * 0.5
        tensors = {
            "node_features": torch.from_numpy(node_features)[None].to(device),
            "node_pos": torch.from_numpy(positions)[None].to(device),
            "edge_pos": torch.from_numpy(edge_positions)[None].to(device),
            "edge_indices": torch.from_numpy(edge_indices)[None].to(device),
            "node_mask": torch.ones(
                (1, len(node_ids)),
                dtype=torch.bool,
                device=device,
            ),
            "edge_mask": torch.ones(
                (1, len(pairs)),
                dtype=torch.bool,
                device=device,
            ),
        }
        if autocast and device.type == "cuda":
            context = torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )
        else:
            context = nullcontext()
        with torch.inference_mode(), context:
            output = model.forward(
                tensors["node_features"],
                tensors["node_pos"],
                tensors["edge_pos"],
                tensors["edge_indices"],
                tensors["node_mask"],
                tensors["edge_mask"],
            )
            logits = output[0].float().reshape(-1).cpu().numpy()
        if logits.shape != (len(pairs),):
            raise RuntimeError(
                f"Unexpected logits shape {logits.shape} for {len(pairs)} edges"
            )
        if not np.isfinite(logits).all():
            raise FloatingPointError(f"Nonfinite HOCT logits at window {start}")
        exponentials = np.exp(np.minimum(logits, 20.0))
        for pair, value in zip(pairs, exponentials, strict=True):
            observations[pair].append(float(value))
        window_rows.append(
            {
                "start": start,
                "nodes": len(node_ids),
                "edges": len(pairs),
                "logit_min": float(logits.min()),
                "logit_max": float(logits.max()),
                "logit_mean": float(logits.mean()),
            }
        )
        print(
            f"HOCT window {window_index}/{len(starts)} start={start} "
            f"nodes={len(node_ids)} edges={len(pairs)}",
            flush=True,
        )
    median_exp = {
        pair: float(np.median(values))
        for pair, values in observations.items()
    }
    denominator: dict[int, float] = defaultdict(lambda: 1.0)
    for (_, target_id), value in median_exp.items():
        denominator[target_id] += value
    similarities = {
        pair: value / denominator[pair[1]]
        for pair, value in median_exp.items()
    }
    return similarities, {
        "windows": window_rows,
        "scored_edges": len(similarities),
        "observation_count": sum(len(values) for values in observations.values()),
        "seconds": time.monotonic() - started,
    }


def physical_edge_distance(
    nodes: dict[int, dict[str, object]],
    pair: tuple[int, int],
    scale: np.ndarray,
) -> float:
    source = np.asarray(
        [nodes[pair[0]][axis] for axis in ("z", "y", "x")],
        dtype=np.float64,
    )
    target = np.asarray(
        [nodes[pair[1]][axis] for axis in ("z", "y", "x")],
        dtype=np.float64,
    )
    return float(np.linalg.norm((target - source) * scale))


def evaluate_labeled_edges(
    dataset: str,
    nodes: dict[int, dict[str, object]],
    baseline_edges: list[dict[str, object]],
    gt_nodes: dict[int, dict[str, object]],
    gt_edges: list[dict[str, object]],
    gt_to_pred: dict[int, int],
    candidate_by_target: dict[int, list[tuple[int, int]]],
    similarities: dict[tuple[int, int], float],
    evaluated_target_times: set[int],
    scale: np.ndarray,
) -> list[dict[str, object]]:
    baseline_pairs = {
        (int(edge["source_id"]), int(edge["target_id"]))
        for edge in baseline_edges
    }
    baseline_parents: dict[int, list[int]] = defaultdict(list)
    for source_id, target_id in baseline_pairs:
        baseline_parents[target_id].append(source_id)
    rows: list[dict[str, object]] = []
    for edge in gt_edges:
        gt_source = int(edge["source_id"])
        gt_target = int(edge["target_id"])
        pred_source = gt_to_pred.get(gt_source)
        pred_target = gt_to_pred.get(gt_target)
        if pred_source is None or pred_target is None:
            continue
        target_time = int(gt_nodes[gt_target]["t"])
        if target_time not in evaluated_target_times:
            continue
        candidates = [
            pair
            for pair in candidate_by_target.get(pred_target, [])
            if pair in similarities
        ]
        true_pair = (pred_source, pred_target)
        ranked = (
            sorted(
                candidates,
                key=lambda pair: (similarities[pair], -pair[0]),
                reverse=True,
            )
            if candidates
            else []
        )
        top_pair = ranked[0] if ranked else None
        nearest_pair = (
            min(
                candidates,
                key=lambda pair: (
                    physical_edge_distance(nodes, pair, scale),
                    pair[0],
                ),
            )
            if candidates
            else None
        )
        true_rank = (
            ranked.index(true_pair) + 1 if true_pair in ranked else None
        )
        parents = sorted(baseline_parents.get(pred_target, []))
        baseline_source = parents[0] if len(parents) == 1 else None
        baseline_pair = (
            (baseline_source, pred_target)
            if baseline_source is not None
            else None
        )
        baseline_score = (
            similarities.get(baseline_pair)
            if baseline_pair is not None
            else None
        )
        margin = (
            similarities[top_pair] - baseline_score
            if baseline_score is not None and top_pair is not None
            else None
        )
        rows.append(
            {
                "dataset": dataset,
                "embryo": dataset.split("_", maxsplit=1)[0],
                "target_t": target_time,
                "gt_source_id": gt_source,
                "gt_target_id": gt_target,
                "pred_source_id": pred_source,
                "pred_target_id": pred_target,
                "candidate_count": len(candidates),
                "true_candidate": int(true_pair in similarities),
                "true_rank": true_rank,
                "true_score": similarities.get(true_pair),
                "top_source_id": top_pair[0] if top_pair is not None else None,
                "top_score": (
                    similarities[top_pair] if top_pair is not None else None
                ),
                "top_correct": int(top_pair == true_pair),
                "nearest_source_id": (
                    nearest_pair[0] if nearest_pair is not None else None
                ),
                "nearest_correct": int(nearest_pair == true_pair),
                "baseline_source_id": baseline_source,
                "baseline_score": baseline_score,
                "baseline_correct": int(true_pair in baseline_pairs),
                "top_differs_from_baseline": int(
                    baseline_pair is not None
                    and top_pair is not None
                    and top_pair != baseline_pair
                ),
                "top_minus_baseline": margin,
            }
        )
    return rows


def summarize_rows(
    rows: list[dict[str, object]],
    thresholds: list[float],
) -> dict[str, object]:
    count = len(rows)
    covered = [row for row in rows if int(row["true_candidate"])]
    baseline_correct = sum(int(row["baseline_correct"]) for row in rows)
    top_correct = sum(int(row["top_correct"]) for row in rows)
    nearest_correct = sum(int(row["nearest_correct"]) for row in rows)
    rescues = sum(
        not int(row["baseline_correct"]) and int(row["top_correct"])
        for row in rows
    )
    harms = sum(
        int(row["baseline_correct"]) and not int(row["top_correct"])
        for row in rows
    )
    reciprocal_ranks = [
        1.0 / int(row["true_rank"])
        for row in covered
        if row["true_rank"] is not None
    ]
    policies: dict[str, object] = {}
    eligible = [
        row
        for row in rows
        if row["baseline_source_id"] is not None
        and row["baseline_score"] is not None
        and row["top_minus_baseline"] is not None
    ]
    for threshold in thresholds:
        correct = 0
        replacements = 0
        policy_rescues = 0
        policy_harms = 0
        for row in eligible:
            replace = (
                bool(row["top_differs_from_baseline"])
                and float(row["top_minus_baseline"]) >= threshold
            )
            selected_correct = (
                bool(row["top_correct"])
                if replace
                else bool(row["baseline_correct"])
            )
            correct += int(selected_correct)
            replacements += int(replace)
            policy_rescues += int(
                replace
                and not bool(row["baseline_correct"])
                and bool(row["top_correct"])
            )
            policy_harms += int(
                replace
                and bool(row["baseline_correct"])
                and not bool(row["top_correct"])
            )
        policies[f"{threshold:.6g}"] = {
            "eligible_edges": len(eligible),
            "replacements": replacements,
            "correct": correct,
            "accuracy": correct / len(eligible) if eligible else None,
            "rescues": policy_rescues,
            "harms": policy_harms,
            "net_rescues": policy_rescues - policy_harms,
        }
    return {
        "evaluated_matched_gt_edges": count,
        "candidate_covered_gt_edges": len(covered),
        "candidate_coverage": len(covered) / count if count else None,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / count if count else None,
        "hoct_top_correct": top_correct,
        "hoct_top_accuracy": top_correct / count if count else None,
        "nearest_correct": nearest_correct,
        "nearest_accuracy": nearest_correct / count if count else None,
        "hoct_top_rescues": rescues,
        "hoct_top_harms": harms,
        "hoct_top_net_rescues": rescues - harms,
        "mean_reciprocal_rank_when_covered": (
            float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None
        ),
        "replacement_policies": policies,
    }


def write_decisions(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    thresholds = validate_args(args)
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "hoct_rerank_audit.json"
    decisions_path = args.output_dir / "labeled_edge_decisions.csv"

    np.random.seed(0)
    torch.manual_seed(0)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")
    device = torch.device(args.device)
    model = torch.jit.load(str(args.model), map_location=device).to(device)
    model.eval()
    input_width = int(model.input_proj.weight.shape[1])
    if input_width != len(FEATURE_NAMES):
        raise RuntimeError(
            f"HOCT expects {input_width} features, audit provides "
            f"{len(FEATURE_NAMES)}"
        )

    namespace = load_notebook_namespace(args)
    scale = np.asarray(namespace["VOXEL_SCALE_UM"], dtype=np.float64)
    baseline_paths = sorted(args.baseline_dir.glob("*.geff"))
    if args.dataset:
        requested = set(args.dataset)
        baseline_paths = [
            path for path in baseline_paths if path.stem in requested
        ]
        missing = requested - {path.stem for path in baseline_paths}
        if missing:
            raise FileNotFoundError(f"Missing requested datasets: {sorted(missing)}")
    if args.max_datasets:
        baseline_paths = baseline_paths[: args.max_datasets]
    if not baseline_paths:
        raise FileNotFoundError(f"No baseline GEFFs in {args.baseline_dir}")

    report: dict[str, object] = {
        "config": {
            "notebook": str(args.notebook.resolve()),
            "notebook_sha256": file_sha256(args.notebook),
            "model": str(args.model.resolve()),
            "model_sha256": file_sha256(args.model),
            "expected_model_sha256": EXPECTED_MODEL_SHA256,
            "model_input_width": input_width,
            "device": str(device),
            "torch_version": torch.__version__,
            "window_size": args.window_size,
            "neighbors": args.neighbors,
            "max_distance_um": args.max_distance_um,
            "patch_radius_um": args.patch_radius_um,
            "match_radius_um": args.match_radius_um,
            "position_mode": args.position_mode,
            "max_windows_per_dataset": args.max_windows_per_dataset,
            "autocast": not args.no_autocast,
            "voxel_scale_um": scale.tolist(),
            "replacement_thresholds": thresholds,
            "datasets": [path.stem for path in baseline_paths],
        },
        "hypothesis": (
            "Pinned HOCT geometry and local intensity context can rank the "
            "matched sparse-GT parent above E000's selected parent often "
            "enough to yield positive net edge rescues."
        ),
        "notes": [
            "The audit does not modify or write a prediction graph.",
            "Official labels are sparse; only edges with both endpoints matched "
            "to E000 nodes are evaluated.",
            "Constant shape features use the published HOCT training mean; "
            "intensity features come from a deterministic physical ellipsoid.",
            "Promotion requires positive net rescues across both embryo groups "
            "and later improvement under the pinned official scorer.",
        ],
        "dataset_results": {},
        "aggregates": {},
    }
    all_rows: list[dict[str, object]] = []
    for dataset_index, baseline_path in enumerate(baseline_paths, start=1):
        dataset = baseline_path.stem
        print(
            f"[{dataset_index:02d}/{len(baseline_paths):02d}] {dataset}",
            flush=True,
        )
        gt_path = args.ground_truth_dir / f"{dataset}.geff"
        if not gt_path.is_dir():
            raise FileNotFoundError(gt_path)
        _, raw_nodes, raw_edges = graph_rows(namespace, baseline_path)
        apply_filter_config(
            namespace,
            (True, False, False, False, 0.20, 0.12),
        )
        nodes, baseline_edges, filter_stats = namespace[
            "filter_output_graph"
        ](
            raw_nodes,
            raw_edges,
            dataset=dataset,
            deepcenter_bundle=None,
        )
        _, gt_nodes, gt_edges = graph_rows(namespace, gt_path)
        nodes_by_frame = frame_nodes(nodes)
        gt_to_pred, match_stats = match_gt_nodes(
            nodes,
            gt_nodes,
            scale,
            args.match_radius_um,
        )
        starts = select_window_starts(
            nodes_by_frame,
            gt_edges,
            gt_nodes,
            args.window_size,
            args.max_windows_per_dataset,
        )
        if not starts:
            raise RuntimeError(f"{dataset}: no labeled HOCT windows selected")
        required_frames = {
            t
            for start in starts
            for t in range(start, start + args.window_size)
        }
        interval_times = {
            t
            for start in starts
            for t in range(start + 1, start + args.window_size)
        }
        candidates, candidates_by_target, candidate_stats = build_candidates(
            nodes,
            nodes_by_frame,
            baseline_edges,
            interval_times,
            scale,
            args.neighbors,
            args.max_distance_um,
        )
        features, feature_stats = extract_node_features(
            namespace,
            dataset,
            nodes,
            nodes_by_frame,
            required_frames,
            scale,
            args.patch_radius_um,
        )
        similarities, inference_stats = infer_candidate_scores(
            model,
            device,
            nodes,
            nodes_by_frame,
            features,
            candidates_by_target,
            starts,
            args.window_size,
            scale,
            args.position_mode,
            args.max_nodes_per_window,
            args.max_edges_per_window,
            not args.no_autocast,
        )
        if set(similarities) - set(candidates):
            raise AssertionError(f"{dataset}: scored an unknown candidate")
        rows = evaluate_labeled_edges(
            dataset,
            nodes,
            baseline_edges,
            gt_nodes,
            gt_edges,
            gt_to_pred,
            candidates_by_target,
            similarities,
            interval_times,
            scale,
        )
        all_rows.extend(rows)
        result = {
            "raw_nodes": len(raw_nodes),
            "raw_edges": len(raw_edges),
            "e000_nodes": len(nodes),
            "e000_edges": len(baseline_edges),
            "e000_filter_stats": {
                "gap_added_nodes": int(filter_stats.get("gap_added_nodes", 0)),
                "safe_divisions_added": int(
                    filter_stats.get("safe_divisions_added", 0)
                ),
            },
            "selected_window_starts": starts,
            "required_frames": len(required_frames),
            "matching": match_stats,
            "candidates": candidate_stats,
            "features": feature_stats,
            "inference": inference_stats,
            "labeled_edge_summary": summarize_rows(rows, thresholds),
        }
        report["dataset_results"][dataset] = result
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_decisions(decisions_path, all_rows)
        summary = result["labeled_edge_summary"]
        print(
            f"{dataset}: baseline={summary['baseline_accuracy']} "
            f"hoct={summary['hoct_top_accuracy']} "
            f"net={summary['hoct_top_net_rescues']}",
            flush=True,
        )

    report["aggregates"]["all"] = summarize_rows(all_rows, thresholds)
    for embryo in sorted({str(row["embryo"]) for row in all_rows}):
        report["aggregates"][embryo] = summarize_rows(
            [row for row in all_rows if row["embryo"] == embryo],
            thresholds,
        )
    report["outputs"] = {
        "decisions_csv": str(decisions_path),
        "decisions_sha256": file_sha256(decisions_path),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")
    print(f"Wrote {decisions_path}")


if __name__ == "__main__":
    main()
