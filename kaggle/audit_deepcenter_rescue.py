#!/usr/bin/env python3
"""Audit whether DeepCenter peaks complement the E000 detector.

This diagnostic loads the committed Notebook as the source of E000
postprocessing and DeepCenter inference. It never changes a prediction graph.
For each complete-ground-truth frame it:

* matches E000 nodes to ground truth with the official 7 um radius;
* extracts and physically suppresses DeepCenter local maxima;
* removes peaks already covered by E000 nodes; and
* measures whether the remaining peaks recover unmatched ground-truth nodes.

Three prediction-only candidate pools are reported: every novel peak, peaks
near an E000 track endpoint or start, and peaks bridging both. The report also
marks whether each clip appeared in the DeepCenter training or validation
manifest so in-sample diagnostics cannot be mistaken for independent evidence.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

from validate_e006_postprocess import (
    apply_filter_config,
    file_sha256,
    graph_rows,
    load_deepcenter_bundle,
    load_notebook_namespace,
)


SELECTIONS = ("novel", "endpoint_any", "bridge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--support-src", type=Path, required=True)
    parser.add_argument("--deepcenter-checkpoint", type=Path, required=True)
    parser.add_argument("--deepcenter-split-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60),
    )
    parser.add_argument("--match-radius-um", type=float, default=7.0)
    parser.add_argument("--existing-gate-um", type=float, default=7.0)
    parser.add_argument("--topology-radius-um", type=float, default=7.0)
    parser.add_argument("--nms-radius-um", type=float, default=4.2)
    parser.add_argument("--peak-min-distance", type=int, default=1)
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--max-frames-per-dataset", type=int, default=0)
    parser.add_argument("--skip-centroid-refinement", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[float]:
    required_dirs = (
        args.baseline_dir,
        args.image_dir,
        args.ground_truth_dir,
        args.runtime_dir,
        args.support_src,
    )
    for path in required_dirs:
        if not path.is_dir():
            raise NotADirectoryError(path)
    for path in (args.notebook, args.deepcenter_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        args.deepcenter_split_manifest is not None
        and not args.deepcenter_split_manifest.is_file()
    ):
        raise FileNotFoundError(args.deepcenter_split_manifest)
    thresholds = sorted({float(value) for value in args.thresholds})
    if not thresholds or thresholds[0] < 0.0 or thresholds[-1] > 1.0:
        raise ValueError("Thresholds must be unique values in [0, 1]")
    positive_values = {
        "match_radius_um": args.match_radius_um,
        "existing_gate_um": args.existing_gate_um,
        "topology_radius_um": args.topology_radius_um,
        "nms_radius_um": args.nms_radius_um,
    }
    for name, value in positive_values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.peak_min_distance < 1:
        raise ValueError("--peak-min-distance must be at least 1")
    if args.max_datasets < 0 or args.max_frames_per_dataset < 0:
        raise ValueError("Dataset and frame limits cannot be negative")
    return thresholds


def frame_nodes(
    nodes: dict[int, dict[str, object]],
) -> dict[int, list[int]]:
    by_frame: dict[int, list[int]] = defaultdict(list)
    for node_id, node in nodes.items():
        by_frame[int(node["t"])].append(node_id)
    for ids in by_frame.values():
        ids.sort()
    return dict(by_frame)


def points_um(
    node_ids: Iterable[int],
    nodes: dict[int, dict[str, object]],
    scale: np.ndarray,
) -> np.ndarray:
    rows = [
        [
            float(nodes[node_id]["z"]),
            float(nodes[node_id]["y"]),
            float(nodes[node_id]["x"]),
        ]
        for node_id in node_ids
    ]
    if not rows:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64) * scale[None, :]


def maximum_radius_matching(
    left_points_um: np.ndarray,
    right_points_um: np.ndarray,
    radius_um: float,
) -> np.ndarray:
    """Return right indices for a maximum-cardinality radius matching."""
    result = np.full(len(left_points_um), -1, dtype=np.int64)
    if len(left_points_um) == 0 or len(right_points_um) == 0:
        return result
    tree = cKDTree(right_points_um)
    rows: list[int] = []
    columns: list[int] = []
    for left_idx, neighbours in enumerate(
        tree.query_ball_point(left_points_um, float(radius_um))
    ):
        rows.extend([left_idx] * len(neighbours))
        columns.extend(int(idx) for idx in neighbours)
    if not rows:
        return result
    adjacency = csr_matrix(
        (
            np.ones(len(rows), dtype=np.int8),
            (np.asarray(rows), np.asarray(columns)),
        ),
        shape=(len(left_points_um), len(right_points_um)),
    )
    matching = maximum_bipartite_matching(adjacency, perm_type="column")
    if len(matching) != len(left_points_um):
        raise RuntimeError("Unexpected SciPy bipartite matching shape")
    return matching.astype(np.int64, copy=False)


def nearest_distance(
    point_um: np.ndarray,
    tree: cKDTree | None,
) -> float:
    if tree is None:
        return float("inf")
    distance, _ = tree.query(point_um, k=1)
    return float(distance)


def refine_centroid(volume: np.ndarray, point: np.ndarray) -> np.ndarray:
    zc, yc, xc = [int(round(float(value))) for value in point]
    z0, z1 = max(0, zc - 2), min(volume.shape[0], zc + 3)
    y0, y1 = max(0, yc - 5), min(volume.shape[1], yc + 6)
    x0, x1 = max(0, xc - 5), min(volume.shape[2], xc + 6)
    patch = volume[z0:z1, y0:y1, x0:x1].astype(
        np.float32,
        copy=False,
    )
    if patch.size == 0:
        return point.astype(np.float64, copy=False)
    background = float(np.percentile(patch, 20.0))
    weights = np.maximum(patch - background, 0.0)
    total = float(weights.sum())
    if total <= 1e-6:
        local = np.unravel_index(int(np.argmax(patch)), patch.shape)
        return np.asarray(
            [z0 + local[0], y0 + local[1], x0 + local[2]],
            dtype=np.float64,
        )
    zz, yy, xx = np.indices(patch.shape, dtype=np.float32)
    return np.asarray(
        [
            z0 + float((zz * weights).sum() / total),
            y0 + float((yy * weights).sum() / total),
            x0 + float((xx * weights).sum() / total),
        ],
        dtype=np.float64,
    )


def physical_nms(
    coordinates: np.ndarray,
    scores: np.ndarray,
    scale: np.ndarray,
    radius_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(coordinates) == 0:
        return coordinates, scores
    physical = coordinates * scale[None, :]
    tree = cKDTree(physical)
    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(len(coordinates), dtype=bool)
    keep: list[int] = []
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        neighbours = tree.query_ball_point(physical[idx], float(radius_um))
        suppressed[np.asarray(neighbours, dtype=np.int64)] = True
    keep_array = np.asarray(keep, dtype=np.int64)
    return coordinates[keep_array], scores[keep_array]


def extract_peaks(
    heatmap: np.ndarray,
    volume: np.ndarray,
    pool_factor: int,
    min_threshold: float,
    min_distance: int,
    scale: np.ndarray,
    nms_radius_um: float,
    refine: bool,
) -> tuple[np.ndarray, np.ndarray]:
    size = 2 * int(min_distance) + 1
    local_max = maximum_filter(heatmap, size=size, mode="nearest")
    peak_indices = np.argwhere(
        (heatmap == local_max) & (heatmap >= float(min_threshold))
    )
    if len(peak_indices) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    scores = heatmap[
        peak_indices[:, 0],
        peak_indices[:, 1],
        peak_indices[:, 2],
    ].astype(np.float64)
    coordinates = peak_indices.astype(np.float64)
    coordinates[:, 1] = (
        coordinates[:, 1] * pool_factor + (pool_factor - 1) / 2.0
    )
    coordinates[:, 2] = (
        coordinates[:, 2] * pool_factor + (pool_factor - 1) / 2.0
    )
    if refine:
        coordinates = np.asarray(
            [refine_centroid(volume, point) for point in coordinates],
            dtype=np.float64,
        )
    coordinates[:, 0] = np.clip(
        coordinates[:, 0],
        0,
        volume.shape[0] - 1,
    )
    coordinates[:, 1] = np.clip(
        coordinates[:, 1],
        0,
        volume.shape[1] - 1,
    )
    coordinates[:, 2] = np.clip(
        coordinates[:, 2],
        0,
        volume.shape[2] - 1,
    )
    return physical_nms(
        coordinates,
        scores,
        scale,
        nms_radius_um,
    )


def greedy_candidate_matching(
    candidates: list[dict[str, object]],
    candidate_indices: list[int],
    gt_ids: list[int],
    gt_points_um: np.ndarray,
    radius_um: float,
) -> dict[int, int]:
    """Match high-score candidates first, then choose the nearest free GT."""
    if not candidate_indices or not gt_ids:
        return {}
    tree = cKDTree(gt_points_um)
    used_gt_indices: set[int] = set()
    result: dict[int, int] = {}
    order = sorted(
        candidate_indices,
        key=lambda idx: (
            -float(candidates[idx]["score"]),
            float(candidates[idx]["z"]),
            float(candidates[idx]["y"]),
            float(candidates[idx]["x"]),
        ),
    )
    for candidate_idx in order:
        point_um = np.asarray(
            candidates[candidate_idx]["point_um"],
            dtype=np.float64,
        )
        neighbours = tree.query_ball_point(point_um, float(radius_um))
        available = [idx for idx in neighbours if idx not in used_gt_indices]
        if not available:
            continue
        best_idx = min(
            available,
            key=lambda idx: (
                float(np.linalg.norm(gt_points_um[idx] - point_um)),
                int(gt_ids[idx]),
            ),
        )
        used_gt_indices.add(int(best_idx))
        result[candidate_idx] = int(gt_ids[best_idx])
    return result


def empty_metric() -> dict[str, int]:
    return {
        "candidates": 0,
        "matched_missing_gt": 0,
        "false_candidates": 0,
        "recoverable_gt_edges": 0,
        "matched_with_incoming_context": 0,
        "matched_with_outgoing_context": 0,
        "matched_with_two_sided_context": 0,
    }


def add_metric(target: dict[str, int], source: dict[str, int]) -> None:
    for key in target:
        target[key] += int(source[key])


def finalise_metric(
    metric: dict[str, int],
    baseline_matched: int,
    gt_nodes: int,
) -> dict[str, int | float | None]:
    candidates = int(metric["candidates"])
    matched = int(metric["matched_missing_gt"])
    missing = max(0, int(gt_nodes) - int(baseline_matched))
    output: dict[str, int | float | None] = dict(metric)
    output["precision"] = matched / candidates if candidates else None
    output["missing_gt_recall"] = matched / missing if missing else None
    output["oracle_node_recall"] = (
        min(gt_nodes, baseline_matched + matched) / gt_nodes
        if gt_nodes
        else None
    )
    return output


def load_split_membership(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    membership: dict[str, str] = {}
    for name in manifest.get("train", []):
        membership[str(name)] = "checkpoint_train"
    for name in manifest.get("val", []):
        previous = membership.setdefault(str(name), "checkpoint_val")
        if previous != "checkpoint_val":
            raise ValueError(f"Dataset appears in both split lists: {name}")
    return membership


def write_report(path: Path, report: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def aggregate_groups(
    dataset_results: dict[str, dict[str, object]],
    thresholds: list[float],
) -> dict[str, dict[str, object]]:
    group_names = {"all"}
    for dataset, result in dataset_results.items():
        group_names.add(dataset.split("_", 1)[0])
        group_names.add(str(result["checkpoint_membership"]))

    output: dict[str, dict[str, object]] = {}
    for group in sorted(group_names):
        selected = {
            dataset: result
            for dataset, result in dataset_results.items()
            if (
                group == "all"
                or dataset.startswith(f"{group}_")
                or result["checkpoint_membership"] == group
            )
        }
        baseline_matched = sum(
            int(result["baseline_matched_gt"]) for result in selected.values()
        )
        gt_nodes = sum(int(result["gt_nodes"]) for result in selected.values())
        group_result: dict[str, object] = {
            "datasets": len(selected),
            "frames": sum(int(result["frames"]) for result in selected.values()),
            "gt_nodes": gt_nodes,
            "baseline_matched_gt": baseline_matched,
            "baseline_node_recall": (
                baseline_matched / gt_nodes if gt_nodes else None
            ),
            "selections": {},
        }
        selections: dict[str, object] = {}
        for selection in SELECTIONS:
            rows: dict[str, object] = {}
            for threshold in thresholds:
                metric = empty_metric()
                key = f"{threshold:.6g}"
                for result in selected.values():
                    add_metric(
                        metric,
                        result["selections"][selection][key],
                    )
                rows[key] = finalise_metric(
                    metric,
                    baseline_matched,
                    gt_nodes,
                )
            selections[selection] = rows
        group_result["selections"] = selections
        output[group] = group_result
    return output


def write_threshold_csv(
    path: Path,
    aggregates: dict[str, dict[str, object]],
) -> None:
    columns = [
        "group",
        "selection",
        "threshold",
        "datasets",
        "frames",
        "gt_nodes",
        "baseline_matched_gt",
        "baseline_node_recall",
        "candidates",
        "matched_missing_gt",
        "false_candidates",
        "precision",
        "missing_gt_recall",
        "oracle_node_recall",
        "recoverable_gt_edges",
        "matched_with_incoming_context",
        "matched_with_outgoing_context",
        "matched_with_two_sided_context",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for group, group_result in sorted(aggregates.items()):
            for selection, threshold_rows in group_result[
                "selections"
            ].items():
                for threshold, metric in threshold_rows.items():
                    writer.writerow(
                        {
                            "group": group,
                            "selection": selection,
                            "threshold": threshold,
                            "datasets": group_result["datasets"],
                            "frames": group_result["frames"],
                            "gt_nodes": group_result["gt_nodes"],
                            "baseline_matched_gt": group_result[
                                "baseline_matched_gt"
                            ],
                            "baseline_node_recall": group_result[
                                "baseline_node_recall"
                            ],
                            **metric,
                        }
                    )


def main() -> None:
    args = parse_args()
    thresholds = validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "deepcenter_rescue_audit.json"

    namespace = load_notebook_namespace(args)
    deepcenter_bundle = load_deepcenter_bundle(
        namespace,
        args.deepcenter_checkpoint,
    )
    namespace["USE_DEEPCENTER_VETO"] = True
    membership = load_split_membership(args.deepcenter_split_manifest)
    scale = np.asarray(namespace["VOXEL_SCALE_UM"], dtype=np.float64)
    pool_factor = int(
        getattr(deepcenter_bundle["cfg"], "pool_factor", 4)
    )

    baseline_paths = sorted(args.baseline_dir.glob("*.geff"))
    if args.max_datasets:
        baseline_paths = baseline_paths[: args.max_datasets]
    if not baseline_paths:
        raise FileNotFoundError(f"No GEFF graphs in {args.baseline_dir}")

    report: dict[str, object] = {
        "config": {
            "notebook": str(args.notebook.resolve()),
            "notebook_sha256": file_sha256(args.notebook),
            "baseline_dir": str(args.baseline_dir.resolve()),
            "image_dir": str(args.image_dir.resolve()),
            "ground_truth_dir": str(args.ground_truth_dir.resolve()),
            "deepcenter_checkpoint": str(
                args.deepcenter_checkpoint.resolve()
            ),
            "deepcenter_checkpoint_sha256": file_sha256(
                args.deepcenter_checkpoint
            ),
            "deepcenter_split_manifest": (
                str(args.deepcenter_split_manifest.resolve())
                if args.deepcenter_split_manifest is not None
                else None
            ),
            "thresholds": thresholds,
            "match_radius_um": args.match_radius_um,
            "existing_gate_um": args.existing_gate_um,
            "topology_radius_um": args.topology_radius_um,
            "nms_radius_um": args.nms_radius_um,
            "peak_min_distance": args.peak_min_distance,
            "centroid_refinement": not args.skip_centroid_refinement,
            "max_datasets": args.max_datasets,
            "max_frames_per_dataset": args.max_frames_per_dataset,
            "voxel_scale_um": scale.tolist(),
            "pool_factor": pool_factor,
        },
        "notes": [
            "No prediction graph is modified by this diagnostic.",
            "Candidate matches are greedy in descending DeepCenter score.",
            "Checkpoint train/validation clips are not independent evidence.",
        ],
        "dataset_results": {},
        "aggregates": {},
    }

    for dataset_index, baseline_path in enumerate(baseline_paths, start=1):
        dataset = baseline_path.stem
        gt_path = args.ground_truth_dir / f"{dataset}.geff"
        if not gt_path.is_dir():
            raise FileNotFoundError(gt_path)

        _, raw_nodes, raw_edges = graph_rows(namespace, baseline_path)
        apply_filter_config(
            namespace,
            (True, False, False, False, 0.20, 0.12),
        )
        baseline_nodes, baseline_edges, _ = namespace[
            "filter_output_graph"
        ](
            copy.deepcopy(raw_nodes),
            copy.deepcopy(raw_edges),
            dataset=dataset,
            deepcenter_bundle=None,
        )
        _, gt_nodes, gt_edges = graph_rows(namespace, gt_path)

        baseline_by_frame = frame_nodes(baseline_nodes)
        gt_by_frame = frame_nodes(gt_nodes)
        outgoing = {
            int(edge["source_id"]) for edge in baseline_edges
        }
        incoming = {
            int(edge["target_id"]) for edge in baseline_edges
        }
        endpoints_by_frame = {
            frame: [node_id for node_id in ids if node_id not in outgoing]
            for frame, ids in baseline_by_frame.items()
        }
        starts_by_frame = {
            frame: [node_id for node_id in ids if node_id not in incoming]
            for frame, ids in baseline_by_frame.items()
        }
        gt_predecessors: dict[int, list[int]] = defaultdict(list)
        gt_successors: dict[int, list[int]] = defaultdict(list)
        for edge in gt_edges:
            source_id = int(edge["source_id"])
            target_id = int(edge["target_id"])
            gt_successors[source_id].append(target_id)
            gt_predecessors[target_id].append(source_id)

        zarr_meta = (
            args.image_dir
            / f"{dataset}.zarr"
            / "0"
            / "zarr.json"
        )
        shape = tuple(
            int(value)
            for value in json.loads(
                zarr_meta.read_text(encoding="utf-8")
            )["shape"]
        )
        frame_count = shape[0]
        if args.max_frames_per_dataset:
            frame_count = min(frame_count, args.max_frames_per_dataset)

        baseline_gt_map: dict[int, int] = {}
        baseline_matched = 0
        gt_total = 0
        candidates_by_frame: dict[int, list[dict[str, object]]] = {}
        unmatched_gt_by_frame: dict[int, list[int]] = {}
        frame_cache: dict[int, np.ndarray] = {}
        heatmap_cache: dict[tuple[str, int], np.ndarray] = {}

        for t in range(frame_count):
            pred_ids = baseline_by_frame.get(t, [])
            gt_ids = gt_by_frame.get(t, [])
            pred_points = points_um(pred_ids, baseline_nodes, scale)
            gt_points = points_um(gt_ids, gt_nodes, scale)
            frame_matching = maximum_radius_matching(
                pred_points,
                gt_points,
                args.match_radius_um,
            )
            matched_gt_indices = {
                int(index) for index in frame_matching if index >= 0
            }
            baseline_matched += len(matched_gt_indices)
            gt_total += len(gt_ids)
            for pred_idx, gt_idx in enumerate(frame_matching):
                if gt_idx >= 0:
                    baseline_gt_map[int(gt_ids[gt_idx])] = int(
                        pred_ids[pred_idx]
                    )
            unmatched_gt_by_frame[t] = [
                gt_id
                for idx, gt_id in enumerate(gt_ids)
                if idx not in matched_gt_indices
            ]

            heatmap = namespace["deepcenter_heatmap_for_frame"](
                dataset,
                t,
                deepcenter_bundle,
                frame_cache,
                heatmap_cache,
            )
            volume = namespace["read_test_frame"](
                dataset,
                t,
                frame_cache,
            )
            peak_coordinates, peak_scores = extract_peaks(
                heatmap,
                volume,
                pool_factor,
                thresholds[0],
                args.peak_min_distance,
                scale,
                args.nms_radius_um,
                not args.skip_centroid_refinement,
            )
            existing_tree = (
                cKDTree(pred_points) if len(pred_points) else None
            )
            previous_ids = endpoints_by_frame.get(t - 1, [])
            next_ids = starts_by_frame.get(t + 1, [])
            previous_tree = (
                cKDTree(points_um(previous_ids, baseline_nodes, scale))
                if previous_ids
                else None
            )
            next_tree = (
                cKDTree(points_um(next_ids, baseline_nodes, scale))
                if next_ids
                else None
            )
            frame_candidates: list[dict[str, object]] = []
            for coordinate, score in zip(
                peak_coordinates,
                peak_scores,
                strict=True,
            ):
                point_um = coordinate * scale
                existing_distance = nearest_distance(
                    point_um,
                    existing_tree,
                )
                if existing_distance < args.existing_gate_um:
                    continue
                previous_distance = nearest_distance(
                    point_um,
                    previous_tree,
                )
                next_distance = nearest_distance(point_um, next_tree)
                near_previous = (
                    previous_distance <= args.topology_radius_um
                )
                near_next = next_distance <= args.topology_radius_um
                frame_candidates.append(
                    {
                        "score": float(score),
                        "z": float(coordinate[0]),
                        "y": float(coordinate[1]),
                        "x": float(coordinate[2]),
                        "point_um": point_um.tolist(),
                        "existing_distance_um": existing_distance,
                        "previous_endpoint_distance_um": previous_distance,
                        "next_start_distance_um": next_distance,
                        "endpoint_any": near_previous or near_next,
                        "bridge": near_previous and near_next,
                    }
                )
            candidates_by_frame[t] = frame_candidates
            frame_cache.pop(t, None)
            heatmap_cache.pop((dataset, t), None)

            if (t + 1) % 25 == 0 or t + 1 == frame_count:
                print(
                    f"[{dataset_index:02d}/{len(baseline_paths):02d}] "
                    f"{dataset} frame {t + 1}/{frame_count}",
                    flush=True,
                )

        selection_metrics: dict[str, dict[str, dict[str, int]]] = {
            selection: {
                f"{threshold:.6g}": empty_metric()
                for threshold in thresholds
            }
            for selection in SELECTIONS
        }
        for t in range(frame_count):
            candidates = candidates_by_frame.get(t, [])
            gt_ids = unmatched_gt_by_frame.get(t, [])
            gt_points = points_um(gt_ids, gt_nodes, scale)
            for selection in SELECTIONS:
                if selection == "novel":
                    candidate_indices = list(range(len(candidates)))
                else:
                    candidate_indices = [
                        idx
                        for idx, candidate in enumerate(candidates)
                        if bool(candidate[selection])
                    ]
                matched = greedy_candidate_matching(
                    candidates,
                    candidate_indices,
                    gt_ids,
                    gt_points,
                    args.match_radius_um,
                )
                for threshold in thresholds:
                    key = f"{threshold:.6g}"
                    eligible = [
                        idx
                        for idx in candidate_indices
                        if float(candidates[idx]["score"]) >= threshold
                    ]
                    metric = selection_metrics[selection][key]
                    metric["candidates"] += len(eligible)
                    for candidate_idx in eligible:
                        gt_id = matched.get(candidate_idx)
                        if gt_id is None:
                            metric["false_candidates"] += 1
                            continue
                        metric["matched_missing_gt"] += 1
                        incoming_context = sum(
                            predecessor in baseline_gt_map
                            for predecessor in gt_predecessors.get(
                                gt_id,
                                [],
                            )
                        )
                        outgoing_context = sum(
                            successor in baseline_gt_map
                            for successor in gt_successors.get(
                                gt_id,
                                [],
                            )
                        )
                        metric["recoverable_gt_edges"] += (
                            incoming_context + outgoing_context
                        )
                        metric["matched_with_incoming_context"] += int(
                            incoming_context > 0
                        )
                        metric["matched_with_outgoing_context"] += int(
                            outgoing_context > 0
                        )
                        metric[
                            "matched_with_two_sided_context"
                        ] += int(
                            incoming_context > 0 and outgoing_context > 0
                        )

        dataset_result: dict[str, object] = {
            "checkpoint_membership": membership.get(dataset, "unseen"),
            "frames": frame_count,
            "gt_nodes": gt_total,
            "baseline_nodes": sum(
                len(baseline_by_frame.get(t, []))
                for t in range(frame_count)
            ),
            "baseline_matched_gt": baseline_matched,
            "baseline_node_recall": (
                baseline_matched / gt_total if gt_total else None
            ),
            "baseline_missing_gt": gt_total - baseline_matched,
            "selections": selection_metrics,
        }
        report["dataset_results"][dataset] = dataset_result
        report["aggregates"] = aggregate_groups(
            report["dataset_results"],
            thresholds,
        )
        write_report(report_path, report)
        print(
            f"[{dataset_index:02d}/{len(baseline_paths):02d}] {dataset} "
            f"baseline_recall={dataset_result['baseline_node_recall']:.4f} "
            f"missing={dataset_result['baseline_missing_gt']}",
            flush=True,
        )

    report["aggregates"] = aggregate_groups(
        report["dataset_results"],
        thresholds,
    )
    write_report(report_path, report)
    write_threshold_csv(
        args.output_dir / "deepcenter_rescue_thresholds.csv",
        report["aggregates"],
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
