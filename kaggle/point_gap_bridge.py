"""Observation-supported, unambiguous one-frame bridges on a frozen graph.

No labels, learned edge scores, or parameter search are used. Existing nodes
and edges are retained. Ambiguous endpoint/peak assignments are all rejected.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

SCALE = np.asarray((1.625, 0.40625, 0.40625))
CONFIG = {
    "missing_frames": 1,
    "context_edges_each_side": 2,
    "minimum_peak_probability": 0.5,
    "existing_node_exclusion_um": 2.0,
    "maximum_step_um": 5.0,
    "maximum_prediction_residual_um": 3.5,
    "maximum_added_fraction": 0.03,
    "ambiguity_policy": "reject_every_shared_endpoint_or_peak",
    "iterations": 1,
}


def validate_peaks(coords, scores, frames, shape):
    coords, scores = np.asarray(coords), np.asarray(scores)
    if (coords.ndim != 2 or coords.shape[1] != 4 or scores.shape != (len(coords),)
            or len(shape) != 4 or list(frames) != list(range(shape[0]))
            or not np.isfinite(coords).all() or not np.isfinite(scores).all()
            or not np.equal(coords, np.rint(coords)).all()
            or (coords < 0).any() or (coords >= np.asarray(shape)).any()
            or (scores < 0.2).any() or (scores > 1).any()
            or len(np.unique(coords, axis=0)) != len(coords)):
        raise ValueError("Invalid or incomplete frozen peak cache")


def bridge_one_frame(ids, points, pairs, peak_coords, peak_scores):
    """Return appended-node/edge proposals and counters, without mutating inputs.

    ``points`` and ``peak_coords`` are integer TZYX in original voxels. A tail
    needs two unique incoming context edges; a head two unique outgoing ones.
    Mean two-step velocities must both predict the observed missing-frame peak.
    """
    ids = np.asarray(ids)
    points = np.asarray(points)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    peak_coords, peak_scores = np.asarray(peak_coords), np.asarray(peak_scores)
    if (ids.ndim != 1 or points.shape != (len(ids), 4) or len(set(ids)) != len(ids)
            or not np.isfinite(points).all() or not np.equal(points, np.rint(points)).all()
            or peak_coords.shape != (len(peak_scores), 4)
            or not np.isfinite(peak_coords).all() or not np.isfinite(peak_scores).all()
            or len(set(map(tuple, pairs))) != len(pairs)):
        raise ValueError("Invalid graph or peak arrays")
    index = {int(node): i for i, node in enumerate(ids)}
    children, parents = defaultdict(list), defaultdict(list)
    for source, target in pairs:
        if (source not in index or target not in index
                or points[index[target], 0] - points[index[source], 0] != 1):
            raise ValueError("Dangling or nonconsecutive original edge")
        children[int(source)].append(int(target))
        parents[int(target)].append(int(source))
    if any(len(v) > 2 for v in children.values()) or any(len(v) > 1 for v in parents.values()):
        raise ValueError("Invalid original lineage degree")
    positions = {node: points[i, 1:] * SCALE for node, i in index.items()}
    times = {node: int(points[i, 0]) for node, i in index.items()}

    def context(node, backward):
        chain, current = [node], node
        links, reverse = (parents, children) if backward else (children, parents)
        for _ in range(CONFIG["context_edges_each_side"]):
            if len(links[current]) != 1:
                return None
            next_node = links[current][0]
            if len(reverse[next_node]) != 1:
                return None
            chain.append(next_node)
            current = next_node
        return chain

    tails, heads = defaultdict(list), defaultdict(list)
    for node in sorted(index):
        if not children[node]:
            chain = context(node, True)
            if chain is not None:
                tails[times[node]].append((node, (positions[node] - positions[chain[-1]]) / 2))
        if not parents[node]:
            chain = context(node, False)
            if chain is not None:
                heads[times[node]].append((node, (positions[chain[-1]] - positions[node]) / 2))
    stats = {"context_tails": sum(map(len, tails.values())),
             "context_heads": sum(map(len, heads.values())), "free_peaks": 0,
             "feasible_triples": 0, "ambiguous_triples": 0,
             "budget_rejected": 0, "added_nodes": 0, "added_edges": 0}
    triples = []
    for time, tail_list in sorted(tails.items()):
        head_list = heads.get(time + 2, [])
        if not head_list:
            continue
        peak_indices = np.flatnonzero((peak_coords[:, 0] == time + 1) &
                                     (peak_scores >= CONFIG["minimum_peak_probability"]))
        if not len(peak_indices):
            continue
        peak_positions = peak_coords[peak_indices, 1:] * SCALE
        existing = points[points[:, 0] == time + 1, 1:] * SCALE
        if len(existing):
            free = cKDTree(existing).query(peak_positions)[0] > CONFIG["existing_node_exclusion_um"]
            peak_indices, peak_positions = peak_indices[free], peak_positions[free]
        stats["free_peaks"] += len(peak_indices)
        if not len(peak_indices):
            continue
        peak_tree = cKDTree(peak_positions)
        head_tree = cKDTree([positions[node] for node, _ in head_list])
        for tail, velocity_in in tail_list:
            a = positions[tail]
            for head_i in head_tree.query_ball_point(a, 2 * CONFIG["maximum_step_um"]):
                head, velocity_out = head_list[head_i]
                b = positions[head]
                midpoint = (a + b) / 2
                for peak_i in peak_tree.query_ball_point(midpoint, CONFIG["maximum_prediction_residual_um"]):
                    p = peak_positions[peak_i]
                    steps = (p - a, b - p)
                    residuals = (np.linalg.norm(p - (a + velocity_in)),
                                 np.linalg.norm(p - (b - velocity_out)))
                    if (max(map(np.linalg.norm, steps)) > CONFIG["maximum_step_um"]
                            or max(residuals) > CONFIG["maximum_prediction_residual_um"]
                            or np.dot(steps[0], velocity_in) < 0
                            or np.dot(steps[1], velocity_out) < 0):
                        continue
                    peak_id = int(peak_indices[peak_i])
                    triples.append((tail, head, peak_id, float(sum(residuals))))
    stats["feasible_triples"] = len(triples)
    use_tail = Counter(a for a, _, _, _ in triples)
    use_head = Counter(b for _, b, _, _ in triples)
    use_peak = Counter(p for _, _, p, _ in triples)
    unique = [row for row in triples if use_tail[row[0]] == use_head[row[1]] == use_peak[row[2]] == 1]
    stats["ambiguous_triples"] = len(triples) - len(unique)
    unique.sort(key=lambda row: (row[3], -float(peak_scores[row[2]]), row[0], row[1], row[2]))
    budget = int(np.floor(len(ids) * CONFIG["maximum_added_fraction"]))
    stats["budget_rejected"] = max(0, len(unique) - budget)
    selected = unique[:budget]
    first_id = max(index, default=-1) + 1
    bridges = [{"node_id": first_id + offset, "source_id": a, "target_id": b,
                "peak_index": p, "point": list(map(int, peak_coords[p])),
                "probability": float(peak_scores[p]), "residual_sum_um": cost}
               for offset, (a, b, p, cost) in enumerate(selected)]
    stats.update(added_nodes=len(bridges), added_edges=2 * len(bridges))
    return bridges, stats
