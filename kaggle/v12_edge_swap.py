"""E040: reciprocal two-edge swaps on fixed E029 nodes, without labels.

Only non-division outgoing edges can participate. Node positions, every degree,
all division edges and total edge count remain invariant. No iterative relinking.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

CONFIG = {
    "maximum_new_edge_um": 5.5,
    "minimum_pair_logit_gain": math.log(9.0),
    "no_link_logit": 0.0,
    "reciprocal_unique_top1": True,
    "iterations": 1,
    "preserve_nodes_coordinates_degrees_division_edges": True,
}
SCALE = np.asarray((1.625, 0.40625, 0.40625))


def swap_frame(src_ids, dst_ids, src_points, dst_points, pairs, logits):
    """Return changed frame edges, decisions and counts without mutating inputs.

    Points are ZYX in original voxels. Logits include *all* original nodes in
    both frames, in the supplied ID order. Each accepted exchange must be a
    reciprocal two-cycle in that complete matrix, not in a prefiltered subset.
    The logit-gain threshold is a fixed score margin, not calibrated confidence.
    """
    src_ids, dst_ids = np.asarray(src_ids), np.asarray(dst_ids)
    src_points, dst_points = np.asarray(src_points), np.asarray(dst_points)
    raw_pairs = np.asarray(pairs).reshape(-1, 2)
    if not np.isfinite(raw_pairs).all() or not np.equal(raw_pairs, np.rint(raw_pairs)).all():
        raise ValueError("Noninteger original edge IDs")
    pairs = raw_pairs.astype(np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    for ids, pts in ((src_ids, src_points), (dst_ids, dst_points)):
        if (ids.ndim != 1 or len(set(ids)) != len(ids) or not np.isfinite(ids).all()
                or not np.equal(ids, np.rint(ids)).all()
                or pts.shape != (len(ids), 3) or not np.isfinite(pts).all()
                or (pts < 0).any() or (pts >= (64, 256, 256)).any()):
            raise ValueError("Invalid fixed node IDs or original-voxel coordinates")
    if set(src_ids) & set(dst_ids):
        raise ValueError("Frame node IDs must be disjoint")
    if logits.shape != (len(src_ids), len(dst_ids)) or not np.isfinite(logits).all():
        raise ValueError("Incomplete/nonfinite fixed-node logits")
    if len(set(map(tuple, pairs))) != len(pairs):
        raise ValueError("Duplicate original edge")
    source_index = {int(n): i for i, n in enumerate(src_ids)}
    target_index = {int(n): i for i, n in enumerate(dst_ids)}
    if any(int(a) not in source_index or int(b) not in target_index for a, b in pairs):
        raise ValueError("Dangling or wrong-frame original edge")
    outdegree = Counter(map(int, pairs[:, 0]))
    indegree = Counter(map(int, pairs[:, 1]))
    if max(outdegree.values(), default=0) > 2 or max(indegree.values(), default=0) > 1:
        raise ValueError("Invalid original lineage degree")
    stats = {"original_edges": len(pairs), "eligible_sources": sum(v == 1 for v in outdegree.values()),
             "reciprocal_two_cycles": 0, "distance_rejected": 0, "margin_rejected": 0,
             "accepted_swaps": 0, "changed_edges": 0}
    changed = pairs.copy()
    if not len(src_ids) or not len(dst_ids):
        return changed, [], stats
    chosen = np.argmax(logits, axis=1)
    reverse = np.argmax(logits, axis=0)
    row_max = logits[np.arange(len(src_ids)), chosen]
    col_max = logits[reverse, np.arange(len(dst_ids))]
    row_unique = (logits == row_max[:, None]).sum(axis=1) == 1
    col_unique = (logits == col_max[None, :]).sum(axis=0) == 1
    existing = {int(a): (int(b), i) for i, (a, b) in enumerate(pairs) if outdegree[int(a)] == 1}
    parent = {int(b): int(a) for a, b in pairs}
    decisions = []
    touched = set()
    for a in sorted(existing):
        b, ab = existing[a]
        ai, bi = source_index[a], target_index[b]
        di = int(chosen[ai])
        d = int(dst_ids[di])
        c = parent.get(d)
        if d == b or c not in existing or a >= c:
            continue
        _, cd = existing[c]
        ci = source_index[c]
        if (int(chosen[ci]) != bi or int(reverse[di]) != ai or int(reverse[bi]) != ci
                or not (row_unique[ai] and row_unique[ci] and col_unique[di] and col_unique[bi])
                or min(logits[ai, di], logits[ci, bi]) <= CONFIG["no_link_logit"]):
            continue
        stats["reciprocal_two_cycles"] += 1
        distances = [float(np.linalg.norm((src_points[ai] - dst_points[di]) * SCALE)),
                     float(np.linalg.norm((src_points[ci] - dst_points[bi]) * SCALE))]
        if max(distances) > CONFIG["maximum_new_edge_um"]:
            stats["distance_rejected"] += 1
            continue
        gain = float(logits[ai, di] + logits[ci, bi] - logits[ai, bi] - logits[ci, di])
        if gain < CONFIG["minimum_pair_logit_gain"]:
            stats["margin_rejected"] += 1
            continue
        if {a, b, c, d} & touched:
            raise ValueError("Reciprocal two-cycles unexpectedly share endpoints")
        touched.update((a, b, c, d))
        changed[ab, 1], changed[cd, 1] = d, b
        decisions.append({"removed": [[a, b], [c, d]], "added": [[a, d], [c, b]],
                          "pair_logit_gain": gain, "new_edge_um": distances})
    if (Counter(map(int, changed[:, 0])) != outdegree or Counter(map(int, changed[:, 1])) != indegree
            or len(set(map(tuple, changed))) != len(pairs)):
        raise ValueError("Swap changed graph degree or introduced duplicate edges")
    divisions = {tuple(edge) for edge in pairs if outdegree[int(edge[0])] == 2}
    if not divisions <= set(map(tuple, changed)):
        raise ValueError("Original division edge changed")
    stats.update(accepted_swaps=len(decisions), changed_edges=2 * len(decisions))
    return changed, decisions, stats
