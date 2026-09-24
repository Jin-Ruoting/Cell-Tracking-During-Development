"""Frozen E044 integer-anchor and collision-safe coordinate policy."""
from __future__ import annotations

from collections import Counter, defaultdict

SHAPE = (64, 256, 256)
DOWNSAMPLE = (1, 4, 4)
SCALE = (1.625, 0.40625, 0.40625)
POLICY = {
    "anchor": "nearest_even_grid_index_clamped_to_0_63",
    "output": "nearest_even_raw_voxel",
    "ambiguous_anchor": "retain_all_original_positions",
    "invalid_bounds": "retain_original_without_clipping",
    "new_collision": "retain_original_for_all_conflicting_proposals",
    "original_occupancy": "reject_other_original_positions",
    "iterations": 1,
    "preserve_node_ids_times_and_all_edges": True,
}


def original_positions(points):
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or not (points == np.rint(points)).all()
            or (points < 0).any() or (points >= np.asarray(SHAPE)).any()):
        raise ValueError("Invalid original integer positions")
    return points.astype(np.int64)


def native_anchors(points):
    import numpy as np

    original = original_positions(points)
    return np.clip(np.rint(original / np.asarray(DOWNSAMPLE)), 0, 63).astype(np.int64)


def preserve_collision_classes(original, candidate):
    """Distinct original positions must remain distinct after quantization."""
    groups = defaultdict(set)
    for before, after in zip(original, candidate):
        groups[tuple(after)].add(tuple(before))
    if any(len(values) > 1 for values in groups.values()):
        raise ValueError("New same-frame coordinate collision")


def safe_proposal(points, proposed, anchors):
    """Simultaneously accept valid proposals; all conflict checks are label-free."""
    import numpy as np

    original = original_positions(points)
    anchors = np.asarray(anchors)
    proposed = np.asarray(proposed, dtype=np.float64)
    if (anchors.shape != original.shape or not np.array_equal(anchors, native_anchors(original))
            or proposed.shape != original.shape or not np.isfinite(proposed).all()):
        raise ValueError("Invalid proposal or changed anchor convention")
    counts = Counter(map(tuple, anchors))
    ambiguous = np.asarray([counts[tuple(a)] > 1 for a in anchors], dtype=bool)
    rounded = np.rint(proposed)
    bounds = np.any((proposed < 0) | (proposed >= np.asarray(SHAPE))
                    | (rounded < 0) | (rounded >= np.asarray(SHAPE)), axis=1)
    # Select a valid representable value before conversion, even for huge finite residuals.
    integer = np.where(bounds[:, None], original, rounded).astype(np.int64)
    occupied = set(map(tuple, original))
    other_original = np.asarray([tuple(new) != tuple(old) and tuple(new) in occupied
                                 for old, new in zip(original, integer)], dtype=bool)
    groups = defaultdict(list)
    for i, new in enumerate(integer):
        groups[tuple(new)].append(i)
    collision = np.zeros(len(original), dtype=bool)
    for indices in groups.values():
        if len({tuple(original[i]) for i in indices}) > 1:
            collision[indices] = True
    rejected = ambiguous | bounds | other_original | collision
    result = integer.copy()
    result[rejected] = original[rejected]
    preserve_collision_classes(original, result)
    changed = np.any(result != original, axis=1)
    movement = np.linalg.norm((result - original) * np.asarray(SCALE), axis=1)
    audit = {"nodes": len(original), "changed_nodes": int(changed.sum()),
             "ambiguous_anchor_nodes": int(ambiguous.sum()), "bounds_rejected_nodes": int(bounds.sum()),
             "original_occupancy_rejected_nodes": int(other_original.sum()),
             "proposal_collision_nodes": int(collision.sum()), "retained_original_nodes": int(rejected.sum()),
             "movement_um_quantiles": dict(zip(("min", "median", "q95", "max"),
                                              map(float, np.quantile(movement, [0, .5, .95, 1])))) if len(original) else {}}
    return result, audit


def validate_rewrite(original, candidate, fields, expected_changes):
    if len(original) != len(candidate):
        raise ValueError("Row count changed")
    count = 0
    before_frames, after_frames = defaultdict(list), defaultdict(list)
    for old, new in zip(original, candidate):
        allowed = {"z", "y", "x"} if old["row_type"] == "node" else set()
        if any(old[k] != new[k] for k in fields if k not in allowed):
            raise ValueError("Node identity/time or edge field changed")
        if old["row_type"] == "node":
            before_frames[old["t"]].append([int(old[k]) for k in ("z", "y", "x")])
            after_frames[old["t"]].append([int(new[k]) for k in ("z", "y", "x")])
            count += any(old[k] != new[k] for k in ("z", "y", "x"))
    for t in before_frames:
        old, new = original_positions(before_frames[t]), original_positions(after_frames[t])
        preserve_collision_classes(old, new)
    if count != expected_changes:
        raise ValueError("Changed-node count differs from recorded policy")
