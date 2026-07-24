"""Prediction-only direct-division rule used by the E005 notebook."""

import math
from collections import defaultdict


VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)
SEARCHED_DIVISION_RULE_NAME = "search_overall_rounded"
SEARCHED_DIVISION_RULE = {
    "max_candidate_rank": 1,
    "max_parent_candidate_um": 12.0,
    "min_parent_existing_um": 2.5,
    "min_sister_um": 8.0,
    "max_sister_um": 19.0,
    "max_midpoint_um": 4.25,
    "max_child_distance_delta_um": 9.0,
    "min_existing_edge_probability": 0.69,
    "min_parent_motion_um": 1.5,
    "max_parent_motion_um": 8.0,
}
SEARCH_MAX_PARENT_UM = 16.0
SEARCH_MAX_SISTER_UM = 22.0


def _node_point(node: dict[str, object]) -> tuple[float, float, float]:
    return (
        float(node["z"]),
        float(node["y"]),
        float(node["x"]),
    )


def _physical_distance(
    left: dict[str, object],
    right: dict[str, object],
) -> float:
    a = _node_point(left)
    b = _node_point(right)
    return math.sqrt(
        sum(
            ((a[axis] - b[axis]) * VOXEL_SCALE_UM[axis]) ** 2
            for axis in range(3)
        )
    )


def _midpoint_distance(
    parent: dict[str, object],
    first_child: dict[str, object],
    second_child: dict[str, object],
) -> float:
    parent_point = _node_point(parent)
    first_point = _node_point(first_child)
    second_point = _node_point(second_child)
    midpoint = tuple(
        (first_point[axis] + second_point[axis]) / 2.0
        for axis in range(3)
    )
    return math.sqrt(
        sum(
            (
                (parent_point[axis] - midpoint[axis])
                * VOXEL_SCALE_UM[axis]
            )
            ** 2
            for axis in range(3)
        )
    )


def edge_probability_map(
    edge_rows,
) -> dict[tuple[int, int], float]:
    probabilities: dict[tuple[int, int], float] = {}
    for row in edge_rows:
        value = row.get("edge_prob") if hasattr(row, "get") else None
        if value is None:
            continue
        probability = float(value)
        if not math.isfinite(probability):
            continue
        key = (int(row["source_id"]), int(row["target_id"]))
        probabilities[key] = max(probabilities.get(key, float("-inf")), probability)
    return probabilities


def _matches_searched_rule(row: dict[str, object]) -> bool:
    required = (
        "candidate_edge_probability",
        "existing_edge_probability",
        "parent_motion_um",
    )
    if any(row.get(name) is None for name in required):
        return False
    numeric_values = (
        "candidate_edge_probability",
        "existing_edge_probability",
        "parent_motion_um",
        "candidate_rank",
        "parent_candidate_um",
        "parent_existing_um",
        "sister_um",
        "midpoint_um",
        "child_distance_delta_um",
    )
    if any(
        not math.isfinite(float(row[name]))
        for name in numeric_values
    ):
        return False
    rule = SEARCHED_DIVISION_RULE
    return (
        int(row["candidate_rank"]) <= int(rule["max_candidate_rank"])
        and float(row["parent_candidate_um"])
        <= float(rule["max_parent_candidate_um"])
        and float(row["parent_existing_um"])
        >= float(rule["min_parent_existing_um"])
        and float(row["sister_um"]) >= float(rule["min_sister_um"])
        and float(row["sister_um"]) <= float(rule["max_sister_um"])
        and float(row["midpoint_um"]) <= float(rule["max_midpoint_um"])
        and float(row["child_distance_delta_um"])
        <= float(rule["max_child_distance_delta_um"])
        and float(row["existing_edge_probability"])
        >= float(rule["min_existing_edge_probability"])
        and float(row["parent_motion_um"])
        >= float(rule["min_parent_motion_um"])
        and float(row["parent_motion_um"])
        <= float(rule["max_parent_motion_um"])
    )


def select_searched_division_edges(
    nodes_by_id: dict[int, dict[str, object]],
    raw_edges: list[dict[str, object]],
    preilp_edge_probabilities: dict[tuple[int, int], float],
) -> list[dict[str, object]]:
    successors: dict[int, list[int]] = defaultdict(list)
    predecessors: dict[int, list[int]] = defaultdict(list)
    for edge in raw_edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        successors[source_id].append(target_id)
        predecessors[target_id].append(source_id)

    roots_by_time: dict[int, list[int]] = defaultdict(list)
    for node_id, node in nodes_by_id.items():
        if not predecessors.get(node_id):
            roots_by_time[int(node["t"])].append(node_id)

    rows: list[dict[str, object]] = []
    for parent_id, child_ids in sorted(successors.items()):
        if len(child_ids) != 1:
            continue
        parent = nodes_by_id.get(parent_id)
        existing_child_id = child_ids[0]
        existing_child = nodes_by_id.get(existing_child_id)
        if (
            parent is None
            or existing_child is None
            or int(existing_child["t"]) != int(parent["t"]) + 1
        ):
            continue

        predecessor_ids = predecessors.get(parent_id, [])
        parent_motion_um = (
            _physical_distance(nodes_by_id[predecessor_ids[0]], parent)
            if len(predecessor_ids) == 1
            and predecessor_ids[0] in nodes_by_id
            else None
        )
        parent_existing_um = _physical_distance(parent, existing_child)
        candidates: list[dict[str, object]] = []
        for candidate_id in roots_by_time.get(int(parent["t"]) + 1, []):
            candidate = nodes_by_id[candidate_id]
            parent_candidate_um = _physical_distance(parent, candidate)
            if parent_candidate_um > SEARCH_MAX_PARENT_UM:
                continue
            sister_um = _physical_distance(existing_child, candidate)
            if sister_um > SEARCH_MAX_SISTER_UM:
                continue
            candidates.append(
                {
                    "parent_id": parent_id,
                    "existing_child_id": existing_child_id,
                    "candidate_child_id": candidate_id,
                    "candidate_edge_probability": preilp_edge_probabilities.get(
                        (parent_id, candidate_id)
                    ),
                    "existing_edge_probability": preilp_edge_probabilities.get(
                        (parent_id, existing_child_id)
                    ),
                    "parent_candidate_um": parent_candidate_um,
                    "parent_existing_um": parent_existing_um,
                    "sister_um": sister_um,
                    "midpoint_um": _midpoint_distance(
                        parent,
                        existing_child,
                        candidate,
                    ),
                    "child_distance_delta_um": abs(
                        parent_candidate_um - parent_existing_um
                    ),
                    "parent_motion_um": parent_motion_um,
                }
            )

        candidates.sort(
            key=lambda row: (
                -float(row["candidate_edge_probability"])
                if row["candidate_edge_probability"] is not None
                else 1.0,
                float(row["midpoint_um"]),
                float(row["parent_candidate_um"]),
                int(row["candidate_child_id"]),
            )
        )
        for rank, row in enumerate(candidates, start=1):
            row["candidate_rank"] = rank
            if _matches_searched_rule(row):
                rows.append(row)

    rows.sort(
        key=lambda row: (
            int(row["candidate_rank"]),
            -float(row["candidate_edge_probability"]),
            float(row["midpoint_um"]),
            float(row["parent_candidate_um"]),
            int(row["parent_id"]),
            int(row["candidate_child_id"]),
        )
    )
    selected: list[dict[str, object]] = []
    used_parents: set[int] = set()
    used_candidates: set[int] = set()
    for row in rows:
        parent_id = int(row["parent_id"])
        candidate_id = int(row["candidate_child_id"])
        if parent_id in used_parents or candidate_id in used_candidates:
            continue
        selected.append(row)
        used_parents.add(parent_id)
        used_candidates.add(candidate_id)
    return sorted(selected, key=lambda row: int(row["parent_id"]))


def apply_searched_division_edges(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    output = [dict(edge) for edge in edges]
    successors: dict[int, list[int]] = defaultdict(list)
    predecessors: dict[int, list[int]] = defaultdict(list)
    edge_pairs: set[tuple[int, int]] = set()
    for edge in output:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        successors[source_id].append(target_id)
        predecessors[target_id].append(source_id)
        edge_pairs.add((source_id, target_id))

    stats = {
        "searched_division_candidates": len(selected_rows),
        "searched_divisions_added": 0,
        "searched_divisions_skipped_missing_node": 0,
        "searched_divisions_skipped_parent_topology": 0,
        "searched_divisions_skipped_candidate_topology": 0,
        "searched_divisions_skipped_nonconsecutive": 0,
        "searched_divisions_skipped_duplicate": 0,
    }
    for row in selected_rows:
        parent_id = int(row["parent_id"])
        existing_child_id = int(row["existing_child_id"])
        candidate_id = int(row["candidate_child_id"])
        parent = nodes_by_id.get(parent_id)
        candidate = nodes_by_id.get(candidate_id)
        if parent is None or candidate is None or existing_child_id not in nodes_by_id:
            stats["searched_divisions_skipped_missing_node"] += 1
            continue
        if successors.get(parent_id, []) != [existing_child_id]:
            stats["searched_divisions_skipped_parent_topology"] += 1
            continue
        if predecessors.get(candidate_id):
            stats["searched_divisions_skipped_candidate_topology"] += 1
            continue
        if int(candidate["t"]) != int(parent["t"]) + 1:
            stats["searched_divisions_skipped_nonconsecutive"] += 1
            continue
        if (parent_id, candidate_id) in edge_pairs:
            stats["searched_divisions_skipped_duplicate"] += 1
            continue
        output.append(
            {
                "source_id": parent_id,
                "target_id": candidate_id,
                "edge_prob": float(row["candidate_edge_probability"]),
                "distance_um": _physical_distance(parent, candidate),
            }
        )
        successors[parent_id].append(candidate_id)
        predecessors[candidate_id].append(parent_id)
        edge_pairs.add((parent_id, candidate_id))
        stats["searched_divisions_added"] += 1
    return output, stats
