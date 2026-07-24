from scripts.analyze_division_candidates import (
    extract_candidate_rows,
    summarize,
)
from scripts.evaluate_edge_predictions import Graph, Node


def _node(t, z, y, x):
    return Node(t=t, z=z, y=y, x=x)


def test_extract_candidate_rows_labels_recoverable_fork():
    predicted = Graph(
        nodes={
            1: _node(0, 0, 0, 0),
            2: _node(1, 0, 4, 0),
            3: _node(1, 0, -4, 0),
            4: _node(2, 0, -5, 0),
        },
        edges=[(1, 2), (3, 4)],
    )
    ground_truth = Graph(
        nodes={
            10: _node(0, 0, 0, 0),
            20: _node(1, 0, 4, 0),
            30: _node(1, 0, -4, 0),
        },
        edges=[(10, 20), (10, 30)],
    )

    rows = extract_candidate_rows(
        "44b6_example",
        predicted,
        ground_truth,
        {(1, 2): 0.9, (1, 3): 0.8},
        predicted_to_ground={1: 10, 2: 20, 3: 30},
    )

    row = next(row for row in rows if row["parent_id"] == 1)
    assert row["candidate_child_id"] == 3
    assert row["candidate_edge_probability"] == 0.8
    assert row["existing_child_has_successor"] is False
    assert row["candidate_child_has_successor"] is True
    assert row["is_true_fork"] is True


def test_extract_candidate_rows_rejects_wrong_root_as_true_fork():
    predicted = Graph(
        nodes={
            1: _node(0, 0, 0, 0),
            2: _node(1, 0, 2, 0),
            3: _node(1, 0, -2, 0),
            4: _node(1, 0, 8, 0),
        },
        edges=[(1, 2)],
    )
    ground_truth = Graph(
        nodes={
            10: _node(0, 0, 0, 0),
            20: _node(1, 0, 2, 0),
            30: _node(1, 0, -2, 0),
            40: _node(1, 0, 8, 0),
        },
        edges=[(10, 20), (10, 30)],
    )

    rows = extract_candidate_rows(
        "6bba_example",
        predicted,
        ground_truth,
        {(1, 2): 0.9, (1, 3): 0.7, (1, 4): 0.8},
        predicted_to_ground={1: 10, 2: 20, 3: 30, 4: 40},
    )

    by_candidate = {row["candidate_child_id"]: row for row in rows}
    assert by_candidate[3]["is_true_fork"] is True
    assert by_candidate[4]["is_true_fork"] is False
    assert by_candidate[4]["candidate_rank"] == 1


def test_summarize_counts_unique_recoverable_divisions():
    rows = [
        {
            "dataset": "44b6_a",
            "embryo": "44b6",
            "ground_truth_parent_id": 10,
            "is_true_fork": True,
            "candidate_edge_probability": 0.8,
            "existing_edge_probability": 0.9,
            "parent_candidate_um": 4.0,
            "parent_existing_um": 3.0,
            "sister_um": 6.0,
            "midpoint_um": 1.0,
            "child_distance_delta_um": 1.0,
            "parent_motion_um": None,
            "candidate_rank": 1,
        },
        {
            "dataset": "6bba_b",
            "embryo": "6bba",
            "ground_truth_parent_id": 20,
            "is_true_fork": False,
            "candidate_edge_probability": None,
            "existing_edge_probability": 0.7,
            "parent_candidate_um": 8.0,
            "parent_existing_um": 5.0,
            "sister_um": 10.0,
            "midpoint_um": 3.0,
            "child_distance_delta_um": 3.0,
            "parent_motion_um": 2.0,
            "candidate_rank": 2,
        },
    ]

    summary = summarize(rows)

    assert summary["candidates"] == 2
    assert summary["candidates_with_preilp_edge"] == 1
    assert summary["recoverable_divisions"] == 1
