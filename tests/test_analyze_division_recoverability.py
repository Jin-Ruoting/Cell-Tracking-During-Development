from __future__ import annotations

from scripts.analyze_division_recoverability import analyze_pair, summarize
from scripts.evaluate_edge_predictions import Graph, Node


def _ground_truth() -> Graph:
    return Graph(
        nodes={
            1: Node(t=0, z=0.0, y=10.0, x=10.0),
            2: Node(t=1, z=0.0, y=9.0, x=10.0),
            3: Node(t=1, z=0.0, y=11.0, x=10.0),
        },
        edges=[(1, 2), (1, 3)],
    )


def test_detects_addable_direct_fork() -> None:
    predicted = Graph(
        nodes={
            10: Node(t=0, z=0.0, y=10.0, x=10.0),
            20: Node(t=1, z=0.0, y=9.0, x=10.0),
            30: Node(t=1, z=0.0, y=11.0, x=10.0),
        },
        edges=[(10, 20)],
    )

    rows = analyze_pair(
        "44b6_example",
        predicted,
        _ground_truth(),
        predicted_to_ground={10: 1, 20: 2, 30: 3},
    )

    assert len(rows) == 1
    assert rows[0]["direct_triplet_detected"] is True
    assert rows[0]["existing_correct_edges"] == 1
    assert rows[0]["missing_targets_are_roots"] is True
    assert rows[0]["addable_direct_fork"] is True
    assert summarize(rows)["addable_direct_fork"] == 1


def test_rejects_second_child_with_another_parent() -> None:
    predicted = Graph(
        nodes={
            10: Node(t=0, z=0.0, y=10.0, x=10.0),
            11: Node(t=0, z=0.0, y=12.0, x=10.0),
            20: Node(t=1, z=0.0, y=9.0, x=10.0),
            30: Node(t=1, z=0.0, y=11.0, x=10.0),
        },
        edges=[(10, 20), (11, 30)],
    )

    row = analyze_pair(
        "6bba_example",
        predicted,
        _ground_truth(),
        predicted_to_ground={10: 1, 20: 2, 30: 3},
    )[0]

    assert row["direct_triplet_detected"] is True
    assert row["missing_targets_are_roots"] is False
    assert row["addable_direct_fork"] is False


def test_missing_daughter_detection_is_not_recoverable() -> None:
    predicted = Graph(
        nodes={
            10: Node(t=0, z=0.0, y=10.0, x=10.0),
            20: Node(t=1, z=0.0, y=9.0, x=10.0),
        },
        edges=[(10, 20)],
    )

    row = analyze_pair(
        "44b6_example",
        predicted,
        _ground_truth(),
        predicted_to_ground={10: 1, 20: 2},
    )[0]

    assert row["direct_triplet_detected"] is False
    assert row["addable_direct_fork"] is False
