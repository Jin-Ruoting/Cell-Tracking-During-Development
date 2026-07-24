from __future__ import annotations

import math

from scripts.analyze_ground_truth_divisions import (
    extract_divisions,
    summarize_events,
)
from scripts.evaluate_edge_predictions import Graph, Node


def test_extract_division_geometry() -> None:
    graph = Graph(
        nodes={
            1: Node(t=0, z=1.0, y=10.0, x=10.0),
            2: Node(t=1, z=1.0, y=11.0, x=10.0),
            3: Node(t=2, z=1.0, y=12.0, x=9.0),
            4: Node(t=2, z=1.0, y=12.0, x=11.0),
        },
        edges=[(1, 2), (2, 3), (2, 4)],
    )

    events = extract_divisions("44b6_example", graph)

    assert len(events) == 1
    event = events[0]
    assert event.embryo == "44b6"
    assert event.t == 1
    assert math.isclose(event.child_a_um, math.sqrt(2) * 0.40625)
    assert math.isclose(event.child_b_um, math.sqrt(2) * 0.40625)
    assert math.isclose(event.sister_um, 2 * 0.40625)
    assert math.isclose(event.midpoint_um, 0.40625)
    assert math.isclose(event.predecessor_um or 0.0, 0.40625)


def test_nonconsecutive_fork_is_not_annotated_division() -> None:
    graph = Graph(
        nodes={
            1: Node(t=0, z=0.0, y=0.0, x=0.0),
            2: Node(t=1, z=0.0, y=0.0, x=1.0),
            3: Node(t=2, z=0.0, y=0.0, x=2.0),
        },
        edges=[(1, 2), (1, 3)],
    )

    assert extract_divisions("6bba_example", graph) == []


def test_summary_reports_both_daughter_distances() -> None:
    graph = Graph(
        nodes={
            1: Node(t=0, z=0.0, y=0.0, x=0.0),
            2: Node(t=1, z=0.0, y=1.0, x=0.0),
            3: Node(t=1, z=0.0, y=3.0, x=0.0),
        },
        edges=[(1, 2), (1, 3)],
    )
    summary = summarize_events(extract_divisions("44b6_example", graph))

    assert summary["divisions"] == 1
    parent_child = summary["parent_child_um"]
    assert parent_child["count"] == 2
    assert math.isclose(parent_child["min"], 0.40625)
    assert math.isclose(parent_child["max"], 3 * 0.40625)
