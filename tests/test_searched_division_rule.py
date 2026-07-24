from scripts.searched_division_rule import (
    apply_searched_division_edges,
    select_searched_division_edges,
)


def _node(node_id, t, y, x=0.0, z=0.0):
    return {
        "node_id": node_id,
        "t": t,
        "z": z,
        "y": y,
        "x": x,
    }


def _valid_graph():
    nodes = {
        0: _node(0, 0, -4.0),
        1: _node(1, 1, 0.0),
        2: _node(2, 2, 8.0),
        3: _node(3, 2, -20.0),
    }
    edges = [
        {"source_id": 0, "target_id": 1},
        {"source_id": 1, "target_id": 2},
    ]
    probabilities = {
        (1, 2): 0.8,
        (1, 3): 0.75,
    }
    return nodes, edges, probabilities


def test_selects_prediction_supported_rank_one_fork():
    nodes, edges, probabilities = _valid_graph()

    selected = select_searched_division_edges(nodes, edges, probabilities)

    assert [(row["parent_id"], row["candidate_child_id"]) for row in selected] == [
        (1, 3)
    ]


def test_rejects_missing_parent_motion_and_low_existing_probability():
    nodes, edges, probabilities = _valid_graph()
    without_motion = [edges[1]]
    low_probability = dict(probabilities, **{})
    low_probability[(1, 2)] = 0.68

    assert not select_searched_division_edges(
        nodes,
        without_motion,
        probabilities,
    )
    assert not select_searched_division_edges(
        nodes,
        edges,
        low_probability,
    )


def test_candidate_ranking_precedes_rule_thresholds():
    nodes, edges, probabilities = _valid_graph()
    nodes[4] = _node(4, 2, -7.5)
    probabilities[(1, 4)] = 0.95

    selected = select_searched_division_edges(nodes, edges, probabilities)

    assert not selected


def test_global_assignment_uses_candidate_once():
    nodes, edges, probabilities = _valid_graph()
    nodes.update(
        {
            5: _node(5, 0, -3.5),
            6: _node(6, 1, 0.5),
            7: _node(7, 2, 8.5),
        }
    )
    edges.extend(
        [
            {"source_id": 5, "target_id": 6},
            {"source_id": 6, "target_id": 7},
        ]
    )
    probabilities.update(
        {
            (6, 7): 0.8,
            (6, 3): 0.9,
        }
    )

    selected = select_searched_division_edges(nodes, edges, probabilities)

    assert [(row["parent_id"], row["candidate_child_id"]) for row in selected] == [
        (6, 3)
    ]


def test_apply_rechecks_exact_parent_and_root_topology():
    nodes, edges, probabilities = _valid_graph()
    selected = select_searched_division_edges(nodes, edges, probabilities)

    output, stats = apply_searched_division_edges(nodes, edges, selected)
    assert {(edge["source_id"], edge["target_id"]) for edge in output} == {
        (0, 1),
        (1, 2),
        (1, 3),
    }
    assert stats["searched_divisions_added"] == 1

    changed_edges = [
        {"source_id": 0, "target_id": 1},
        {"source_id": 1, "target_id": 3},
    ]
    output, stats = apply_searched_division_edges(
        nodes,
        changed_edges,
        selected,
    )
    assert len(output) == len(changed_edges)
    assert stats["searched_divisions_skipped_parent_topology"] == 1
