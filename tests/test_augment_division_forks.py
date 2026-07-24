import pytest

from scripts.augment_division_forks import (
    matches_rule,
    select_rows,
    validate_rule,
)


def _row(**updates):
    row = {
        "parent_id": 1,
        "candidate_rank": 1,
        "candidate_edge_probability": 0.7,
        "existing_edge_probability": 0.8,
        "parent_candidate_um": 7.0,
        "parent_existing_um": 5.0,
        "sister_um": 10.0,
        "midpoint_um": 2.0,
        "child_distance_delta_um": 2.0,
        "parent_motion_um": None,
        "candidate_child_has_successor": True,
        "existing_child_has_successor": True,
    }
    row.update(updates)
    return row


def test_matches_rule_applies_bounds_and_allows_missing_motion():
    rule = {
        "name": "balanced",
        "require_preilp_edge": True,
        "min_parent_existing_um": 4.0,
        "max_parent_candidate_um": 8.0,
        "max_parent_motion_um": 5.0,
    }

    assert matches_rule(_row(), rule)
    assert not matches_rule(_row(parent_candidate_um=9.0), rule)
    assert not matches_rule(_row(candidate_edge_probability=None), rule)


def test_select_rows_keeps_best_rank_per_parent():
    rule = {"name": "ranked", "max_candidate_rank": 2}
    rows = [
        _row(candidate_rank=2, candidate_child_id=20),
        _row(candidate_rank=1, candidate_child_id=10),
        _row(parent_id=2, candidate_rank=1, candidate_child_id=30),
    ]

    selected = select_rows(rows, rule)

    assert [row["candidate_child_id"] for row in selected] == [10, 30]


def test_validate_rule_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown"):
        validate_rule({"name": "bad", "oracle_label": True})
