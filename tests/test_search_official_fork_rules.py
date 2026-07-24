from scripts.search_official_fork_rules import (
    Condition,
    canonical_conditions,
    condition_matches,
    rule_statistics,
)


def test_condition_matches_allows_missing_value_for_upper_bound():
    row = {"parent_motion_um": None}

    assert condition_matches(row, Condition("max", "parent_motion_um", 5.0))
    assert not condition_matches(row, Condition("min", "parent_motion_um", 1.0))


def test_canonical_conditions_keeps_strongest_bounds():
    conditions = canonical_conditions(
        (
            Condition("min", "sister_um", 5.0),
            Condition("min", "sister_um", 7.0),
            Condition("max", "sister_um", 15.0),
        )
    )

    assert conditions == (
        Condition("max", "sister_um", 15.0),
        Condition("min", "sister_um", 7.0),
    )


def test_rule_statistics_uses_fixed_division_denominator():
    rows = [
        {
            "embryo": "44b6",
            "outcome": "tp",
            "candidate_edge_probability": 0.8,
        },
        {
            "embryo": "44b6",
            "outcome": "fp",
            "candidate_edge_probability": 0.7,
        },
        {
            "embryo": "6bba",
            "outcome": "ignored",
            "candidate_edge_probability": 0.9,
        },
    ]

    result = rule_statistics(
        rows,
        (Condition("min", "candidate_edge_probability", 0.75),),
        {"44b6": 2, "6bba": 3},
    )

    assert result["added_edges"] == 2
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["division_jaccard"] == 0.2
