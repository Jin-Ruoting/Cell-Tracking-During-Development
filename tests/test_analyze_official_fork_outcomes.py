import pytest

from scripts.analyze_official_fork_outcomes import (
    parse_candidate_spec,
    summarize_outcomes,
)


def test_parse_candidate_spec():
    name, path = parse_candidate_spec("broad=/tmp/candidate")

    assert name == "broad"
    assert str(path) == "/tmp/candidate"


def test_parse_candidate_spec_rejects_missing_separator():
    with pytest.raises(ValueError, match="NAME=PATH"):
        parse_candidate_spec("/tmp/candidate")


def test_summarize_outcomes_counts_embryos():
    rows = [
        {"embryo": "44b6", "outcome": "tp"},
        {"embryo": "44b6", "outcome": "ignored"},
        {"embryo": "6bba", "outcome": "fp"},
    ]

    summary = summarize_outcomes(rows)

    assert summary["added_edges"] == 3
    assert summary["official_tp_forks"] == 1
    assert summary["official_fp_forks"] == 1
    assert summary["official_ignored_forks"] == 1
    assert summary["by_embryo"]["44b6"]["official_tp_forks"] == 1
