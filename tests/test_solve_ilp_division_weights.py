from pathlib import Path

import pytest

from scripts.solve_ilp_division_weights import prediction_paths, weight_slug


@pytest.mark.parametrize(
    ("weight", "expected"),
    [
        (1.0, "1"),
        (0.75, "0p75"),
        (0.0, "0"),
        (-0.25, "m0p25"),
    ],
)
def test_weight_slug(weight, expected):
    assert weight_slug(weight) == expected


def test_prediction_paths_rejects_duplicate_dataset_names(tmp_path):
    first = tmp_path / "shard_0" / "sample.geff"
    second = tmp_path / "shard_1" / "sample.geff"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    with pytest.raises(ValueError, match="duplicate"):
        prediction_paths(tmp_path)


def test_prediction_paths_orders_by_dataset_name(tmp_path):
    (tmp_path / "shard_0" / "b.geff").mkdir(parents=True)
    (tmp_path / "shard_1" / "a.geff").mkdir(parents=True)

    assert prediction_paths(tmp_path) == [
        Path(tmp_path / "shard_1" / "a.geff"),
        Path(tmp_path / "shard_0" / "b.geff"),
    ]
