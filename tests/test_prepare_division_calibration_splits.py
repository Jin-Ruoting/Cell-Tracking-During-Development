from scripts.prepare_division_calibration_splits import build_manifest


def _report():
    return {
        "per_dataset": {
            "44b6_pos_a": {"divisions": 2, "edges": 100},
            "44b6_pos_b": {"divisions": 1, "edges": 50},
            "44b6_neg_a": {"divisions": 0, "edges": 95},
            "44b6_neg_b": {"divisions": 0, "edges": 45},
            "6bba_pos_a": {"divisions": 3, "edges": 80},
            "6bba_pos_visible": {"divisions": 4, "edges": 90},
            "6bba_neg_a": {"divisions": 0, "edges": 75},
            "6bba_neg_b": {"divisions": 0, "edges": 10},
        }
    }


def test_build_manifest_excludes_labels_and_matches_negatives():
    manifest = build_manifest(
        _report(),
        {"6bba_pos_visible"},
        positive_limit_per_embryo=1,
        shard_count=2,
    )

    assert manifest["selected"] == [
        "44b6_neg_a",
        "44b6_pos_a",
        "6bba_neg_a",
        "6bba_pos_a",
    ]
    assert "6bba_pos_visible" not in manifest["roles"]
    assert manifest["roles"]["44b6_pos_a"] == "positive"
    assert manifest["roles"]["44b6_neg_a"] == "negative"


def test_build_manifest_is_disjoint_and_deterministic():
    first = build_manifest(
        _report(),
        set(),
        positive_limit_per_embryo=2,
        shard_count=2,
    )
    second = build_manifest(
        _report(),
        set(),
        positive_limit_per_embryo=2,
        shard_count=2,
    )

    assert first == second
    flattened = [name for shard in first["shards"] for name in shard]
    assert len(flattened) == len(set(flattened))
    assert sorted(flattened) == first["selected"]
    assert len(first["shard_edge_loads"]) == 2
