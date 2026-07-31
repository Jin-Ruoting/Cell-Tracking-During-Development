from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[1]
    / "kaggle"
    / "run_pu_appearance_filter.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location(
    "run_pu_appearance_filter",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
PU = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PU)


class AppearanceFeatureTests(unittest.TestCase):
    def test_features_are_finite_and_keep_the_frozen_width(self) -> None:
        volume = np.arange(5 * 9 * 9, dtype=np.float32).reshape(5, 9, 9)
        points = np.asarray(
            [[2.0, 4.0, 4.0], [0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        scale = np.asarray([2.0, 0.5, 0.5], dtype=np.float64)

        features = PU.appearance_features(volume, points, scale)

        self.assertEqual(features.shape, (2, len(PU.FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())


class ComponentFilterTests(unittest.TestCase):
    def test_removes_only_the_complete_nondivision_chain(self) -> None:
        nodes = {
            node_id: {"t": node_id, "z": 0, "y": 0, "x": 0}
            for node_id in range(6)
        }
        nodes.update(
            {
                100: {"t": 0, "z": 0, "y": 0, "x": 0},
                101: {"t": 1, "z": 0, "y": 0, "x": 0},
                102: {"t": 1, "z": 0, "y": 0, "x": 0},
                103: {"t": 2, "z": 0, "y": 0, "x": 0},
                104: {"t": 2, "z": 0, "y": 0, "x": 0},
                105: {"t": 3, "z": 0, "y": 0, "x": 0},
            }
        )
        edges = [
            {"source_id": node_id, "target_id": node_id + 1}
            for node_id in range(5)
        ]
        edges.extend(
            [
                {"source_id": 100, "target_id": 101},
                {"source_id": 100, "target_id": 102},
                {"source_id": 101, "target_id": 103},
                {"source_id": 102, "target_id": 104},
                {"source_id": 103, "target_id": 105},
            ]
        )
        scores = {node_id: 0.1 for node_id in nodes}

        with mock.patch.object(PU, "MAX_REMOVED_FRACTION", 1.0):
            removed, report = PU.select_removed_components(
                nodes,
                edges,
                scores,
                threshold=0.5,
            )

        self.assertEqual(removed, set(range(6)))
        self.assertEqual(report["eligible_components"], 1)
        self.assertEqual(report["selected_components"], 1)


class PuTrainingTests(unittest.TestCase):
    def test_tiny_cpu_fit_is_deterministic_and_finite(self) -> None:
        rng = np.random.default_rng(17)
        features = rng.normal(size=(48, len(PU.FEATURE_NAMES))).astype(
            np.float32
        )
        labels = np.zeros(48, dtype=np.bool_)
        labels[:24] = True
        node_ids = np.arange(48, dtype=np.int64)

        with tempfile.TemporaryDirectory() as temporary:
            cache_path = Path(temporary) / "44b6_tiny.npz"
            np.savez(
                cache_path,
                node_ids=node_ids,
                features=features,
                labels=labels,
            )
            caches = {"44b6_tiny": cache_path}
            patches = (
                mock.patch.object(PU, "EPOCHS", 2),
                mock.patch.object(PU, "POSITIVE_BATCH_SIZE", 8),
                mock.patch.object(PU, "UNLABELED_BATCH_SIZE", 8),
            )
            with patches[0], patches[1], patches[2]:
                first = PU.train_pu_model(
                    caches,
                    ["44b6_tiny"],
                    seed=31,
                    device_name="cpu",
                )
                second = PU.train_pu_model(
                    caches,
                    ["44b6_tiny"],
                    seed=31,
                    device_name="cpu",
                )
                first_scores = PU.predict_probabilities(
                    first,
                    features,
                    "cpu",
                )
                second_scores = PU.predict_probabilities(
                    second,
                    features,
                    "cpu",
                )

        self.assertTrue(np.isfinite(first_scores).all())
        self.assertTrue(np.all((0.0 <= first_scores) & (first_scores <= 1.0)))
        np.testing.assert_array_equal(first_scores, second_scores)


if __name__ == "__main__":
    unittest.main()
