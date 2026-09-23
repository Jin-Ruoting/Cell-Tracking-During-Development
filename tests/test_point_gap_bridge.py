import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import point_gap_bridge as bridge
from run_point_gap_experiment import append_bridges


class ObservedBridgeTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.arange(40)
        self.points = np.asarray([[t, 5, 8, 4 * t] for t in (0, 1, 2, 4, 5, 6)] +
                                 [[20, 20, 40, i * 8] for i in range(34)])
        self.pairs = np.asarray([(0, 1), (1, 2), (3, 4), (4, 5)])
        self.peaks = np.asarray([[3, 5, 8, 12]])
        self.scores = np.asarray([0.9])

    def run_bridge(self):
        return bridge.bridge_one_frame(self.ids, self.points, self.pairs, self.peaks, self.scores)

    def test_actual_peak_bridges_consecutive_fragments_without_mutating_inputs(self):
        snapshots = [a.copy() for a in (self.ids, self.points, self.pairs, self.peaks, self.scores)]
        proposals, stats = self.run_bridge()
        self.assertEqual(len(proposals), 1)
        item = proposals[0]
        self.assertEqual((item["source_id"], item["target_id"], item["node_id"]), (2, 3, 40))
        self.assertEqual(item["point"], [3, 5, 8, 12])
        self.assertEqual(stats["added_edges"], 2)
        for original, current in zip(snapshots, (self.ids, self.points, self.pairs, self.peaks, self.scores)):
            np.testing.assert_array_equal(original, current)

    def test_missing_or_weak_observation_cannot_create_synthetic_node(self):
        self.scores[:] = 0.49
        self.assertEqual(self.run_bridge()[0], [])
        self.peaks, self.scores = np.empty((0, 4)), np.empty(0)
        self.assertEqual(self.run_bridge()[0], [])

    def test_existing_node_excludes_nearby_peak(self):
        self.points[6] = self.peaks[0]
        self.assertEqual(self.run_bridge()[0], [])

    def test_multiple_plausible_peaks_abstain_instead_of_selecting_by_probability(self):
        self.peaks = np.asarray([[3, 5, 8, 12], [3, 5, 8, 16]])
        self.scores = np.asarray([0.99, 0.6])
        result, stats = self.run_bridge()
        self.assertEqual(result, [])
        self.assertEqual(stats["ambiguous_triples"], 2)

    def test_shared_peak_or_endpoint_cannot_join_two_tracks(self):
        self.points[6:9] = self.points[3:6] + np.asarray([0, 0, 1, 0])
        self.pairs = np.concatenate([self.pairs, [(6, 7), (7, 8)]])
        self.assertEqual(self.run_bridge()[0], [])

    def test_reverse_context_and_division_context_are_rejected(self):
        self.points[5, 3] = 0
        self.assertEqual(self.run_bridge()[0], [])
        self.setUp()
        self.points[6] = [2, 5, 8, 10]
        self.pairs = np.concatenate([self.pairs, [(1, 6)]])
        self.assertEqual(self.run_bridge()[0], [])

    def test_budget_and_invalid_original_graph_fail_safely(self):
        self.ids, self.points = self.ids[:6], self.points[:6]
        self.assertEqual(self.run_bridge()[0], [])
        self.pairs[0] = [0, 999]
        with self.assertRaisesRegex(ValueError, "Dangling"):
            self.run_bridge()

    def test_append_preserves_original_rows_and_only_adds_supported_path(self):
        import pandas as pd
        columns = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
        rows = [[i, "movie", "node", i, *p, -1, -1] for i, p in zip(self.ids, self.points)]
        rows += [[len(rows) + i, "movie", "edge", -1, -1, -1, -1, -1, a, b]
                 for i, (a, b) in enumerate(self.pairs)]
        group = pd.DataFrame(rows, columns=columns)
        result = append_bridges(group, self.run_bridge()[0])
        pd.testing.assert_frame_equal(result.iloc[:len(group)], group)
        self.assertEqual(result.iloc[-2:][["source_id", "target_id"]].values.tolist(), [[2, 40], [40, 3]])

    def test_cache_rejects_missing_frames_duplicate_coordinates_and_nonfinite_values(self):
        bridge.validate_peaks(self.peaks, self.scores, np.arange(7), (7, 32, 64, 64))
        for coords, scores, frames in ((self.peaks, self.scores, np.arange(6)),
                                      (np.repeat(self.peaks, 2, axis=0), np.repeat(self.scores, 2), np.arange(7)),
                                      (self.peaks, np.asarray([np.nan]), np.arange(7))):
            with self.assertRaises(ValueError):
                bridge.validate_peaks(coords, scores, frames, (7, 32, 64, 64))


if __name__ == "__main__":
    unittest.main()
