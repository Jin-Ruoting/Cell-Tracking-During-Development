import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import audit_observed_peak_coverage as coverage


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("scipy"), "Array runtime required")
class CoverageBoundaryTests(unittest.TestCase):
    def test_respects_time_score_exclusion_scale_and_many_to_one_boundary(self):
        import numpy as np
        nodes = [{"node_id": 0, "t": 0, "z": 1, "y": 10, "x": 10}]
        truth = [{"node_id": i, "t": t, "z": 1, "y": 10, "x": x}
                 for i, t, x in ((1, 0, 10), (2, 0, 40), (3, 0, 41), (4, 1, 40), (5, 0, 100))]
        low = np.array([[0, 1, 10, 10], [0, 1, 10, 40], [0, 1, 10, 100]])
        scores = np.array([0.99, 0.99, 0.6])
        high = coverage.potential_coverage(truth, {1, 2, 3, 4, 5}, nodes, low, scores, 0.965)
        self.assertEqual(high, {2, 3})
        weak = coverage.potential_coverage(truth, {1, 2, 3, 4, 5}, nodes, low, scores, 0.5)
        self.assertEqual(weak, {2, 3, 5})


if __name__ == "__main__":
    unittest.main()
