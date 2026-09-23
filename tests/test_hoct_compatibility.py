import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
from probe_hoct_consensus import rasterize, snap_nodes


class GeometryTests(unittest.TestCase):
    def test_physical_radius_and_lost_node_rejection(self):
        points = np.array([[0, 3, 12, 12]], dtype=float)
        labels = rasterize(points, (1, 7, 25, 25))
        self.assertEqual(labels[0, 3, 12, 19], 1)  # 2.84 um in x
        self.assertEqual(labels[0, 5, 12, 12], 0)  # 3.25 um in z
        with self.assertRaisesRegex(ValueError, "lost an input node"):
            rasterize(np.repeat(points, 2, axis=0), labels.shape)

    def test_mapping_uses_time_and_rejects_ambiguity_or_distance(self):
        original = np.array([[0, 3, 12, 12], [1, 3, 12, 12]], dtype=float)
        mapping, distance = snap_nodes(original[::-1], original)
        self.assertEqual(mapping.tolist(), [1, 0])
        self.assertEqual(distance, 0)
        for invalid in (np.repeat(original[:1], 2, axis=0), original + [0, 2, 0, 0]):
            with self.assertRaisesRegex(ValueError, "distant or ambiguous"):
                snap_nodes(invalid, original)


if __name__ == "__main__":
    unittest.main()
