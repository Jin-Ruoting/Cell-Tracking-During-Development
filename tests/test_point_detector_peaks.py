import sys
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
from capture_point_detector_peaks import extract_peaks, normalize_volume


class PointInputTests(unittest.TestCase):
    def test_strided_sampling_metadata_quantiles_and_no_upper_clip(self):
        raw = np.full((4, 8, 8), 20, dtype=np.uint16)
        raw[:, ::4, ::4] = 100
        raw[0, 0, 0] = 0
        result = normalize_volume(raw, 10, 50)
        self.assertEqual(result.shape, (4, 2, 2))
        self.assertEqual(result[0, 0, 0], 0.0)
        self.assertAlmostEqual(float(result[1, 0, 0]), 2.25, places=5)
        with self.assertRaises(ValueError):
            normalize_volume(raw, 10, 10)

    def test_nms_time_and_original_voxel_scale(self):
        logits = torch.full((1, 6, 6, 6), -10.0)
        logits[0, 1, 1, 1] = 4.0
        logits[0, 1, 1, 2] = 3.0  # Adjacent lower peak is suppressed.
        logits[0, 4, 4, 4] = 0.0
        coords, scores = extract_peaks(logits, 17)
        np.testing.assert_array_equal(coords, [[17, 1, 4, 4], [17, 4, 16, 16]])
        np.testing.assert_allclose(scores, [torch.sigmoid(torch.tensor(4.0)).item(), 0.5])
        empty, empty_scores = extract_peaks(torch.full((1, 3, 3, 3), -10.0), 0)
        self.assertEqual(empty.shape, (0, 4))
        self.assertEqual(empty_scores.shape, (0,))


if __name__ == "__main__":
    unittest.main()
