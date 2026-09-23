import ast
import importlib.util
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import capture_frozen_detection_peaks as capture


@unittest.skipUnless(os.environ.get("BIOHUB_E029_PREDICTOR_TEST"), "Pinned E029 runtime required")
class FrozenCaptureSourceTests(unittest.TestCase):
    def test_removes_only_edge_loop_and_preserves_detection_calls(self):
        path = Path(os.environ["BIOHUB_E029_PREDICTOR_TEST"])
        self.assertEqual(capture.reference.stability.file_sha256(path), capture.PREDICTOR_SHA256)
        original = path.read_text()
        # The frozen source was materialized under the raw run directory.
        raw = path.parents[2]
        changed = capture.capture_source(original, raw, Path("/new-run"))
        compile(changed, "capture", "exec")
        before = ast.parse(original)
        after = ast.parse(changed)
        for a, b in zip(before.body, after.body):
            if isinstance(a, ast.FunctionDef) and a.name == "predict_video":
                continue
            self.assertEqual(ast.dump(a), ast.dump(b))
        self.assertNotIn(str(raw), changed)
        calls = lambda tree, name: [ast.dump(n) for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]
        original_calls = calls(before, "_detect_cells_pooled")
        changed_calls = calls(after, "_detect_cells_pooled")
        self.assertEqual(len(changed_calls), len(original_calls) + 1)
        self.assertTrue(all(call in changed_calls for call in original_calls))
        function = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == "predict_video")
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr == "predict_edges" for n in ast.walk(function)))

    def test_source_drift_does_not_silently_skip_instrumentation(self):
        for source in ("pass", "def predict_video():\n    return [], []"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                capture.instrument_peaks(source)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "Array runtime required")
class PeakContractTests(unittest.TestCase):
    def test_rejects_bad_scale_probabilities_duplicates_and_coverage(self):
        import numpy as np
        coords = np.array([[0, 2, 4, 8], [1, 3, 8, 12]], dtype=np.int16)
        scores = np.array([0.8, 0.9], dtype=np.float32)
        capture.validate_arrays(coords, scores, [0, 1], [2, 8, 16, 16])
        for points, probs, frames in (
            (coords + [0, 0, 0, 8], scores, [0, 1]),
            (coords, np.array([0.1, 0.9]), [0, 1]),
            (coords, np.array([np.nan, 0.9]), [0, 1]),
            (coords[[0, 0]], scores, [0, 1]),
            (coords, scores, [0]),
        ):
            with self.subTest(points=points, probs=probs, frames=frames), self.assertRaises(ValueError):
                capture.validate_arrays(points, probs, frames, [2, 8, 16, 16])


if __name__ == "__main__":
    unittest.main()
