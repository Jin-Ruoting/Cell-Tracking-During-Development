import ast
import csv
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

KAGGLE = Path(__file__).resolve().parents[1] / "kaggle"
sys.path.insert(0, str(KAGGLE))
import run_geometric_reference as reference


class ReferenceBoundaryTests(unittest.TestCase):
    def test_adaptation_changes_only_top_level_infrastructure(self):
        source = """from pathlib import Path
WORKING_DIR = Path('.')
def predict():
    WORKING_DIR = Path('untouched')
    return WORKING_DIR
ensure_dependencies(ARTIFACTS)
materialize_inference_repo(ARTIFACTS)
"""
        adapted = reference.adapt_cell(source, {"WORKING_DIR": "Path('/run')"},
                                       ("ensure_dependencies",))
        tree = ast.parse(adapted)
        original = ast.parse(source)
        self.assertEqual(ast.dump(tree.body[2]), ast.dump(original.body[2]))
        self.assertIn("Path('/run')", adapted)
        self.assertNotIn("ensure_dependencies(ARTIFACTS)", adapted)
        self.assertIn("materialize_inference_repo(ARTIFACTS)", adapted)

    def test_missing_or_ambiguous_anchor_fails(self):
        for source in ("OTHER = 1", "WORKING_DIR = 1\nWORKING_DIR = 2"):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "anchors"):
                reference.adapt_cell(source, {"WORKING_DIR": "3"})

    def test_reference_checksum_fails_before_code_is_parsed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.ipynb"
            path.write_text("not even JSON")
            with self.assertRaisesRegex(ValueError, "checksum"):
                reference.read_reference(path)

    def test_changed_control_corpus_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "44b6_wrong.geff").mkdir()
            with self.assertRaisesRegex(ValueError, "corpus"):
                reference.selected_names(path, "smoke")

    def test_worker_paths_preserve_runtime_in_both_execution_branches(self):
        source = 'shard_env = {**os.environ, "PYTHONPATH": "src"}\nenv = {**os.environ, "PYTHONPATH": "src"}'
        adapted = reference.adapt_worker_paths(source)
        import os
        from unittest.mock import patch
        namespace = {"os": os}
        with patch.dict(os.environ, {"PYTHONPATH": "/pinned/runtime"}):
            exec(adapted, namespace)
        for key in ("shard_env", "env"):
            self.assertEqual(namespace[key]["PYTHONPATH"], "src" + os.pathsep + "/pinned/runtime")
        with self.assertRaisesRegex(ValueError, "anchors"):
            reference.adapt_worker_paths('env = {}')


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("pandas"),
                     "Submission audit tests need the server's existing numpy/pandas runtime")
class SubmissionAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.columns = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
        self.rows = [
            [0, "movie", "node", 1, 0, 1, 1, 1, -1, -1],
            [1, "movie", "node", 2, 1, 1, 1, 1, -1, -1],
            [2, "movie", "edge", -1, -1, -1, -1, -1, 1, 2],
        ]

    def audit(self):
        # Import compiled extensions before snapshotting sys.modules. Restoring
        # that dictionary must not unload a just-imported NumPy extension.
        import numpy
        import pandas
        path = self.root / "submission.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.columns)
            writer.writerows(self.rows)
        zarr = types.SimpleNamespace(open=lambda *a, **k: {"0": types.SimpleNamespace(shape=(3, 8, 16, 16))})
        with mock.patch.dict(sys.modules, {"zarr": zarr}):
            return reference.validate_submission(path, self.root, ["movie"])

    def test_valid_graph_passes(self):
        self.assertTrue(self.audit()["passed"])

    def test_volume_boundary_is_exclusive(self):
        self.rows[1][5] = 8
        with self.assertRaisesRegex(ValueError, "out-of-volume"):
            self.audit()

    def test_fractional_coordinate_is_rejected(self):
        self.rows[1][6] = 1.25
        with self.assertRaisesRegex(ValueError, "noninteger"):
            self.audit()

    def test_dangling_edge_is_rejected(self):
        self.rows[2][-1] = 99
        with self.assertRaisesRegex(ValueError, "dangling"):
            self.audit()

    def test_nonconsecutive_edge_is_rejected(self):
        self.rows[1][4] = 2
        with self.assertRaisesRegex(ValueError, "nonconsecutive"):
            self.audit()

    def test_three_children_are_rejected(self):
        self.rows.extend([
            [3, "movie", "node", 3, 1, 1, 2, 1, -1, -1],
            [4, "movie", "node", 4, 1, 1, 3, 1, -1, -1],
            [5, "movie", "edge", -1, -1, -1, -1, -1, 1, 3],
            [6, "movie", "edge", -1, -1, -1, -1, -1, 1, 4],
        ])
        with self.assertRaisesRegex(ValueError, "lineage degree"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
