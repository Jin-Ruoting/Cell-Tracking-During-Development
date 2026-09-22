import ast
import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
