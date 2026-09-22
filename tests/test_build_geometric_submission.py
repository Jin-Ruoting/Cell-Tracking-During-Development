import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import build_geometric_submission as builder


class SubmissionPackageTests(unittest.TestCase):
    def test_component_tradeoff_requires_an_explicit_policy_and_keeps_failed_gate(self):
        gates = dict.fromkeys(builder.REQUIRED_GATES, True)
        gates["adjusted_edge_not_regressed"] = False
        report = {"gates": gates, "promotion_passed": False,
                  "groups": {"all": {"delta": {"division_jaccard": 0.08, "adj_edge_jaccard": -0.001}}}}
        with self.assertRaises(ValueError):
            builder.classify_selection(report)
        decision = builder.classify_selection(report, True)
        self.assertFalse(decision["original_frozen_gate_passed"])
        self.assertEqual(decision["failed_frozen_gates"], ["adjusted_edge_not_regressed"])
        gates["both_embryos_positive"] = False
        with self.assertRaises(ValueError):
            builder.classify_selection(report, True)

    def test_smoke_receipt_cannot_authorize_a_package(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "run_manifest.json").write_text(json.dumps({"mode": "smoke"}))
            with self.assertRaisesRegex(ValueError, "provenance"):
                builder.require_promotion(path)

    def test_presentation_cleanup_preserves_configuration_and_assertions(self):
        source = 'value = 0.965\nassert value > 0\nprint("old score")\n'
        tree = ast.parse(builder.remove_status_prints(source))
        self.assertEqual([type(node) for node in tree.body], [ast.Assign, ast.Assert])

    def test_package_excludes_reference_validator_and_sweep(self):
        sources = [f"cell_number = {index}" for index in range(12)]
        sources[7] = "raise RuntimeError('training-label validator executed')"
        sources[10] = "raise RuntimeError('parameter sweep executed')"
        notebook = builder.make_notebook(sources, {"test": True})
        text = json.dumps(notebook)
        self.assertNotIn("training-label validator executed", text)
        self.assertNotIn("parameter sweep executed", text)
        self.assertIn("validate_submission", text)
        self.assertEqual(len(notebook["cells"]), 9)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])


if __name__ == "__main__":
    unittest.main()
