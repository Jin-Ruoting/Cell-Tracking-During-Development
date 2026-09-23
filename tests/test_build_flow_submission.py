import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import build_flow_submission as builder


def passing_report():
    return {
        "gates": dict.fromkeys(builder.REQUIRED_GATES, True), "promotion_passed": True,
        "minimum_pooled_delta": 0.001,
        "groups": {"all": {
            "control": {"score": builder.flow.E029_SCORE},
            "candidate": {"score": builder.flow.E029_SCORE + 0.002},
            "delta": {"score": 0.002, "adj_edge_jaccard": 0.001}},
            **{key: {"delta": {"score": 0.002}} for key in ("44b6", "6bba", "half0", "half1")}},
        "paired": {"wins": 40, "losses": 24, "median_delta": 0.001},
    }


class FlowPackagingTests(unittest.TestCase):
    def test_every_frozen_gate_is_required_without_a_tradeoff_switch(self):
        report = passing_report()
        self.assertEqual(builder.check_report(report)["selection_policy"], "all_frozen_e031_gates")
        for gate in builder.REQUIRED_GATES:
            failed = copy.deepcopy(report)
            failed["gates"][gate] = False
            with self.subTest(gate=gate), self.assertRaisesRegex(ValueError, "Every frozen"):
                builder.check_report(failed)

    def test_rejects_e029_old_gate_contract_and_changed_threshold(self):
        for changes in ({"gates": dict.fromkeys(builder.geometric.REQUIRED_GATES, True)},
                        {"minimum_pooled_delta": 0.0}, {"promotion_passed": False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                builder.check_report({**passing_report(), **changes})

    def test_gate_booleans_cannot_mask_a_regression_or_nonfinite_result(self):
        for key, field, value in (("all", "score", 0.0009),
                                  ("all", "adj_edge_jaccard", -0.0001),
                                  ("44b6", "score", 0.0), ("half1", "score", -0.001),
                                  ("6bba", "score", float("nan"))):
            report = passing_report()
            report["groups"][key]["delta"][field] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                builder.check_report(report)
        report = passing_report()
        report["groups"]["all"]["candidate"]["score"] = float("nan")
        with self.assertRaises(ValueError):
            builder.check_report(report)

    def test_rejects_wrong_control_and_affected_median(self):
        report = passing_report()
        report["groups"]["all"]["control"]["score"] = 0.9014331472
        with self.assertRaisesRegex(ValueError, "control parity"):
            builder.check_report(report)
        report = passing_report()
        report["paired"]["median_delta"] = 0
        with self.assertRaises(ValueError):
            builder.check_report(report)

    def test_smoke_evidence_cannot_build_a_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_manifest.json").write_text(json.dumps({"mode": "smoke"}))
            with self.assertRaisesRegex(ValueError, "provenance"):
                builder.require_promotion(root, "def motion_relink_edges(): pass")

    def test_package_ships_only_selected_flow_and_keeps_output_audits(self):
        sources = [f"cell_number = {index}" for index in range(12)]
        sources[5] = "def motion_relink_edges(): return 'old'\nwrite_test_submission('base')"
        sources[7] = "raise RuntimeError('author_label_validator')"
        sources[10] = "raise RuntimeError('author_parameter_sweep')"
        notebook = builder.make_notebook(sources, "def motion_relink_edges(): return 'selected_flow'", {})
        text = json.dumps(notebook)
        for omitted in ("return 'old'", "author_label_validator", "author_parameter_sweep", "e029_output_audit"):
            self.assertNotIn(omitted, text)
        for required in ("selected_flow", "e031_output_audit", "normalize_export_boundary",
                         "validate_submission", "flow_function_sha256", "flow_frames"):
            self.assertIn(required, text)
        self.assertEqual(len(notebook["cells"]), 9)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])


if __name__ == "__main__":
    unittest.main()
