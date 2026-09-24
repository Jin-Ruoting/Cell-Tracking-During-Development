import ast
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import build_kaggle_paired as build
import run_kaggle_paired as paired


def repeat_evidence():
    diagnostic = {"candidate_run": False, "promotion_allowed": False, "observed": {"n": 2, "score": 0.96},
                  "cloud_repeatability": {"csv_bytes_reproduced": True, "score_reproduced": True,
                                          "reference": {"csv_sha256": "frozen", "score": 0.96}},
                  "control_parity": {"actual_csv_sha256": "frozen", "byte_parity": False,
                                     "predictor_source_audit": {"canonical_sha256": paired.cloud.PREDICTOR_SHA256}}}
    run = {"mode": "smoke", "datasets": list(paired.reference.SMOKE_NAMES),
           "reference_sha256": paired.reference.REFERENCE_SHA256,
           "frozen_overrides": paired.reference.FROZEN_OVERRIDES,
           "control_predicted_in_this_run": True, "control_prediction_complete": True,
           "predictions_read_ground_truth": False, "runtime_versions": {"python": "fixed"}, "git_commit": "source"}
    return diagnostic, run


class KagglePairedTests(unittest.TestCase):
    def test_cloud_gate_requires_independent_exact_repeat_without_claiming_server_parity(self):
        diagnostic, run = repeat_evidence()
        self.assertEqual(paired.verify_repeat_evidence(diagnostic, run, "frozen")["score"], 0.96)
        for key in ("csv_bytes_reproduced", "score_reproduced"):
            changed = copy.deepcopy(diagnostic)
            changed["cloud_repeatability"][key] = False
            with self.assertRaises(ValueError):
                paired.verify_repeat_evidence(changed, run, "frozen")
        with self.assertRaises(ValueError):
            paired.verify_repeat_evidence(diagnostic, {**run, "control_predicted_in_this_run": False}, "frozen")
        with self.assertRaises(ValueError):
            paired.verify_repeat_evidence(diagnostic, run, "changed")
        changed = copy.deepcopy(diagnostic)
        changed["observed"]["score"] += 0.001
        with self.assertRaises(ValueError):
            paired.verify_repeat_evidence(changed, run, "frozen")

    def test_anchor_fingerprint_allows_only_id_offset_and_row_order(self):
        fields = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
        rows = [[0, "movie", "node", 4, 0, 1, 2, 3, -1, -1],
                [1, "movie", "node", 9, 1, 1, 2, 4, -1, -1],
                [2, "movie", "edge", -1, -1, -1, -1, -1, 4, 9]]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "graph.csv"

            def fingerprint(data):
                with path.open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
                    writer.writerows(data)
                return paired.graph_fingerprints(path, ["movie"])

            expected = fingerprint(rows)
            shifted = copy.deepcopy(rows)
            shifted[0][3] += 100
            shifted[1][3] += 100
            shifted[2][8] += 100
            shifted[2][9] += 100
            for r in shifted:
                r[0] += 500
            self.assertEqual(fingerprint(list(reversed(shifted))), expected)
            changed = copy.deepcopy(rows)
            changed[1][7] += 0.0001
            self.assertNotEqual(fingerprint(changed), expected)
            changed = copy.deepcopy(rows)
            changed[2][8], changed[2][9] = changed[2][9], changed[2][8]
            self.assertNotEqual(fingerprint(changed), expected)
            changed = copy.deepcopy(rows)
            changed[1][3] += 1
            changed[2][9] += 1
            self.assertNotEqual(fingerprint(changed), expected)
            with self.assertRaisesRegex(ValueError, "Missing"):
                fingerprint([])

    def test_paired_control_cannot_change_after_freezing(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(paired, "WORK", Path(folder)):
            path = Path(folder) / "control/control.csv"
            path.parent.mkdir()
            path.write_text("original bytes")
            digest = paired.reference.stability.file_sha256(path)
            (Path(folder) / "frozen_control.json").write_text(json.dumps({"csv_sha256": digest}))
            paired.frozen_control()
            path.write_text("modified bytes")
            with self.assertRaisesRegex(ValueError, "changed"):
                paired.frozen_control()

    def test_full_requires_same_protocol_technical_smoke_not_a_favorable_smoke_score(self):
        diagnostic, run = repeat_evidence()
        baseline = paired.verify_repeat_evidence(diagnostic, run, "frozen")
        receipt = {"experiment": "E039", "mode": "smoke", "technical_check_passed": True,
                   "protocol_sha256": "protocol", "cloud_baseline": baseline,
                   "development_promotion_passed": False,
                   "control": {"paired_control_frozen_before_candidate": True, "csv_sha256": "frozen"}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "smoke.json"
            path.write_text(json.dumps(receipt))
            build.require_paired_smoke(path, "protocol", baseline)
            with self.assertRaises(ValueError):
                build.require_paired_smoke(path, "changed", baseline)
            with self.assertRaises(ValueError):
                build.require_paired_smoke(None, "protocol", baseline)
            with self.assertRaises(ValueError):
                build.require_paired_smoke(path, "protocol", {**baseline, "csv_sha256": "other"})

    def test_explicit_paired_dispatch_retains_phase_order(self):
        source = build.packaging.bootstrap_source({}, runner_name="run_kaggle_paired.py",
                                                  work_name="e039-development", phases=("control", "peaks", "bridges", "score"))
        tree = ast.parse(source)
        loop = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "phase")
        calls = []
        exec(compile(ast.Module(body=[loop], type_ignores=[]), "phase-order", "exec"),
             {"runner": ["paired"], "run_logged": lambda label, command: calls.append(label)})
        self.assertEqual(calls, ["control", "peaks", "bridges", "score"])
        self.assertIn("run_kaggle_paired.py", source)
        self.assertIn("/kaggle/working/logs/e039-development", source)


if __name__ == "__main__":
    unittest.main()
