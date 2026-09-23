import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))
import audit_motion_ema_kernel_output as audit
import build_motion_ema_submission as builder


class EmaSubmissionTests(unittest.TestCase):
    def test_only_ema_function_and_runtime_audit_are_packaged(self):
        sources = [f"cell_number = {index}" for index in range(12)]
        sources[5] = "def motion_relink_edges(): return 'old'\nwrite_test_submission('base')"
        sources[7] = "raise RuntimeError('author_label_validator')"
        sources[10] = "raise RuntimeError('author_parameter_sweep')"
        notebook = builder.make_notebook(sources, "def motion_relink_edges(): return 'fixed_ema'", {})
        text = json.dumps(notebook)
        for omitted in ("return 'old'", "author_label_validator", "author_parameter_sweep", "FLOW_SEED_K", "e029_output_audit"):
            self.assertNotIn(omitted, text)
        for needed in ("fixed_ema", "e035_output_audit", "normalize_export_boundary", "validate_submission", "ema_changed_velocities"):
            self.assertIn(needed, text)
        self.assertEqual(len(notebook["cells"]), 9)

    def test_smoke_cannot_authorize_a_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "run_manifest.json").write_text(json.dumps({"mode": "smoke"}))
            with self.assertRaisesRegex(ValueError, "provenance"):
                builder.require_promotion(path, "def motion_relink_edges(): pass")

    def test_receipt_rejects_changed_parameters_inactive_ema_or_validation(self):
        proof = {"selection_policy": "all_frozen_e035_gates", "motion_function_sha256": "evaluated",
                 "outside_author_selection": {"passed": True}}
        receipt = {
            "external_reference_sha256": builder.reference.REFERENCE_SHA256,
            "frozen_overrides": builder.reference.FROZEN_OVERRIDES,
            "motion_function_sha256": "evaluated", "ema_alpha": 0.4, "velocity_weight": 0.25,
            "offline_promotion": proof, "status": "kernel_output_audited_public_score_pending", "passed": True,
            "ema_updates": 100, "ema_changed_velocities": 90,
            "effective_environment": {"BIOHUB_VALIDATOR_ENABLE": "0", **{
                "BIOHUB_" + key: str(value) for key, value in builder.reference.FROZEN_OVERRIDES.items()}},
        }
        audit.check_receipt(receipt, proof)
        for key, value in (("ema_alpha", 0.5), ("velocity_weight", 0.5), ("motion_function_sha256", "other"),
                           ("ema_updates", 89), ("ema_changed_velocities", 0), ("offline_promotion", {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.check_receipt({**receipt, key: value}, proof)
        changed = copy.deepcopy(receipt)
        changed["effective_environment"]["BIOHUB_VALIDATOR_ENABLE"] = "1"
        with self.assertRaises(ValueError):
            audit.check_receipt(changed, proof)


if __name__ == "__main__":
    unittest.main()
